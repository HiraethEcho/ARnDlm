# 本机实测性能 (Radeon 880M / gfx1150)

> **本文件从 `~/Projects/dlm/devlog/01-hardware-perf.md` 继承 (2026-09-30 实测)。**
> 机器和 torch 版本与本项目一致 (torch 2.12.0+rocm10.0.0), 所以数字直接可用。
> 但 **`~/Projects/dlm` 已被冻结, benchmark 脚本没有一起搬过来** ——
> 需要重测时得在本项目里重建 `test/` 里的性能脚本, 或者回去跑 dlm 的。
>
> **换机器 / 换 torch 版本后本文件全部作废。** 测量方法论见 `notes/05-measurement.md`。


> **2026-09-30 晚: 换过环境, 已核对。** 从系统包 (torch 2.14.0+hip / ROCm 7.2)
> 换成 `.venv` (torch 2.12.0+rocm10.0.0 / hip 7.15)。
>
> 我一度以为新环境慢 27% —— **那是测量错误** (单次 10 迭代, 预热不够)。
> 用同一脚本、充分预热、5 轮取最快, 两个环境**等价**:
>
> ```
>              系统(ROCm7.2)   .venv(ROCm10.0)   差
>   1024^3        5.41            5.50          +1.7%
>   2048^3        7.38            7.19          -2.6%
>   4096^3        6.82            6.51          -4.5%
> ```
>
> 真实训练负载上方向甚至不一致 (Qwen3-0.6B 在 .venv 反而快 11%)。
> **本文件的数字在两个环境都成立, 不需要重建基线。**

2026-09-30 实测 (旧环境: torch 2.14.0+hip, ROCm 7.2)。机器: Arch Linux, Ryzen AI 9 H 365, Radeon 880M (gfx1150),
32GB RAM, torch 2.14.0+hip (ROCm 7.2.53211), bf16。

所有数字由 `test/` 下的脚本测得, 重跑命令见 `test/README.md`。

> **测量方法 (2026-09-30 重测后)**
> 每个脚本跑 **5 轮取最快**, 且 `iters` 会**自动定标**到「每轮至少 50ms」。
> 理由: 快的 kernel 测不准 —— q_proj 一次 1.68ms, 只跑 8 次的话采样期
> 14ms, CPU 调度抖动就占 62%。定标后 spread 从 20-137% 降到 2-8%。
>
> **每行都带 spread 列。spread > 20% 的数字不要相信。**
> 我自己踩过: 把 5.36 vs 7.33 的噪声读成了「新环境慢 27%」, 还写进了文件。
> 参见 `notes/05-measurement.md`。

硬件两个上限 (test/01-bw-and-matmul.py):

```
  GTT 带宽 (copy 1024MB)      : 75.6 GB/s    (read-only sum: 92.7 GB/s)
  matmul 峰值                 : 8.6 TFLOPS @ 2048,  7.6 @ 4096,  2.7 @ 8192
  训练实际达成                : ~1.7 TFLOPS  (峰值的 20%)
  3090 对照: 936 GB/s, 71 TFLOPS
```

8192³ 掉到 2.67 TFLOPS 且 spread 18% —— 操作数 402MB 装不进 L2,
那个数字不可信, 峰值看 2048-4096 区间。

---

## 结论: 瓶颈是 attention 的 backward kernel

**不是带宽, 不是算力, 不是 CPU, 也不是 backend 选择。**

ROCm 在 gfx1150 上没有 attention 的高效 backward kernel:
**forward 12.5 ms, backward 105 ms —— 反向是前向的 8.4 倍。** (CUDA 上这个比值约 2x)

它只跑到 **0.25 TFLOPS**, 而同一个 step 里的 linear GEMM 跑到 **5.0-7.8 TFLOPS**
(接近 6.8 的峰值)。**差 25 倍。**

### 1. 各部分实测效率 (minimind-3, B16 T768, 双向)

```
  组件                        ms      效率            判定
─────────────────────────────────────────────────────────────────
  q_proj  [12288,768]x[768,768] 1.68    8.60 TFLOPS   OK  (spread 43%)
  k_proj  [12288,768]x[768,384] 0.95    7.59 TFLOPS   OK
  v_proj  [12288,768]x[768,384] 0.98    7.43 TFLOPS   OK
  o_proj  [12288,768]x[768,768] 1.66    8.73 TFLOPS   OK
  gate    [12288,768]x[768,2432] 5.63   8.16 TFLOPS   OK
  up      [12288,768]x[768,2432] 5.85   7.85 TFLOPS   OK
  down    [12288,2432]x[2432,768] 4.97   9.23 TFLOPS   OK
  lm_head [12288,768]x[768,6400] 17.56  6.88 TFLOPS   OK
─────────────────────────────────────────────────────────────────
  SDPA forward  (1 层)          10.9    2.65 TFLOPS   慢
  SDPA backward (1 层)          97.6    0.28 TFLOPS   ← 瓶颈  (spread 4%)
```

linear GEMM 全部接近 6.8-8 的峰值。**它们不是问题。**

**attention 只占 12.9% 的 FLOPs (0.23 / 1.80 TFLOPs), 却吃掉 36% 的 step 时间。**
单位 FLOP 的代价是 linear GEMM 的约 25 倍。

时间账 (实测 step = 621 ms fwd + 2271 ms bwd):

```
fwd  attention   8 × 12.5       =  100 ms  (16%)
     linear GEMM 8 × 26.1 + 17  =  226 ms  (36%)
     elementwise / norm / RoPE  =  295 ms  (48%)

bwd  attention   8 × 105.0      =  840 ms  (37%)  ← 最大单项
     linear GEMM 2 × 209        =  418 ms  (18%)
     elementwise 反向            ≈ 1013 ms  (45%)
```

**attention 合计约 940 ms / 2591 ms = 36% 的 step 时间。**

### 2. SDPA backend 对比 — 换 backend 没用

```
backend                  fwd ms   fwd TF   fwd+bwd ms   fwd+bwd TF   spread
DEFAULT                    10.95    2.65        108.53       0.27      22%
FLASH_ATTENTION            13.35    2.17        126.97       0.23      11%
EFFICIENT_ATTENTION        11.63    2.49        132.26       0.22       6%
CUDNN_ATTENTION            FAIL — torch-opt-rocm 未编译 cuDNN attention
MATH                       125.52   0.23        279.20       0.10       8%
```

**三个 backend 的 backward 都比 forward 慢 8-10 倍** (108-132 ms vs 11-13 ms)。
DEFAULT 的 22% spread 偏大, 但量级结论不受影响 —— 问题在 backward kernel 本身,
不是选择问题。`flash_attn` 包也没装。

### 3. 因果 vs 双向

```
is_causal=True    fwd+bwd  94.07 ms   0.31 TFLOPS
is_causal=False   fwd+bwd 131.01 ms   0.22 TFLOPS
```

**双向 (DLM 模式) 比因果 (AR 模式) 慢 39%。** 这是 A2D 的固定成本。

---

## 带宽不是瓶颈 (实测反证)

小 kernel 的实测带宽, 张量 `[16,768,768]` bf16 = 18 MB:

```
op                          ms     GB/s    占 78GB/s
silu                        0.435    86.7       111%   ← 已经超过拷贝带宽 (L2 命中)
x*x                         0.396    95.4       122%
x + 1.0 (broadcast)         0.392    96.4       124%
x.sum(-1)                   0.220   171.7       220%
rms_norm                    1.070    35.3        45%
x.transpose(1,2).contig     2.004    18.8        24%   ← 只有这个真的慢
────────────────────────────────────────────
sum 1 GiB (带宽上限参照)    10.231   104.9       135%
```

**elementwise kernel 已经跑满甚至超过拷贝带宽。** 唯一的例外是
`transpose().contiguous()` (18.8 GB/s) —— 那是 strided 访问, 但量小。

---

## ⚠️ 更正: 2026-09-30 早先的判断是错的

早先根据两个现象判断「瓶颈是带宽」:

1. **batch 扫描平坦** (B=1→64, tok/s 只涨 30%)
2. **bwd/fwd = 3.66x** (理论 2.0x)

**这个推论错了。** 正确的解释是 attention backward 本身极慢, 而它的占比
随 batch 增大而上升 (更多 token → attention 的 T² 项占比更高) → 抵消了
大 batch 带来的 GEMM 效率提升 → 看起来「平坦」。

旧项目 `MEASUREMENT.md` 的判断 (「瓶颈是 kernel 效率, 不是带宽」)
才是对的。它当时用「权重只占 202 MB/s」来论证, 那个论证也不完整
(没算 activation), 但**结论正确**。

**教训: 「吞吐对 batch 不敏感」不能推出「带宽受限」。**
需要看具体是哪个 kernel 慢。

---

## 实际推论: 序列长度比 FLOPs 更重要

attention backward ∝ T², 其余部分 ∝ T。所以缩短序列的收益远大于 FLOPs 预期。

```
固定 B=16, 变 T (spread 2-7%):
  T=  128    420 ms   4882 tok/s     2.6% attn
  T=  256    807 ms   5075 tok/s     5.1%
  T=  340   1083 ms   5021 tok/s     6.6%
  T=  512   1610 ms   5089 tok/s     9.6%
  T=  768   2872 ms   4279 tok/s    13.8%
  T= 1024   3906 ms   4195 tok/s    17.6%
```

**T=128..512 吞吐持平, 到 768 才掉 (约 16%)。** 拐点在 512-768 之间。

> ⚠️ 这条结论在重测后**变了**。早先的噪声数据看起来像
> 「T=256→512 掉 37%」—— 那是测量误差。教训: 不要把硬编码的百分比
> 写进脚本注释, 要么实算, 要么标日期。



---

## CPU / tokenizer — 已排除

Qwen3 tokenizer 在 `sft_t2t_mini.jsonl` 实测:

```
批量 (单线程)     : 1,155,009 tok/s
单条延迟         : 0.3 ms  → 串行上限 2,977 tok/s
GPU 需要         : ~4,000 tok/s
本机 core 数     : 20
```

**一个 core 就够喂饱 GPU。** 但仍应离线预 tokenize 成
`tokens.bin` + `doc_lengths.npy` (mmap 读, 避免每步解码开销)。

---

## Gradient checkpointing — 本机上纯亏

```
                  无 ckpt    有 ckpt   变化
minimind-3 69M   2591 ms    3805 ms   +47% 慢
peak memory       4.6 GB     1.5 GB   -3.1 GB
```

**和 CUDA 上的直觉相反。** CUDA 上是「省显存换时间」, 这里是「花时间省显存」。
24GB 放得下 4.6GB 的 69M 模型 → **小模型一律别开。**
Qwen3-0.6B @ B2 已经占 9.9GB, 大模型才考虑。

---

## 时间速查

多轮实测 (test/08-train-step-bench.py, 双向 attention)。
**run-to-run 方差可达 ±20%** —— 小矩阵和短 benchmark 尤其不稳。

> ⚠️ **2026-10-01 补充: ±20% 只在「卡真的空着」时成立。**
> 这张 880M 是核显, 桌面合成器 (niri + Brave) 和另一个 pi 会话都共享它。
> 实测同一配置三次跑出 8,824 / 6,054 / 3,251 tok/s —— **2.7 倍离散度**。
> 测之前先确认没有别的 GPU 任务, 详见 `01-local-state.md` 的「吞吐冒烟测试」。

```
模型                        params    tok/s    20M tok   peakGB
minimind-3  H768 L8  V6400   64M      3,700-4,700   70-90m   4.6
H768 L16    V32000           139M     2,115         158m    11.7
H1024 L16   V32000           244M     1,348         247m    19.0
Qwen2.5-0.5B (近似)         362M       779         428m    12.8
Qwen3-0.6B   (近似)         524M       499         667m     9.0
```

加上 grad-checkpoint 的代价 (小模型):
```
minimind-3    4,163 → 3,204 tok/s   (-23%)   4.6 GB → 1.5 GB
H1024 L16    1,348 → 1,101 tok/s   (-18%)  19.0 GB → 7.8 GB
```

**5 小时预算 ≈ 55-75M tokens。**

minimind 官方 `pretrain_t2t_mini` 是 **332M tokens** → 本机 **19-25 小时 1 epoch**。
**照搬官方配置会严重超时, 必须缩数据。**

`max_seq_len` 的影响 (test/04, B=16):

```
 T=128   7,744 tok/s      T=512   4,744 tok/s
 T=256   8,099 tok/s      T=768   3,998 tok/s
 T=340   5,345 tok/s      T=1024  3,945 tok/s
```

**T=256 → 512 吞吐掉 41%, 而 FLOPs 只涨 1 倍。** 短序列是硬约束。
建议 `max_seq_len` ≤ 512。

---

## 实操

- **不开 gradient checkpointing** (小模型慢 18-23%, 只省 3GB)
- **batch 16 就够** (B=1 到 B=64 吞吐差 30%, B=16 已接近峰值)
- **max_seq_len ≤ 512** (attention 是 O(T²), 短序列收益远超 FLOPs 预期)
- 一次预 tokenize, 存 uint16 `.bin` + `doc_lengths.npy`
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` — 已在 `dev.zsh`
- `MIOPEN_FIND_MODE=FAST` — 已在 `dev.zsh`
- **报告吞吐必须同时给 train-only 和 wall-clock 两个数** (见 `notes/05-measurement.md`)
- **重跑 `test/` 验证**: 换 torch/ROCm 版本或换硬件后数字会变,
  变了说明之前的结论要重新检查

## 跑测试

```sh
cd /home/revo/Projects/dlm/test
python 01-bw-and-matmul.py        # 带宽 + matmul 峰值
python 05-sdpa-attention.py       # ← 最关键, 瓶颈在这
```

索引见 `test/README.md`。

