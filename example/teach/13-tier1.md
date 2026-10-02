# Step 13 — Tier 1 定档

**你要写的**：不动代码。**决定**：正式模型的规模。
**验收**：`logs/dev/02-hardware-perf.md` 里有一条 Tier 1 阶梯，
且阶梯上的每个数字都有 `Predicted` 和 `Measured` 两列。

---

## 13.1 先测吞吐，再定参数（顺序不能反）

`AGENTS.md` 写死了：**参数量、序列长度、数据量是耦合的，先拍数字再测是本末倒置。**

实验版的 Tier 0 跑完也不能用来定档 —— 3.2M / seq 128 上，
**手写 torch 比 transformers 快 5%**（47.5k vs 45.2k tok/s）。
这个结论在 50M / seq 512 上大概率反过来。

所以 step 13 的第一件事是**测阶梯**，不是选模型。

## 13.2 候选（`SPEC` 4，不预先承诺任何一个）

| | hidden | layers | Q/KV | head_dim | inter | ctx | 参数量 |
|---|---|---|---|---|---|---|---|
| Tier 1 | 512 | 8 | 8/4 | 64 | 1408 | 512 | ~50M |
| Tier 2 | 768 | 12 | 12/4 | 64 | 2048 | 512 | ~150M |

**行1 / 行2 两个 base 都要在阶梯上各测一遍** —— 实验版已经证明这两者的
相对快慢会随规模变化。

## 13.3 已知约束（`AGENTS.md` + `logs/dev/02-hardware-perf.md`）

| 约束 | 依据 | 阶梯要验证的 |
|---|---|---|
| `max_seq_len ≤ 512` | attention O(T²)，seq 1024 实测比 512 慢 18% | seq 512 的 tok/s |
| `batch 8` 就够 | B=8 vs B=16 吞吐 indistinguishable，B8 省一半显存 | B=4/8/16 三点 |
| 不开 gradient checkpointing | 本机慢 18-23%，只省 3GB | 有/无各测一次 |
| packing 优于 padding | 高 1.71x | 固定对照 |
| 瓶颈是 attention backward | ROCm 在 gfx1150 上没有高效 backward kernel | 逐 kernel 测 |
| run-to-run 方差可达 ±20% | 共享核显 | **同配置跑 3 次报离散度** |

## 13.4 测吞吐前必做的四件事（`notes/05-measurement.md`）

1. `rocm-smi --showuse` 确认 GPU use 接近 0%
   ——「没进程」不等于「卡是空的」，VRAM 可能被桌面端占着
2. 同一配置 **3 次**，报离散度，不报单次
3. warmup ≥ 50 步（前 ~25 步落在预热相）
4. benchmark ≥ 400 步

**同时报 train-only 和 wall-clock**，分子分母范围一致。

## 13.5 Tier 1 要重跑的对拍

| 对拍 | 为什么要在新规模上再跑一次 |
|---|---|
| C1 logits `1e-5` | Tier 0 上是 6.91e-06，**余量只有 1.4 倍**。规模变大后 fp32 累加顺序的差异会放大 |
| C2/C3 loss `< 1%` | 实测 1e-7 量级，余量极大，主要是防回归 |
| C4 resume | 与规模无关，但换数据管线后要重验 |
| rep-4 | **这次才是真判断** —— 几百 token 的生成 |

## 13.6 smoke 阶段的数字哪些能带走，哪些不能

| 能带走 | 不能带走 |
|---|---|
| 全部对拍阈值（C1-C5 的量级） | 吞吐（3.2M 上手写更快，正式规模未知） |
| loss 曲线的形状（A2D vs Direct 的相对关系） | 任何 loss 的绝对值（20 万 token vs 332M token） |
| 「A2D 的 eval loss 初始比 AR 高」这个机制 | 「AR 权重值 0.67 nat」这个数 |
| rep-4 的排序（A2D < Direct < LoRA） | rep-4 的具体值（32 token 样本） |

## 13.7 定档的输出

写进 `logs/dev/02-hardware-perf.md`：

```
model            hidden  layers  Q/KV  ctx  batch  params
tier1            512     8       8/4   512   8      50M
train-only       ______ tok/s   (3 runs: __ / __ / __, 离散度 __%)
wall-clock       ______ tok/s
peak             ______ GB
332M token 需要  __ h (train-only) / __ h (wall)
```

然后按 `AGENTS.md` 的路线：**先有 AR baseline，再谈 DLM**。
没有可用权重时无法判断 DLM 的输出是「变差了」还是「本来就差」。