# SPEC — ARnDlm

> **状态：待审核。** 这份是契约，不是进度汇报。进度看 `logs/dev/01-local-state.md`，
> 步骤和验收看 `ROADMAP.md`，规则看 `AGENTS.md`。
>
> 审核要点：§3 的路线划分、§5 的环境契约、§6 的验收标准、§8 的不做什么。
> 标 🔴 的是我做了决定、你可能想推翻的地方。

---

## 1. 这个问题

**在同一套 tokenizer / vocab / 数据集 / 环境下，把自回归（AR）和扩散（DLM）
放到可比的位置上，用实测数字回答「DLM 到底行不行」。**

## 2. 非目标


| 不做 | 为什么 |
|---|---|
| 追求 SOTA 或刷 benchmark | 目标是对比和理解，不是比赛 |
| 多卡 / 分布式 | 单卡集显，deepspeed 装了也没用 |
| 多模态 | 纯文本 |
| 把 dllm 改造成更好用的库 | 只打最小 patch，其余照上游用 |
| 上传 HuggingFace / 发布权重 | 私人学习项目 |

## 3. 架构

### 3.1 七条路线

```
base ──> utils ──> transformer ──┐
  │                  │           ├──> dllm-a2d ──> dllm
  └──> minimind ─────┴──> a2d ────┘
```

依赖**单向**，永不反向。`a2d` 从 `transformer` 或 `minimind` 拿 AR 权重，
但**不 import 它们的训练循环** —— 否则 A2D 的对比就被污染了。

| 目录 | 是什么 | 路线 |
|---|---|---|
| `base/` | tokenizer、dataset、packing、masking、objective、config | 公用底座 |
| `utils/` | logger、web 展示、device、seed、checkpoint、metrics | 公用底座 |
| `transformer/` | 调用 `transformers` 包训练 + LoRA 后训练 | **T1a** |
| `minimind/` | 原生 pytorch 训练 + 后训练 | **T1b** |
| `a2d/` | 用 **minimind-dlm** 思路手动做 A2D | **T2a** |
| `dllm-a2d/` | 用 **dllm** 包做 A2D | **T2b** |
| `dllm/` | 直接用 dllm 包训 DLM | **T3** |

T2c（对开源模型做 A2D）**不开新目录**，复用 `a2d/` 或 `dllm-a2d/`，
只是权重来源不同。

### 3.2 核心设计：每个功能都有两条路

| | 不用包 | 用包 |
|---|---|---|
| AR 训练 | `minimind/` | `transformer/` |
| A2D | `a2d/` | `dllm-a2d/` |

### 3.3 写代码的分工

| | 谁写 | 位置 |
|---|---|---|
| **A 版** | AI | `example/` — 用 `transformers` / `dllm` 包当正确性基准 |
| **B 版** | **你** | `src/` 的七个目录 |

`src/` 里 AI 只写 docstring，不写实现。包括 `transformer/` 和 `dllm-a2d/`
—— 它们名字上像「调包」，但写在 `src/` 里就按 `src/` 的规矩办。

## 4. 数据与 tokenizer 契约

三条线**共用同一份**，这是「公平对比」的前提。

```
data -> /mnt/extra/data/prepared/minimind3-6400/
  tokens.bin         665 MB, uint16, 332,495,324 token
  doc_lengths.npy    int32, (1270238,) mean 261.8 max 11802
  meta.json          vocab_size=6400, n_docs=1270238, val_frac=0.01
```

```
src/base/tokenizer.json   BPE, vocab 6400, 6108 merges

PAD 0 <|endoftext|>   BOS 1 <|im_start|>   EOS 2 <|im_end|>   MASK 27 <|buffer1|>
```

T2c 用 Qwen3-0.6B 则是另一套：vocab 151936，mask token **151669**
（267 个空闲 slot 之一，`convert_ids_to_tokens` 返回 `None` 可验证）。

🔴 **mask token 为什么是 27**：`<|buffer1|>` 是 minimind tokenizer 里的空闲
占位符。这个选择在 `notes/02-vocab-tokenizer-mask.md` 里有三种方案的对比。
如果将来想用「扩 vocab_size 加新 token」而不是复用空闲 slot，那是一次架构级
改动，会让 T1 的权重和 T2 的词表不一致 —— 到时再说，现在不做。

**已知的语料事实**：613 篇文档（0.05%）长度 < 16，内容是
`'CCertainly'` `'200Hello'` 这样的退化碎片。数据本身是好的（随机和中位长度
文档 decode 出来是通顺中文），但**按长度过滤数据会优先留下这批碎片**。

## 5. 环境契约

**单 venv，可复现，换机器跑一遍结果一致。**

| | 版本 | 为什么 |
|---|---|---|
| Python | 3.14.7 | ROCm 官方文档的版本，cp314 wheel 已在索引里 |
| torch | `2.12.0+rocm10.0.0` | ROCm 10.0 官方 wheel，gfx1150 |
| transformers | `4.57.0` | **dllm 的 pin。不许升** |
| datasets | `4.2.0` | dllm 的 pin。`trl` 会顶到 5.x，装完强制回钉 |
| peft | `0.17.1` | T1a 的 LoRA |
| accelerate | — | **peft 的硬依赖**，不能砍 |
| GPU | AMD Radeon 880M (gfx1150) | |

**不装**：deepspeed（ZeRO 多卡切片，单卡用不上）、wandb、
torchvision / torchaudio（纯文本）。

**dllm 的装法**（见 `USAGE.md` 的详细说明）：

```
上游 https://github.com/ZHZisZZ/dllm.git
  → clone 到 ~/.cache/ARnDlm/dllm
  → checkout 钉死的 commit ca17675
  → 打 patches/dllm-0001-a2d-config-class.patch
  → uv pip install --no-deps -e
```

**为什么这么做**：dllm pin 了 transformers 4.57 但代码和 4.57 不兼容
（三个 A2D 模型类缺 `config_class`，撞上 4.57 新增的 `AutoModel.register`
一致性校验）。patch 进 git 而不是改本机的 `~/Public/dllm`，是为了
**换机器可复现**。用 commit 而不是 `main`，是为了上游一提交环境不会静默变。

`.venv` 6.2G / 106 包，其中 +400MB 是 dllm 顶层急切 import 拖进来的
`lm_eval`/`scipy`/`sklearn` —— **本项目一个都不会用**，但不装就 import 不了。

## 6. 验收标准

**每一步的验收都要能证伪。** 「loss 降了」不是验收标准。

### T1 · AR 基准模型

| | 验收 |
|---|---|
| **T1a** `transformer/` | 文本通顺；两个吞吐口径都报出来；能存能续 |
| **T1b** `minimind/` | **与 T1a 逐 token 一致** —— 同 batch 同权重，logits 对齐到 1e-5 |

T1b 的验收是「**和 T1a 数值一致**」，不是「loss 差不多」。
差 1e-5 和差 0.3 都是失败，只是严重程度不同。

### T2 · A2D

| | 验收 |
|---|---|
| **T2a** `a2d/` | 与 T2b 在同权重同 seed 下 **loss 差 < 1%** |
| **T2b** `dllm-a2d/` | 同上（互为对照） |
| **T2c** 开源模型 | 能跑通；报 repetition rate |

A2D 的两步：`is_causal: True → False` + `vocab += [MASK]`。

🔴 **A2D 的低代价微调策略**：`notes/03-minimind-dlm.md` §10 记录了作者的结论
是「只训 Q/K 就够」。我把它列为**待自己重做消融**的事项，而不是直接照搬 ——
这是原作者在一组特定超参下的结论，换个规模结论可能变。

### T3 · dllm 直训 DLM

验收 = **报 repetition rate，不只报 loss**。

## 7. 核心指标

🔴 **主指标是重复率（repetition rate），不是 loss。**

```
rep_n(text, n) = 1 - (不同的 n-gram 数 / 总 n-gram 数)
```

**理由**：DLM 失败最典型的表现就是复读，而 loss 照样下降。只看 loss 会把
失败的模型当成成功的。这是 dlm 项目 AGENTS.md 里定下的教训，本项目继承。

参考值：`n=4` 时，正常文本 ~0%，AR 训崩 ~60%+。

`test/05-repetition-rate.py` 的 Part B 直接对比 AR 与 DLM，
并打印两者的差值 —— **差距是否随训练收窄**才是有意义的信号。

## 8. 已知约束

来自 `logs/dev/02-hardware-perf.md`（继承自 dlm 项目，同机器同 torch 版本）：

- **瓶颈是 attention 的 backward kernel**，不是带宽不是算力。换 SDPA backend 无解。
- **`max_seq_len ≤ 512`** —— attention 是 O(T²)。
- **`batch 16 就够** —— B=1 到 B=64 吞吐只差 30-53%。
- **packing 比 padding 快 1.71x** —— 零 padding 浪费。
- **不开 gradient checkpointing** —— 小模型慢 18-23%，只省 3GB。
- 实测量级 **~3,700-4,700 tok/s**。332M tokens 跑满是 19-25 小时，**必须缩数据**。
- run-to-run 方差可达 **±20%**，短 benchmark 尤其不稳。

**汇报数字的硬规矩**：必须同时报 `train-only` 和 `wall-clock` 两个吞吐；
`Predicted` 和 `Measured` 分两列；benchmark ≥ 400 步。
完整 checklist 见 `notes/05-measurement.md`。

## 9. 未定的事项

🔴 **参数量不钉死。**

已验证的候选（Qwen3 架构 + vocab 6400 + tied embedding）：

| | hidden | layers | Q/KV | head_dim | inter | params |
|---|---|---|---|---|---|---|
| 迷你 | 512 | 8 | 8/4 | 64 | 1536 | 28.5M |
| **中** | **576** | **12** | **8/4** | **72** | **1728** | **51,478,272**（实测） |
| 偏大 | 640 | 10 | 8/4 | 80 | 1920 | 53.3M |

**先跑冒烟测试测吞吐，再回头定参数量 / batch / seq_len。**
这三个数和参数量是耦合的，先拍数字再测是本末倒置。

## 10. 目录结构

```
AGENTS.md       规则: 分工、知识沉淀、环境、汇报纪律
SPEC.md         本文件 —— 契约
ROADMAP.md      步骤、产出、验收、状态
USAGE.md        命令 + 安装细节
requirements.txt
patches/        dllm 的上游 patch (进 git, 保证可复现)
notes/          纯知识 (概念/原理/坑), 换机器仍成立
logs/dev/       手写开发记录 (进 git)
logs/run/       程序产物 (不进 git)
example/        参考实现 (AI 写)
test/           验证脚本
scripts/        setup_env / setup_symlinks / install_dllm / plot / start chat webui etc
configs/        base.yaml + 各实验配置
src/            B 版, 七条路线
```

**知识沉淀的判据**：「换一台机器或换一个项目这句话还成立吗？」成立 → `notes/`；
不成立 → `logs/dev/`。再问「人写的还是程序吐的？」程序吐的 → `logs/run/`。

## 11. 当前状态

```
scaffold        完成 (1ba36ca)
环境            完成 (348ee33) —— 5 个测试全 PASS
src/ 实现       0%
训练            未开始
```
