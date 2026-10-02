# ARnDlm SPEC

## 1. Project objective and scope

在一致的 tokenizer、vocab、数据集和运行环境下，比较 Qwen-style Transformer 的 AR 与 DLM：

- HF Transformers 与原生 PyTorch 两种 AR 实现；
- MiniMind-DLM A2D、dllm A2D、Direct DLM 三种 DLM 路线；
- 自训练小模型与 Qwen3-0.6B 的 A2D。

项目用于学习、复现和可测量对比，不追求 SOTA，不做多卡、多模态、权重发布或 dllm 库重构。

本项目分两个阶段：

1. **预训练**（已完）：§2.3 的 10 格，比较 AR 与 DLM 的训练与生成。
2. **后训练**（本节新增，T4）：把两个 AR 小模型（`hf-ar-example` / `minimind-ar-example`）做 SFT 与 DPO，
   得到能对话的模型。见 §1.6。

## 2. Experimental matrix

### 2.1 Base models

| ID | Model | Implementation | Size | Training |
|---|---|---|---|---|
| B1 | Qwen-mini-HF | Transformers | 冒烟后决定 | AR |
| B2 | Qwen-mini-MiniMind | 原生 PyTorch | 冒烟后决定 | AR |
| B3 | Qwen3-0.6B | Transformers | 600M | 现有 checkpoint |

### 2.2 DLM methods

| ID | Method | Description |
|---|---|---|
| A1 | dLLM-A2D | 使用 dllm 将 AR checkpoint 转为 DLM |
| A2 | MiniMind-DLM-A2D | 按 MiniMind-DLM 思路手写 A2D |
| A3 | Direct-DLM | 随机初始化后直接进行 diffusion training |

### 2.3 Experiment matrix

| Base model | dLLM-A2D | MiniMind-DLM-A2D | Direct-DLM |
|---|---:|---:|---:|
| Qwen-mini-HF | ✓ | ✓ | ✓ |
| Qwen-mini-MiniMind | ✓ | ✓ | ✓ |
| Qwen3-0.6B | ✓ | ✓ | — |

### 2.4 Post-training matrix (T4)

后训练分两个轴：**base checkpoint**（AR 还是 DLM）和 **objective**（AR 的 next-token CE，还是 DLM 的 masked CE）。
目标不是刷指标，是**训到能对话**（见 §6 的“能 chat”门槛）。

| # | 起始 checkpoint | objective | 目录 | 优先级 |
|---|---|---|---|---|
| T4-1 | `hf-ar-example` | AR CE（assistant 段） | `training/sft_ar_hf/` | P0 |
| T4-2 | `minimind-ar-example` | AR CE（assistant 段） | `training/sft_ar_native/` | P0 |
| T4-3 | `hf-minimind-a2d-example` | masked CE（**只在 assistant 段**加噪） | `training/sft_a2d_hf/` | P0 |
| T4-4 | `native-minimind-a2d-example` | masked CE（只在 assistant 段） | `training/sft_a2d_native/` | P0 |
| T4-5 | `hf-dllm-a2d-example` | dllm `MDLMTrainer`（只在 assistant 段） | 同上 `--loss dllm` | P1 |
| T4-6 | `hf-direct-dlm-example` | masked CE（只在 assistant 段） | 同上 | P2 |
| T4-7 | `minimind-ar-example` | DPO（chosen/rejected） | `training/dpo_ar_native/` | P2 |
| T4-8 | `minimind-dllm-a2d-example` | DPO on DLM | — | **不做**，见下 |

**为什么 DLM-SFT 不是 AR-SFT 换个数据**：DLM 的加噪必须**限制在 assistant 段**
（`notes/03` §4）—— prompt 区被 mask 掉，模型就看不到自己在回答什么，会退化成
“无条件续写”。这正好是 `notes/03` §4 反复强调的那一条，现在 A 版只跑过
裸文本（整块都是 assistant 区），**这条规则一次都没被验证过**。T4 就是验它的地方。

**为什么 DPO on DLM 不做**：DPO 需要序列级 log-prob，而 masked diffusion 的
“序列 log-prob”没有单一定义（每步去噪的归一化项不一样）。dllm 的
`pipelines/rl/grpo/` 是 AR 导向的，与本项目的 DLM 对比无关。这是**待研究的题**，不是工程量。

**Direct-DLM 的 SFT 降到 P2**：它的 eval 5.64 已是三档里最差，且没有任何 AR 知识可继承，
能 chat 的概率低。先把 P0 四格跑通。

## 3. Model, tokenizer and data specification

### 3.1 Unified model architecture

自训练模型使用 Qwen-style decoder-only Transformer：

- RoPE；
- RMSNorm；
- SwiGLU FFN；
- GQA attention；
- QK-Norm；
- tied embedding；
- causal LM head（AR）或双向 attention + denoising objective（DLM）。

`head_dim` 从模型配置计算；GQA 将较少的 K/V heads 复制到 Q heads；tied embedding 的共享参数传给 optimizer 前按 `id()` 去重。

### 3.2 Tokenizer tracks

自训练路线使用项目内的 6400-vocab BPE：

- 文件：`tokenizer/tokenizer.json`；
- vocab size：6400；merges：6108；
- PAD 0 `<|endoftext|>`、BOS 1 `<|im_start|>`、EOS 2 `<|im_end|>`；
- MASK 27 `<|buffer1|>`，复用预留 token，不扩充词表。

Qwen3-0.6B A2D 使用 checkpoint 自带的 Qwen tokenizer：

- vocab size：151,936；
- 保留 Qwen3 的 chat template 和 special-token 定义；
- **MASK 复用 `151662` `<|fim_pad|>`**，不新增 token、不动 embedding 形状；
- 不与 6400-vocab 权重、token id 或逐 token 对拍混用。

> 最初写的是「MASK 使用 id 151669」，实测**越界**：Qwen3-0.6B 的 tokenizer
> `vocab_size` 是 151,669，合法 id 只有 0-151668。改成复用 151662 的理由：
>
> 1. **不动 embedding** —— 新增 token 需要 `resize_token_embeddings(151670)`，
>    把 embedding 从 checkpoint 的 151,936 行缩到 151,670 行，少 266 行。
>    复用则参数量保持精确的 598,071,296，A/B 对拍的基线才干净。
> 2. **语料不会撞** —— 实测 `tinystories-6400` 和 `minimind3-6400` 里
>    **没有任何 token id ≥ 151000**（见 `example/ISSUES.md` OPEN-1）。
> 3. **对 LoRA 两格影响为零** —— LoRA 只挂 attention 四个投影，backbone 冻结，
>    embedding 本来就不训练，所以「新增」与「复用」的可训练参数完全相同。
>
> 注意点：`<|fim_pad|>` 在 Qwen3 自己的预训练语料里**可能出现过**（Qwen 用了 FIM
> 格式），所以它不是随机初始化的一行。这不影响正确性（A2D 只把它当被遮位置
> 的标记），但不要把它当成「空白 token」来期待。

6400-vocab 方案降低小模型的 embedding、输出层和 optimizer 成本，适合自训及 AR/A2D 对照；Qwen tokenizer 对英文、代码、混合文本和 Qwen3 checkpoint 的兼容性更好，但词表和训练成本显著更高。两条 tokenizer track 的 loss、perplexity、token/s 只能在各自 track 内比较；跨 track 应按相同原文和字符数/字节数归一化。

### 3.3 Data

数据必须使用与模型相同的 tokenizer：

- 6400-vocab track：`/mnt/extra/data/prepared/minimind3-6400/`，包含 `tokens.bin`、`doc_lengths.npy`、`meta.json`；
- Qwen3 track：使用 Qwen tokenizer 重新编码的数据；TinyStories 或小型 instruction/text corpus 可作为来源；
- 三个自训练 DLM 路线共享 6400-vocab 数据，保证对比公平。

### 3.4 SFT / DPO 数据（6400-vocab track）

后训练只在 6400-vocab track 上做，两个 base 都是 3.2M 的 Tier 0 模型。

| 项 | 值 |
|---|---|
| 源文件 | `data/raw/minimind_dataset/sft_t2t.jsonl`（14 GiB，509 万条对话） |
| 原始格式 | `{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}` |
| 展平产物 | `data/prepared/sft-6400/` — 与 pretrain 同格式：`tokens.bin` / `doc_lengths.npy` / `meta.json`，每条对话一段 |
| chat 标记 | 纯文本标记，与 minimind 一致，不新增 special token |

**展平是必须的，不是可选的。** `scripts/prepare_data.py` 的 `read_text()` 只取 `text` 键，
对话数据没有 `text`，会被**静默跳过**（0 token，不报错）。展平脚本要写两遍内容：

```
<user>\n{提问}\n<assistant>\n{回答}\n
```

**产物除了 token，还要带 assistant 区间**：

| 文件 | 内容 | 用途 |
|---|---|---|
| `tokens.bin` | 展平后的 token 序列（含 BOS/EOS） | 输入 |
| `doc_lengths.npy` | 每条对话的 token 数 | packing / 边界 |
| `assistant_spans.npy` | `[n_docs, 2]`，每条对话 assistant 段的 token 起止 | **加噪 / loss 区间** |
| `meta.json` | vocab_size / n_docs / total_tokens / source_sha256 | 自检 |

`assistant_spans` 是 T4 的核心。**没有它，AR 只能算全序列 CE（等于没做 SFT），
DLM 会把 prompt 也遮掉（等于没在看问题）**。两条路都静默跑完。

区间约定（与 `notes/03` §4 的示例一致）：

```
tokens:   [BOS] <user> \n 提问 \n <assistant> \n 回答 \n [EOS]
区间:                              ^^^^^^^^^^^^^^^^  ← 含末尾 \n，不含 <assistant> 标记本身
labels:   区间外全 -100
```

DLM 的腐蚀必须**只在区间内**抽 mask：区间外的位置（包括整个 prompt）**不得**被替换成 MASK。

DPO 的源是 `data/raw/minimind_dataset/dpo.jsonl`（53 MB），格式为 chosen/rejected 两条回答，
需要同一套展平 + 区间逻辑。

### 3.5 后训练语料规模（smoke）

smoke 只取 sft 数据前 2000 条对话，展平后约 20-40 万 token（`logs/dev/06-datasets.md` 记的压缩比约 0.5-1.0），
用于验证 SFT loss、mask 区间、对话采样三条路径是否成立。正式规模另定。

## 4. Model configurations

### Tier 0 · Debug model

用于验证 forward、短训练、checkpoint 保存/恢复、A2D 转换、采样和 loss 下降。唯一冒烟配置：

```yaml
hidden_size: 256
num_layers: 2
num_attention_heads: 2
num_key_value_heads: 1
intermediate_size: 768
context_length: 128
```

该配置只用于 pipeline validation，不代表正式模型规模。

实测 3,213,056 params。单步 18.8 ms (seq 128) / 64.4 ms (seq 512), 2M token 约
37 秒, 适合当 smoke test。

⚠️ **前 50 步是 kernel 编译相, 比稳态慢 1.15-1.8 倍**。单步只有十几毫秒时
这个开销相对很大, 起步要多留预热步, 否则前几十步的 loss 不可信。

### Tier 1 · Main experiment candidates

正式规模须在 Tier 0 冒烟测试和吞吐测量后选择。候选配置如下，不预先承诺其中任何一个：

```yaml
# 26,879,488 params (measured, vocab 6400)
hidden_size: 512
num_layers: 8
num_attention_heads: 8
num_key_value_heads: 4
intermediate_size: 1408
context_length: 512
```

实测 (2026-10-01, AMD Radeon 880M, ROCm 10.0, bf16 + AdamW + sdpa, packing 零 padding,
seq 512 / batch 8): **7,995 train-only tok/s**, 跑完 `minimind3-6400` 的 332M token 约
**11.6 小时**。wall-clock 吞吐与 train-only 几乎相等 (step 外开销 0.09%), 所以两个
口径在本项目里给出同一个数。峰值显存 1.41 GB。

batch 8 与 batch 16 吞吐无法区分 (ms/step 精确翻倍)，故取 8。数据见
`logs/dev/02-hardware-perf.md`。

### Tier 2 · Extended candidate

```yaml
# 80,433,408 params (measured, vocab 6400)
hidden_size: 768
num_layers: 12
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
context_length: 512
```

同条件实测: **3,540 train-only tok/s**, 332M token 约 **26.1 小时**,
峰值显存 2.76 GB。吞吐随规模亚线性 (3.0x 参数 → 2.33x
耗时)，即大模型的每 token 算力更高；代价是 wall-clock 翻倍。是否上这一档取决于
单次实验的预算上限，不取决于它能不能跑完。

### Tier 3 · Real model

Qwen3-0.6B 使用现有 checkpoint，A2D 采用 LoRA/adapter 并冻结 backbone；目标是 dLLM-A2D 与 MiniMind-DLM-A2D。

## 5. Environment contract

目标环境为 Python 3.14.7、ROCm torch 2.12.0、Transformers 4.57.0、datasets 4.2.0、peft 0.17.1 和 accelerate。不得安装 deepspeed、wandb、torchvision、torchaudio。dllm 固定 commit，并通过仓库内 patch 安装。环境契约由 PLAN.md 中的环境阶段执行。

## 6. Acceptance criteria

- T1a：AR checkpoint 可保存、恢复和生成；报告 train-only 与 wall-clock tok/s。
- T1b：同 batch、同权重下与 T1a logits 逐 token 对齐到 `1e-5`。
- T2a/T2b：同 checkpoint、同 seed 下 loss 差小于 1%。
- T2c：Qwen3-0.6B A2D 跑通并报告 repetition rate。
- T3：必须报告 repetition rate，不能只报告 loss。
- benchmark 至少 400 步并丢弃预热；`Predicted` 与 `Measured` 分列。

### T4（后训练）验收

- **T4a 展平正确**：抽查 3 条对话，`decode(展平后的 tokens)` 能还原出带 `<user>` / `<assistant>` 标记的原文；
  `meta.json` 里的 assistant 区间长度与回答长度一致（误差 ≤ 1 token）。
- **T4b 区间正确**：随机取一条样本，断言 label 在 user 段全是 -100、在 assistant 段全不是 -100。
  **区间错位是最阴的一类 bug**：loss 照样下降，模型学会了「不看问题直接答」。
- **T4c SFT/DPO 端到端**：从 `hf-ar-example` / `minimind-ar-example` 出发，各能保存、恢复、生成；
  生成走对话格式（`<user>…<assistant>…`），报告 train-only 与 wall-clock tok/s、eval loss（assistant 段）、repetition rate。
- **T4d A/B 对齐**：行1（`training/sft_hf/`）与行2（`training/sft_native/`）在同 seed、同数据、
  同区间下逐步 loss 一致（fp32，`< 1e-6`）。
- **T4e SFT 必须比 AR 续训强**：同一批数据同样步数，SFT 的 assistant 段 eval loss
  低于「用同一批数据做纯 next-token 续训」的 eval loss。这条不满足说明区间 mask 没生效。
- 所有 DLM 相关结论仍必须报 repetition rate（§6 T3 规则对 T4 同样成立）。

### T4 的 DLM 专项

- **T4f 区间外的位置永不被加噪**：抽 100 个 batch，断言 `MASK` 出现的位置 ⊆ assistant 区间。
  这条比 loss 曲线重要得多：prompt 被遮了，loss 照样降，但模型学的是“无条件生成回答”。
- **T4g 两种 objective 在 SFT 下仍然一致**：手写 masked CE vs dllm `MDLMTrainer`，
  同区间、同 seed 下 loss 相对差 `< 1%`（即 C2 的逻辑搬到区间版）。
- **T4h AR-SFT 与 DLM-SFT 的 assistant 段 eval loss 可比**（同一个 tokenizer、同一批区间），
  差值就是「双向去噪 vs 自回归」在同预算下的实际代价。

### “能 chat”的门槛（T4 的真正验收）

不追求指标，只要求模型能像个对话模型一样回话。四条同时满足：

| # | 判据 | 阈值 | 为什么这么定 |
|---|---|---|---|
| 1 | assistant 段 eval loss | ≤ 1.5 × AR-SFT 同条件 | DLM 先天比 AR 难，放 1.5 倍；超了就是没训好而不是方法不行 |
| 2 | rep-4（≥64 token 生成） | < 0.5 | 复读是最典型的 DLM 失败模式（`notes/03` §11） |
| 3 | 生成非空、格式正确 | `Once upon a time` / `The future of AI is` / 一个真实提问，各能接在 `<assistant>` 后输出 | 排掉「输出全 PAD / 全重复字符」 |
| 4 | 不回声提问 | 抽查 2 条，回答里不出现 prompt 原文 | AR 模型的经典失败：把问题重复一遍当成回答 |

**步数**：能 chat 即可，不要求收敛。Tier 0 上先跑 200 步看 rep-4 和格式，
不够再追到 400 / 800 步。**每加一档都要重报 rep-4**，并遵守 `notes/05` 的计时规则。

## 7. Repository layout

```text
configs/                 # 模型和实验配置
scripts/                 # 环境、数据下载/准备、训练和可视化命令
data/                    # dataset、packing、tokenizer 适配
tokenizer/               # 6400 tokenizer
models/hf_qwen/          # Transformers Qwen 适配
models/minimind_qwen/    # 原生 PyTorch Qwen-style model
training/ar_hf/          # T1a
training/ar_native/      # T1b
training/sft_ar_hf/       # T4-1 SFT，行1（AR objective，用包）
training/sft_ar_native/   # T4-2 SFT，行2（AR objective，不用包）
training/sft_a2d_hf/      # T4-3 SFT，行1（DLM objective，区间内加噪）
training/sft_a2d_native/  # T4-4 SFT，行2（DLM objective）
training/dpo_ar_native/   # T4-7 DPO（只在 AR 上做）
training/a2d_minimind/   # T2a
training/a2d_dllm/       # T2b/T2c
training/direct_dlm/     # T3
diffusion/               # corruption、scheduler、loss、sampler
evaluation/              # generation、benchmark、repetition rate
utils/                   # logger、device、seed、checkpoint、metrics
example/                 # 参考实现
notes/ logs/ scripts/patches/ # 知识、开发记录、运行产物、dllm patch
```

路线依赖单向：公共数据和工具被模型、训练和评测使用；A2D 可以读取 AR 权重，但不导入 AR 训练循环。

## 8. 任务表（A 版 / B 版 两个维度）

### 8.1 两个维度的定义

同一件事写两遍。**任务相同，规模不同。**

| | A 版（参考实现） | B 版（正式实现） |
|---|---|---|
| 谁写 | AI | 你 |
| 位置 | `example/` | 根目录 `models/` `training/` `diffusion/` `evaluation/` `utils/` `data/` |
| 模型规模 | **Tier 0**：3.21M 参数 · ctx 128 · 2 层 | **Tier 1（待测）**：~50M · ctx 512 · 8 层 |
| 语料 | tinystories-6400 切片（~20 万 token） | minimind3-6400（332M token） |
| 步数 | 200 | 按 `logs/dev/02` 吞吐阶梯定 |
| 目的 | **pipeline 能不能跑通 + 代码对不对** | 正式实验与可测量对比 |
| checkpoint | `checkpoints/example/…-example/` | `checkpoints/…` |
| 能力 | **不看**（loss 降没降、rep-4 多少都只记录） | 看（这是要拿数字的） |
| 对拍关系 | A 是基准 | **B 的输出必须与 A 对齐** |

**A 版跑通 ≠ B 版可以写。** B 版在写之前，用 `example/teach/` 里那一步的
「验收命令」自己先跑一遍 A 版，拿到基准数。

### 8.2 主任务表

| 任务 | 内容 | A 版（AI 写） | B 版（你写） | 验收 |
|---|---|---|---|---|
| **T0** | 公用底座：数据、存盘、指标、日志 | `example/common/` ✅ | `utils/` `data/` | 两套实现行为一致；B 版跑 `test/01`-`test/05` |
| **T1a** | AR 训练 · 行1（用包） | `example/base/hf/train_ar.py` ✅ | `models/hf_qwen/` + `training/ar_hf/` | 存/恢复/生成；报双吞吐 |
| **T1b** | AR 训练 · 行2（不用包） | `example/base/native/` ✅ | `models/minimind_qwen/` + `training/ar_native/` | **C1：logits `1e-5`**（A 版实测 6.91e-06） |
| **T2a** | MiniMind-DLM A2D | `example/base/{hf,native}/a2d_minimind.py` ✅ | `training/a2d_minimind/` | **C2/C3：loss 差 `< 1%`**（实测 7.6e-08 / 1.5e-07） |
| **T2b** | dLLM-A2D（上游路径） | `example/base/{hf,native}/a2d_dllm.py` ✅ | `training/a2d_dllm/` | **C5：转换 0 丢失 0 漂移** |
| **T2c** | Qwen3-0.6B A2D（LoRA） | `example/base/qwen3_0p6b/` ✅ | `training/a2d_{dllm,minimind}/` 的行3 分支 | 两格跑通 + **报 repetition rate** |
| **T3** | Direct-DLM（从零训） | `example/base/{hf,native}/direct_dlm.py` ✅ | `training/direct_dlm/` | 跑通 + 报 repetition rate |
| **T4-1/2** | SFT · AR（行1 / 行2） | `example/base/{hf,native}/sft_ar.py` ⬜ | `training/sft_ar_{hf,native}/` | T4a-T4e + “能 chat”四条门槛 |
| **T4-3/4** | SFT · DLM（行1 / 行2） | `example/base/{hf,native}/sft_a2d.py` ⬜ | `training/sft_a2d_{hf,native}/` | **T4f：区间外永不加噪** + “能 chat” |
| **T4-5** | SFT · DLM（dllm objective） | 同上 `--loss dllm` ⬜ P1 | 同目录 | T4g：两种 objective 仍一致 |
| **T4-6** | SFT · Direct-DLM | 同上 P2 | 同目录 | 能 chat |
| **T4-7** | DPO（只在 AR 上） | ⬜ P2 | `training/dpo_ar_native/` | chosen/rejected 排序正确 |
| **—** | Tier 1 定档 | `example/teach/13-tier1.md` ✅ | 同左（写在 `logs/dev/02`） | 先测吞吐再定参数量 |

✅ = A 版已完成（在 main） · ⬜ = 未写 · P1/P2 = 排在 P0 之后

### 8.3 依赖顺序

```
T0 ──→ T1a ──→ T1b ──┬──→ T2a ──┬──→ T2c
                     │           ├──→ T4-1/2 (SFT·AR)
                     ├──→ T2b ───┤
                     └──→ T3 ────┴──→ T4-3/4 (SFT·DLM) ──→ T4-5
                                     └──→ T4-6 ──→ T4-7
```

T4 的数据（展平 + `assistant_spans.npy`）是 T4 自己的前置，不依赖 T2。

### 8.4 B 版每一步的对拍命令

| 任务 | 命令 | 判定 |
|---|---|---|
| T0 | `.venv/bin/python example/tests/test_common.py` | 逐条对上 `example/common/` |
| T1a | `.venv/bin/python example/base/hf/train_ar.py` | 拿到 A 版基准数 |
| T1b | `.venv/bin/python example/check_ab.py c1 --layer-report` | `max\|Δ\| ≤ 1e-5`，失败时逐层报告定位 |
| T2a | `.venv/bin/python example/check_ab.py c2` | `rel_diff < 1%` |
| T2b | `.venv/bin/python example/check_ab.py c5` | `missing=unexpected=drifted=0` |
| T3 | `.venv/bin/python example/base/*/direct_dlm.py` | 与 A2D 的 eval 差记录下来 |
| T4 | `.venv/bin/python example/base/*/sft_*.py` | T4a-T4h 逐条 |

**通用**：`bash example/smoke_all.sh --quick` 一次跑完 A 版全部格子，
任何一格 FAILED 会打出来。
