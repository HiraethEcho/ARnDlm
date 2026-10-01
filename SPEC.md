# ARnDlm SPEC

## 1. Project objective and scope

在一致的 tokenizer、vocab、数据集和运行环境下，比较 Qwen-style Transformer 的 AR 与 DLM：

- HF Transformers 与原生 PyTorch 两种 AR 实现；
- MiniMind-DLM A2D、dllm A2D、Direct DLM 三种 DLM 路线；
- 自训练小模型与 Qwen3-0.6B 的 A2D。

项目用于学习、复现和可测量对比，不追求 SOTA，不做多卡、多模态、权重发布或 dllm 库重构。

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
- MASK 使用 id 151669；
- 不与 6400-vocab 权重、token id 或逐 token 对拍混用。

6400-vocab 方案降低小模型的 embedding、输出层和 optimizer 成本，适合自训及 AR/A2D 对照；Qwen tokenizer 对英文、代码、混合文本和 Qwen3 checkpoint 的兼容性更好，但词表和训练成本显著更高。两条 tokenizer track 的 loss、perplexity、token/s 只能在各自 track 内比较；跨 track 应按相同原文和字符数/字节数归一化。

### 3.3 Data

数据必须使用与模型相同的 tokenizer：

- 6400-vocab track：`/mnt/extra/data/prepared/minimind3-6400/`，包含 `tokens.bin`、`doc_lengths.npy`、`meta.json`；
- Qwen3 track：使用 Qwen tokenizer 重新编码的数据；TinyStories 或小型 instruction/text corpus 可作为来源；
- 三个自训练 DLM 路线共享 6400-vocab 数据，保证对比公平。

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
