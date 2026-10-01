# Qwen-style Diffusion Language Model (DLM) Study Project SPEC

## 1. Project Objective

建立一个统一实验平台，比较：

1. **Qwen-style Transformer 的两种实现方式**
   - HuggingFace Transformers
   - MiniMind / PyTorch native
2. **三种 DLM 训练路线**
   - dLLM A2D
   - MiniMind-DLM A2D
   - Direct DLM
3. **两个模型规模**
   - 自训练小模型（10M / 50M / 150M）
   - 预训练模型 Qwen3-0.6B

---

# 2. Experimental Matrix

## 2.1 Base Models

| ID  | Model              | Implementation | Size             | Training            |
| --- | ------------------ | -------------- | ---------------- | ------------------- |
| B1  | Qwen-mini-HF       | Transformers   | 10M / 50M / 150M | AR                  |
| B2  | Qwen-mini-MiniMind | PyTorch native | 10M / 50M / 150M | AR                  |
| B3  | Qwen3-0.6B         | Transformers   | 600M             | Existing checkpoint |

---

## 2.2 DLM Conversion / Training Methods

| ID  | Method           | Description                                 |
| --- | ---------------- | ------------------------------------------- |
| A1  | dLLM-A2D         | Using dLLM package to convert AR checkpoint |
| A2  | MiniMind-DLM-A2D | Using MiniMind-DLM implementation           |
| A3  | Direct-DLM       | Random initialization + diffusion training  |

---

## 2.3 Final Experiment Matrix

### Small Models

| Base Model         | dLLM-A2D | MiniMind-DLM-A2D | Direct-DLM |
| ------------------ | -------- | ---------------- | ---------- |
| Qwen-mini-HF       | ✓        | ✓                | ✓          |
| Qwen-mini-MiniMind | ✓        | ✓                | ✓          |

---

### Large Model

| Base Model | dLLM-A2D | MiniMind-DLM-A2D | Direct-DLM |
| ---------- | -------- | ---------------- | ---------- |
| Qwen3-0.6B | ✓        | ✓                | -          |

---

# 3. Model Specification

## 3.1 Unified Architecture

Qwen-style Decoder Transformer:

- Decoder-only Transformer
- RoPE
- RMSNorm
- SwiGLU FFN
- GQA Attention
- Causal LM Head

Tokenizer:

- Qwen tokenizer

Dataset:

- TinyStories (primary)
- Optional: small instruction/text corpus

---

# 4. Model Configurations

## Tier 0: Debug Model

Target:

快速验证 pipeline。

Configuration:

```yaml
hidden_size: 256
num_layers: 4
num_attention_heads: 4
num_key_value_heads: 2
intermediate_size: 768
context_length: 256
```

Parameter:

~10M

用途:

- AR training
- A2D conversion
- Direct DLM smoke test

---

## Tier 1: Main Experiment Model

Target:

主要实验。

Configuration:

```yaml
hidden_size: 512
num_layers: 8
num_attention_heads: 8
num_key_value_heads: 4
intermediate_size: 1408
context_length: 512
```

Parameter:

~50M

用途:

- 所有 A2D 方法比较
- Direct DLM
- dLLM pipeline

---

## Tier 2: Extended Model

Target:

scale test。

Configuration:

```yaml
hidden_size: 768
num_layers: 12
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
context_length: 512
```

Parameter:

~150M

用途:

- optional scaling experiment

---

## Tier 3: Real Model

Model:

Qwen3-0.6B

Training:

- LoRA / adapter A2D
- freeze backbone

Target:

- dLLM-A2D
- MiniMind-DLM-A2D

---

# 5. Project Structure

```text
qwen-dlm-study/

├── configs/
│   ├── qwen10m.yaml
│   ├── qwen50m.yaml
│   └── qwen150m.yaml
│
├── data/
│   ├── prepare.py
│   └── dataset.py
│
├── tokenizer/
│
├── models/
│   ├── hf_qwen/
│   ├── minimind_qwen/
│   └── qwen3/
│
├── training/
│   ├── train_ar_hf.py
│   ├── train_ar_minimind.py
│   ├── train_dllm_a2d.py
│   ├── train_minimind_dlm.py
│   └── train_direct_dlm.py
│
├── diffusion/
│   ├── scheduler.py
│   ├── corruption.py
│   ├── sampler.py
│   └── loss.py
│
├── experiments/
│   ├── exp01_ar/
│   ├── exp02_dllm_a2d/
│   ├── exp03_minimind_dlm/
│   ├── exp04_direct_dlm/
│   └── exp05_qwen3/
│
└── evaluation/
    ├── generate.py
    └── benchmark.py
```

---

# 6. Implementation Phases

---

# Phase 0: Environment Setup

## Tasks

- ROCm PyTorch
- Transformers
- Accelerate
- dLLM environment
- MiniMind environment

## Checkpoint

```text
GPU training works
model forward works
checkpoint save/load works
```

---

# Phase 1: Qwen-mini AR Baseline

## Implement

### HF Version

```
Qwen-mini
+
Transformers
+
Causal LM training
```

### MiniMind Version

```
Qwen-mini
+
PyTorch native training
```

## Output

```text
hf-qwen50m-ar

minimind-qwen50m-ar
```

---

# Phase 2: MiniMind-DLM A2D

Input:

```
qwen50m-ar checkpoint
```

Implement:

- diffusion corruption
- timestep
- denoising objective
- sampler

Output:

```
qwen50m-minimind-a2d
```

---

# Phase 3: dLLM A2D

Input:

```
qwen50m-ar checkpoint
```

Run:

```
dLLM A2D pipeline
```

Output:

```
qwen50m-dllm-a2d
```

---

# Phase 4: Direct DLM

Input:

```
random initialized qwen50m
```

Training:

```
diffusion objective
```

Output:

```
qwen50m-direct-dlm
```

---

# Phase 5: Qwen3-0.6B A2D

Input:

```
Qwen3-0.6B Base
```

Methods:

1. dLLM-A2D
2. MiniMind-DLM-A2D

Training:

```
LoRA / Adapter
```

Output:

```
qwen3-0.6b-dllm-a2d

qwen3-0.6b-minimind-a2d
```

---

# 7. Runtime Planning

Hardware target:

- AMD AI 9 HX 365
- ROCm
- 32GB RAM
- 24GB GTT

---

# Fast Track (~1 hour)

## Target

Pipeline validation。

Model:

```
Qwen-mini 10M
```

Data:

```
TinyStories small subset
```

完成：

- AR training
- checkpoint
- A2D conversion
- sampling

---

# Normal Track (~3 hours)

## Target

完整小模型实验。

Model:

```
Qwen-mini 50M
```

完成：

- HF AR
- MiniMind AR
- MiniMind-DLM A2D
- dLLM A2D
- Direct DLM

Dataset:

```
TinyStories 50M tokens
```

---

# Extended Track (~1 day)

## Target

扩展实验。

Model:

```
Qwen-mini 150M
```

或者：

```
Qwen3-0.6B LoRA A2D
```

完成：

- scaling experiment
- A2D comparison
- sampling benchmark

---

# 8. Checkpoint Naming Convention

```text
checkpoints/

qwen10m/
    hf-ar/
    minimind-ar/
    dllm-a2d/
    minimind-a2d/
    direct-dlm/


qwen50m/
    hf-ar/
    minimind-ar/
    dllm-a2d/
    minimind-a2d/
    direct-dlm/


qwen3-0.6b/
    dllm-a2d-lora/
    minimind-a2d-lora/
```

---

# 9. Evaluation

统一：

## Training

- loss curve
- convergence speed

## Generation

固定 prompts:

```
Once upon a time
```

```
The future of AI is
```

## Diffusion Metrics

- sampling steps
- generation speed
- output quality

---

# 10. Priority Order

1. Qwen-mini HF AR
2. Qwen-mini MiniMind AR
3. MiniMind-DLM A2D
4. dLLM A2D
5. Qwen3-0.6B A2D
6. Direct DLM
