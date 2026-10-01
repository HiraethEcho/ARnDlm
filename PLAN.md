# ARnDlm PLAN

这是可变的执行计划和进度记录；稳定的项目契约见 [SPEC.md](SPEC.md)。

## 当前进度

| 项目 | 状态 |
|---|---|
| 合并 SPEC 与目录设计 | DONE |
| 根目录 scaffold | DONE |
| uv/ROCm 环境 | 现有版本和 dllm CPU/GPU 极小 forward 通过；训练与恢复待验证 |
| T1a HF AR | PENDING |
| T1b Native AR | PENDING |
| T2a MiniMind-DLM A2D | PENDING |
| T2b dllm A2D | PENDING |
| T2c Qwen3-0.6B A2D | PENDING |
| T3 Direct DLM | PENDING |

## Implementation phases

### Phase 0 · Environment setup

准备 ROCm PyTorch、Transformers、Accelerate、dllm 和 MiniMind 依赖。验收：GPU training、model forward、checkpoint save/load 均可运行。此阶段执行前不改变现有环境。

### Phase 1 · Qwen-mini AR baseline

并行完成：

- HF 版本：Qwen-mini + Transformers + causal LM training，产出 `hf-qwen-ar`；
- Native 版本：Qwen-mini + PyTorch native training，产出 `minimind-qwen-ar`。

先使用 SPEC 的 Tier 0 冒烟配置。完成吞吐测量后，再选择正式模型规模。

### Phase 2 · MiniMind-DLM A2D

输入 AR checkpoint，实现 diffusion corruption、timestep、denoising objective 和 sampler，产出对应的 MiniMind-DLM A2D checkpoint。

### Phase 3 · dllm A2D

使用同一个 AR checkpoint 运行 dllm A2D pipeline，产出 dllm A2D checkpoint，并与 Phase 2 在同 seed 下对拍 loss。

### Phase 4 · Direct DLM

使用选定的自训配置随机初始化模型，按 diffusion objective 训练，产出 Direct-DLM checkpoint。

### Phase 5 · Qwen3-0.6B A2D

对现有 Qwen3-0.6B checkpoint 运行 dLLM-A2D 和 MiniMind-DLM-A2D，使用 LoRA/adapter 并冻结 backbone，分别产出 `qwen3-0.6b-dllm-a2d-lora` 和 `qwen3-0.6b-minimind-a2d-lora`。

## Runtime planning

硬件目标：AMD AI 9 HX 365、ROCm、32GB RAM、24GB GTT。

### Fast track

目标：约 1 小时内完成 pipeline validation。

- 模型：SPEC Tier 0 冒烟配置；
- 数据：小型数据子集；
- 完成：AR training、checkpoint、A2D conversion、sampling。

### Normal track

目标：完成正式小模型实验。

- 模型：Tier 0 吞吐测量后选定的正式规模；
- 完成：HF AR、Native AR、MiniMind-DLM A2D、dllm A2D、Direct DLM；
- 数据量：根据吞吐和时间预算决定，不预先固定。

### Extended track

目标：扩展规模或运行 Qwen3-0.6B LoRA A2D。

- 候选：SPEC Tier 2 或 Qwen3-0.6B；
- 完成：scaling experiment、A2D comparison、sampling benchmark。

## Checkpoint naming

```text
checkpoints/
  qwen-mini/
    hf-ar/
    native-ar/
    dllm-a2d/
    minimind-a2d/
    direct-dlm/
  qwen3-0.6b/
    dllm-a2d-lora/
    minimind-a2d-lora/
```

## Evaluation plan

训练记录 loss curve 和 convergence speed。生成使用固定 prompt，包括：

```text
Once upon a time
The future of AI is
```

DLM 额外记录 sampling steps、generation speed、output quality 和 repetition rate。

## Priority order

1. Qwen-mini HF AR
2. Qwen-mini Native AR
3. MiniMind-DLM A2D
4. dllm A2D
5. Qwen3-0.6B A2D
6. Direct DLM
