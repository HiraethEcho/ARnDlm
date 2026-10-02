# ARnDlm PLAN

这是可变的执行计划和进度记录；稳定的项目契约见 [SPEC.md](SPEC.md)。

## 当前进度

**实验版（`example/`）已全部完成并合入 main**（`cf9d708` + `010dbad`）。
SPEC 2.3 的 10 格全部跑通，Tier 0 规模，200 步，实测数字见 `example/RESULTS.md`。

| 项目 | 状态 |
|---|---|
| 合并 SPEC 与目录设计 | DONE |
| 根目录 scaffold | DONE |
| uv/ROCm 环境 | DONE（transformers 4.57.0 + dllm patch 已打，训练与恢复均已验证） |
| T1a HF AR | **DONE** — `example/base/hf/train_ar.py` · loss 8.44→4.70 · eval 4.6356 |
| T1b Native AR | **DONE** — `example/base/native/train_ar.py` · **C1 = 6.91e-06**（阈值 1e-5） |
| T2a MiniMind-DLM A2D | **DONE** — hf + native 两格 · **C2/C3 = 7.6e-08 / 1.5e-07**（阈值 1%） |
| T2b dllm A2D | **DONE** — hf + native 两格 · **C5 = 0 丢失 / 0 漂移** |
| T2c Qwen3-0.6B A2D | **DONE** — LoRA r=8（0.38% 可训）两格 · eval 4.80 / 4.82 |
| T3 Direct DLM | **DONE** — hf + native 两格 · eval 5.64（比 A2D 差 0.67 nat） |
| C4 resume 等价性 | **DONE** — 40 步 vs 30+resume+10 逐步差 0.0 |
| 实验版教学计划 | **DONE** — `example/teach/` step 0-13 全部细化 |
| 实践版实现 | **PENDING** — 七个目录只有 docstring，按 `example/teach/` 逐步写 |
| Tier 1 定档 | **PENDING** — 见 `example/teach/13-tier1.md`，先测吞吐再定参数量 |

**三个方法在两个 base 上的相对关系一致**：A2D eval ≈ 4.97，Direct-DLM ≈ 5.64。
即 AR 权重在 Tier 0 / 20 万 token 上值 **0.67 nat**，与 `notes/03` §11 的判断方向一致。

**rep-4 不可用于排名**：同一方法换一个 base，A2D 在 0.40 ~ 0.81 之间跳
（32 token 样本的噪声），详见 `example/ISSUES.md` OPEN-6。

**实验版只做了预训练**：10 格都是裸文本，没有 chat template、没有 SFT、没有 RL。
`notes/01` 的三阶段目前只用到第一阶段。

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

1. ~~Qwen-mini HF AR~~ **DONE**
2. ~~Qwen-mini Native AR~~ **DONE** (C1 = 6.91e-06)
3. ~~MiniMind-DLM A2D~~ **DONE** (C2/C3 = 7.6e-08 / 1.5e-07)
4. ~~dLLM A2D~~ **DONE** (C5 = 0 lost, 0 drifted)
5. ~~Qwen3-0.6B A2D~~ **DONE** (LoRA 0.38%, eval 4.80)
6. ~~Direct DLM~~ **DONE** (0.67 nat behind A2D)
7. **实践版实现** — 按 `example/teach/` step 0→13 写, 每步对拍
8. **Tier 1 定档** — 先测吞吐阶梯再定参数量 (`example/teach/13-tier1.md`)
9. **T4 后训练** — SFT 先做 (AR 两格 + DLM 两格), DPO 排后。
   完整任务表见 `SPEC.md` §8 (实验版/实践版 两维度, 含依赖顺序和验收命令)
