# example/teach/ — B 版教学计划

**这里是你（B 版）要写的东西的施工图。** 每完成一格 A 版，就补细对应的一步。

A 版在 `example/` 全部跑通之前，这些步骤只是骨架；跑通一个 milestone 补一节。
执行顺序见 [`../PLAN.md`](../PLAN.md) §10。

## 步骤索引

| Step | 你写什么 | A 版参考 | 对应矩阵格 | 状态 |
|---|---|---|---|---|
| 0 | `utils/` | `example/common/` | 全部（底座） | ✅ 已细化 |
| 1 | `models/hf_qwen/` | `example/base/hf/model.py` | 行1 base | ✅ 已细化 |
| 2 | `training/ar_hf/` | `example/base/hf/train_ar.py` | 行1 base | ✅ 已细化 |
| 3 | `models/minimind_qwen/` | `example/base/native/model.py` | 行2 base | ✅ 已细化 |
| 4 | `training/ar_native/` | `example/base/native/train_ar.py` | 行2 base | ✅ 已细化 |
| 5 | `diffusion/` corruption + scheduler | `example/base/*/a2d_minimind.py` | 三格方法 | ✅ 已细化 |
| 6 | `diffusion/loss.py` | 同上 + `test/04` | 三格方法 | ✅ 已细化 |
| 7 | `diffusion/sampler.py` | 同上 | 三格方法 | ✅ 已细化 |
| 8 | `training/a2d_minimind/` | `example/base/*/a2d_minimind.py` | MiniMind-A2D 列 | ✅ 已细化 |
| 9 | `training/a2d_dllm/` | `example/base/*/a2d_dllm.py` | dLLM-A2D 列 | ✅ 已细化 |
| 10 | `training/direct_dlm/` | `example/base/*/direct_dlm.py` | Direct-DLM 列 | ✅ 已细化 |
| 11 | `evaluation/` | `example/common/metrics.py` + 各格采样 | 全部 | ✅ 已细化 |
| 12 | Qwen3-0.6B LoRA 两格 | `example/base/qwen3_0p6b/` | 行3 两格 | ✅ 已细化 |
| 13 | Tier 1 定档 | `logs/dev/02-hardware-perf.md` | 收尾 | 待写 |

## 怎么用

1. 读对应的一节，看「你要写什么」和「A 版在哪个文件」。
2. 自己写 B 版，**不要复制 A 版代码** —— 复制过来就没有对照意义了。
3. 跑该节的验收命令。两个实现不一致的地方就是你写错了的地方。

## 通则（每一步都适用）

| 规则 | 出处 |
|---|---|
| 同时报 train-only 和 wall-clock 两个吞吐 | `notes/05-measurement.md` Bug 1 |
| 分子分母范围一致（都增量或都累计） | 同上 Bug 2 |
| DLM 必须报 repetition rate，不能只报 loss | `notes/03-minimind-dlm.md` §11 |
| 用 4.57.0 的 API，不要写 5.x | `AGENTS.md` |
| 数值先在 CPU 上对拍，再上 GPU | `logs/dev/01-local-state.md` |