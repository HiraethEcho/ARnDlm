# ARnDlm

学习 Python、PyTorch、Transformers 和 DLM，在同一 tokenizer、vocab、数据集与 venv 下比较 AR 和 DLM。

统一契约见 [SPEC.md](SPEC.md)，步骤与验收见 [PLAN.md](PLAN.md)。当前只完成文档和根目录 scaffold；环境和训练实现暂不修改。

| 路线 | 目录 |
|---|---|
| T1a · HF AR | `models/hf_qwen/` + `training/ar_hf/` |
| T1b · Native AR | `models/minimind_qwen/` + `training/ar_native/` |
| T2a · MiniMind-DLM A2D | `training/a2d_minimind/` + `diffusion/` |
| T2b/T2c · dllm A2D | `training/a2d_dllm/` |
| T3 · Direct DLM | `training/direct_dlm/` |

自训路线使用 `tokenizer/tokenizer.json` 的 6400-vocab BPE；Qwen3-0.6B A2D 使用其原生 Qwen tokenizer。冒烟模型配置见 SPEC 第 3 节，正式规模在冒烟吞吐测试后决定。
