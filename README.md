# ARnDlm

学习 Python、PyTorch、Transformers 和 DLM，在同一 tokenizer、vocab、数据集与 venv 下比较 AR 和 DLM。

统一契约见 [SPEC.md](SPEC.md)，步骤与验收见 [PLAN.md](PLAN.md)，命令见 [USAGE.md](USAGE.md)。

## 两个版本

| | 实验版 | 实践版 |
|---|---|---|
| 代码 | `example/` | 根目录七个目录 |
| 谁写 | AI | 你 |
| 规模 | Tier 0 · 3.2M 参数 · ctx 128 · 20 万 token | Tier 1 · ~27M 参数 · ctx 512 · 332M token |
| 配置 | `configs/example/tier0.yaml` | `configs/practical/`（待填） |
| 结果 | `example/RESULTS.md` | `RESULTS.md`（待填） |
| checkpoint | `checkpoints/<name>-example/` | `checkpoints/<name>/` |

**实验版已全部完成并合入 main**：SPEC 2.3 的 10 格全通，C1–C6 对拍全过，
实测数字见 `example/RESULTS.md`。它只证明 pipeline 成立，**不代表模型能力**。

实践版是 B 版，每个目录目前只有 docstring。

## 路线与目录

| | 不用包 | 用包 |
|---|---|---|
| AR 训练 | `models/minimind_qwen/` + `training/ar_native/` | `models/hf_qwen/` + `training/ar_hf/` |
| A2D | `training/a2d_minimind/` | `training/a2d_dllm/` |
| 从零训 DLM | — | `training/direct_dlm/` |

A/B 的轴是**用不用包**，不是谁来写：`example/` 里两条路各有一份参考实现，
同权重同 batch 下必须输出一致。

自训路线使用 `tokenizer/tokenizer.json` 的 6400-vocab BPE；Qwen3-0.6B A2D
使用其原生 Qwen tokenizer（MASK 复用 `151662`，不扩词表）。正式规模在冒烟
吞吐测试后决定，实测阶梯见 `logs/dev/02-hardware-perf.md`。
