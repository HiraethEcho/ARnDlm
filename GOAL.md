# 长程任务

这个项目的最底层还是学习写代码，通过实现上述模型训练项目。
先在 example/ 里完成全部代码，跑通全部流程。在一个极小的 config 下测验流程，只关心 pipeline 能否成立、检查代码正确性和简洁性，完全不关心模型能力。
然后教我一步步写正式代码。

用一个新的 agent 和对话历史开始这个任务，先搞清楚我的需求，然后发起一个自主推进的任务，直到所有任务都完成。每项任务都用尽量少的数据量和训练轮次，避免耗时超时。

---

## 给新会话的上手说明（2026-10-01，由 setup 写入）

### 这是什么

- 仓库: `/home/revo/Projects/ARnDlm`（主工作区）
- 本 workspace: `/home/revo/Projects/ARnDlm-example`，git worktree，分支 `example-pipeline`
- 任务边界: **只动 `example/`**。`models/` `training/` 是 B 版，归用户手写，见 `AGENTS.md`。

### 环境已就绪，不要重装

| 项 | 值 |
|---|---|
| Python | 用 `.venv/bin/python`。`.venv` 是**指向主仓库的软链接** |
| torch | 2.12.0+rocm10.0.0，GPU 可用（AMD Radeon 880M，核显，**共享**） |
| transformers / tokenizers | 4.57.0 / 0.22.2（**不要升 5.x**，dllm 的 A2D 接不上） |
| peft / datasets | 0.17.1 / 4.2.0 |
| dllm | 已 editable 安装在 `~/.cache/ARnDlm/dllm`，两个 workspace 共用 |
| data | `data/` → `/mnt/extra/data` |
| checkpoints | `checkpoints/` → `/mnt/extra/ckpt`（**与主工作区共用**） |

`scripts/setup_env.sh` 在这个 workspace 里**不要跑** —— `.venv` 是软链接，
跑它会把包装进主仓库那个 venv。

### 先读这些，别重新发现

1. `AGENTS.md` — 分工契约：A 版（`example/`）AI 写，B 版你写
2. `SPEC.md` — 七条路线、模型规格、tokenizer track
3. `notes/03-minimind-dlm.md` — A2D 原理（已收进公式，不用再找博客）
4. `notes/05-measurement.md` — **测量方法论 checklist**，报数字前必读
5. `notes/06-datasets.md` — 数据集格式与实测压缩比
6. `logs/dev/01-local-state.md` — 本机所有坑（dill 崩溃、抢卡、软链接…）
7. `logs/dev/02-hardware-perf.md` — 硬件实测
8. `example/README.md` — A/B 对照契约

### 已验证可用的命令

```sh
.venv/bin/python test/01-tokenizer-mask.py   # PASS
.venv/bin/python test/02-data-pipeline.py   # PASS
.venv/bin/python scripts/bench_throughput.py --seq-len 512 --batch 8
```

### 已知的两个坑

1. **`datasets` 4.2.0 不能用** —— dill 0.4.0 + Python 3.14 会报
   `Pickler._batch_setitems() takes 2 positional arguments but 3 were given`。
   绕法见 `scripts/download_data.py --backend parquet`。不要去动 pin。
2. **GPU 是共享核显** —— 桌面合成器、Brave、可能还有别的会话都在用。
   上一次同一个配置三次跑出 8,824 / 6,054 / 3,251 tok/s。
   **测吞吐前先 `rocm-smi --showuse` 确认卡是空的，同一配置跑 3 次报离散度。**