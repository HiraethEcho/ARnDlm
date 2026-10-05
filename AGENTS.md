# AGENTS.md

此仓库有学习和实验两个目的

- 学习 python pytorch transformers peft safetensors accelerate 的使用
- 实验对比 AR 模型、AR 后训练成 DLM、原生 DLM

## 学习

### 知识沉淀规则

**每次研究或实测得到的新知识, 必须写进文件, 而不是只留在对话里。**

### 三个目录的分工

|             | 放什么                                                                                 | 不放什么                                                    |
| ----------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `notes/`    | **事实笔记** — 换开发环境依然成立：概念、原理、架构、外部知识、规格限制                 | 本机状态、开发过程、时间戳                                  |
| `logs/dev/` | **开发笔记** — 开发中学到的：本机实测、环境故障、踩坑与解法、参考路径                   | 与开发过程无关的纯概念（→ notes）；程序吐的（→ logs/run/）  |
| `logs/run/` | **训练运行产物** — 运行日志、`summary.json`、`metrics.csv`、曲线 PNG                    | 任何手写内容                                                |

判断方法: **「换一台机器这句话还成立吗?」** 成立 → `notes/`, 不成立 → `logs/dev/`。
再问: **「人写的还是程序吐的?」** 程序吐的 → `logs/run/`。

### 写作要求

- **简洁、准确、紧凑。** 不写过渡句, 不重复, 不铺垫。
- **`notes/` 只写实测或注明来源的数字。** 不写「据说」「大概」。
  外部来源给出处 (URL 或 `file:line`)。
- **发现旧笔记有错就改掉, 不要在旁边加一条纠正。** 一份文件只留一个真相。
- **概念类内容优先用表格和代码块**, 少用散文。
- 记完记得在 `logs/dev/01-local-state.md` 的「项目当前进度」里更新。

## 代码

- `common/`：底座 —— 配置 / 种子 / 设备 / tokenizer / 数据（含 prepare）/ 指标。
  每条实现 × 每个阶段都用，不属于任何格子。与训练无关的准备也在这里。
- `src/`：格子专属 —— `models/`（trans + native 包，接口一致）、`training/`（各入口 + state）、`eval/`。
- **代码成对**：`x.py` 骨架只有「这个文件是干什么的」注释，代码由人填；同目录 `x-example.py`
  是 AI 写的正确参考实现（每个代码块附一行功能注释）。骨架填完同 seed 同输入逐位对拍 `-example`，
  一致才算完。
- 注释只写功能。坑、注意事项写 `logs/dev/`。

## 环境

- **单 `.venv`** (uv 建, Python 3.14, ROCm 10.0 官方 wheel)。重建: `./scripts/setup_env.sh`。
  `.venv/` 在 .gitignore 里。
- **`transformers` 钉 `4.57.0`** —— 这是 `dllm` 的 pin。它在 PyPI 上是 yank 状态
  (原因: 打包 setup 出错), 但 wheel 本身可用, 已实测。**不要升到 5.x**:
  `dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` 是照 4.57 的
  `create_causal_mask` 写的, 5.x 改了签名, 接不上。
- **不装 deepspeed / wandb / torchvision / torchaudio。**
  deepspeed 是 ZeRO (多卡参数切片) —— 本项目单卡小模型没有多卡可切, 收益 0,
  而且它在 ROCm + Python 3.14 上大概率编译失败。
  纯文本项目用不上 vision。`dllm` 用 `--no-deps` 装
  (`./scripts/install_dllm.sh`), 缺什么补什么。
- **`accelerate` 要装。** 它看着像「分布式训练才用得上」, 但 `peft` 硬依赖它
  (`peft 0.17.1` 要求 `accelerate>=0.21.0`)。它本身是纯 Python, 不带
  deepspeed, 装了没有代价。不显式装的话, uv 解析时会顺手把 torch 重解一遍。
- **直连 huggingface.com 会 timeout**, 用 `HF_ENDPOINT=https://hf-mirror.com` (已在 dev.zsh 设好)。
- **SwanLab 需要 `SWANLAB_API_KEY`**。令牌在 `~/.config/zsh/dev.zsh` 里，没 export 就跳过，别报错

契约性钉版（transformers 4.57.0 / 不装 deepspeed / dllm 打 patch 装）与 5.x 注意事项在 SPEC §5，
本文件不重复。

## 报告数字

- **必须同时报 `train-only` 和 `wall-clock` 两个吞吐。** 验证和采样会污染分母。
  (benchmark 脚本里两者会重合, 因为没有验证采样; 真正的污染源是训练脚本。)
- **不要只报整体数字** — 逐项测到具体哪个 kernel 慢才可行动。
- **`Predicted` 和 `Measured` 分两列写, 不要合并。**
- benchmark ≥ 400 步, 先丢弃预热。
- 完整 checklist 见 `notes/05-measurement.md`。

## 运行

- **长任务用非阻断 subagent 启动 python, 让 subagent 监督进程, 完成时把结果
  返还给主 agent。** 不要在主面板阻塞等待, 也不要用裸 `nohup &` 之后失联。
- 大文件一律走软链接到 `/mnt/extra` (`data/` `checkpoints/`), 由
  `scripts/setup_symlinks.sh` 创建, 全部在 .gitignore 里。
- `logs/run/` 不链接 (纯文本), 也不进 git。`logs/dev/` **进 git**。
