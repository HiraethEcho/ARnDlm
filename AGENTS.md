# AGENTS.md

## 写代码 —— 分工 (最重要)

| | 谁写 | 位置 | 用什么 |
|---|---|---|---|
| **A 版 (参考实现)** | **AI 写** | `example/` | `transformers` / `dllm` 包 |
| **B 版 (正式实现)** | **你自己写** | `src/` 的七个目录 | 见下 |

`src/` 的七条路线见 `src/README.md`。简言之: `base` `utils` 是公用底座,
剩下五个是路线 —— 同一个功能走两条路 (用包 / 不用包):

| | 不用包 | 用包 |
|---|---|---|
| AR 训练 | `minimind/` (原生 pytorch) | `transformer/` |
| A2D | `a2d/` (minimind-dlm 思路) | `dllm-a2d/` |
| 从零训 DLM | — | `dllm/` |

A/B 对照是**同一路线内部**的对照: 用包的版本和你手写的版本, 在同一 batch
同一权重下输出必须一致。对应表见 `example/README.md`。

## 我能写什么

- **`example/` 里的任何东西** — 参考实现, 正确性基准
- **`src/` 里只有注释** (docstring / `#` 注释)
- 对话里的代码片段

## 我不能写什么

- **`src/` 里的任何实现代码**。哪怕一行。
  包括 `src/transformer/` 和 `src/dllm-a2d/` —— 它们名字上像是「调包」,
  但写在 `src/` 里就按 `src/` 的规矩办。需要参考实现时我在 `example/`
  写一份给你对照。

## 知识沉淀规则 (最重要)

**每次研究或实测得到的新知识, 必须写进文件, 而不是只留在对话里。**

### 三个目录的分工

| | 放什么 | 不放什么 |
|---|---|---|
| `notes/` | **纯知识性内容** — 概念、原理、架构、坑。与机器无关, 换台机器依然成立 | 任何本机专有的数字、时间、路径、状态 |
| `logs/dev/` | **手写开发记录** — 本机实测、文件清单、环境故障、踩过的坑 | 概念解释 (那属于 notes); 程序自动吐的 (那属于 logs/run/) |
| `logs/run/` | **训练运行产物** — 运行日志、`summary.json`、`metrics.csv`、曲线 PNG | 任何手写内容 |

判断方法: **「换一台机器这句话还成立吗?」** 成立 → `notes/`, 不成立 → `logs/dev/`。
再问: **「人写的还是程序吐的?」** 程序吐的 → `logs/run/`。

### 写作要求

- **简洁、准确、紧凑。** 不写过渡句, 不重复, 不铺垫。
- **`notes/` 只写实测或注明来源的数字。** 不写「据说」「大概」。
  外部来源给出处 (URL 或 `file:line`)。
- **发现旧笔记有错就改掉, 不要在旁边加一条纠正。** 一份文件只留一个真相。
- **概念类内容优先用表格和代码块**, 少用散文。
- 记完记得在 `logs/dev/01-local-state.md` 的「项目当前进度」里更新。

## 写代码

- **注释写在 `src/`。** 每个文件的 docstring 写: 作用 / 要写什么 / 注意 / 依赖 / 验收。
- **`test/` 的注释解释 Python 用法** (import 形式、`nn.Module`、
  `view` vs `reshape`、推导式、`try/except`、`del` + `empty_cache`),
  因为这是学 Python 的项目。**但要精简** —— 只写「这行为什么这么写」,
  不写教科书段落。
- **`example/` 的注释解释设计意图和源码机制** (引 `transformers` 源码行号那种),
  同样精简。
- **自己写只限于 A2D 那 ~300 行核心逻辑。** 其他一律参考
  `example/` 里怎么用 `transformers` / `safetensors` / `tokenizers` / `peft`。
- **注释和 docstring 用英文**, 对话和文档用中文。

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
- **系统 `python` 里装的是 ROCm 7.2 的 torch 2.14 + 坏掉的 transformers
  (huggingface-hub 2.0 冲突), 不要用系统解释器。**
- **直连 huggingface.com 会 timeout**, 用 `HF_ENDPOINT=https://hf-mirror.com` (已在 dev.zsh 设好)。
- **SwanLab 需要 `SWANLAB_API_KEY`**。令牌在 `~/.config/zsh/dev.zsh` 里但**被注释掉了**,
  用之前先自己 export。训练脚本要能在这两个状态下都跑通。

## 性能约束

实测数据见 `logs/dev/02-hardware-perf.md` (从 `~/Projects/dlm` 继承, 换机器要重测)。
**关键约束, 直接影响配置选择**:

- **瓶颈是 attention 的 backward kernel**, 不是带宽, 不是算力。换 SDPA backend 无解。
- **`max_seq_len ≤ 512`** — attention 是 O(T²)。
- **`batch 16 就够** — B=1 到 B=64 吞吐只差 30-53%。
- **不开 gradient checkpointing** — 本机上小模型慢 18-23%, 只省 3GB。
- **packing (pad=False) 吞吐高 1.71x** — 带宽受限机器上零 padding 浪费很值。
- 实测量级: **~3,700-4,700 tok/s**。dlm 那个规模 5 小时预算 ≈ 55-75M tokens。
- **run-to-run 方差可达 ±20%**, 小矩阵和短 benchmark 尤其不稳。

**自训模型的目标参数量先不钉死。** 冒烟测试测出实际吞吐后再定 ——
参数量、序列长度、数据量是耦合的, 先拍数字再测是本末倒置。

## 报告数字

- **必须同时报 `train-only` 和 `wall-clock` 两个吞吐。** 验证和采样会污染分母。
- **不要只报整体数字** — 逐项测到具体哪个 kernel 慢才可行动。
- **`Predicted` 和 `Measured` 分两列写, 不要合并。**
- benchmark ≥ 400 步, 先丢弃预热。
- 完整 checklist 见 `notes/05-measurement.md`。

## 核心指标是重复率, 不是 loss

DLM 失败最典型的表现就是**复读**, 而 loss 照样下降。
参见 `notes/03-minimind-dlm.md`。所有 T2/T3 的评测都要报 repetition rate。

## 沟通

- 中英混合说, **code 和 comments 写英文**, technical terms 保持英文。
- 简洁。不要写不必要的段落。

## 运行

- **长任务用非阻断 subagent 启动 python, 让 subagent 监督进程, 完成时把结果
  返还给主 agent。** 不要在主面板阻塞等待, 也不要用裸 `nohup &` 之后失联。
- 大文件一律走软链接到 `/mnt/extra` (`data/` `checkpoints/`), 由
  `scripts/setup_symlinks.sh` 创建, 全部在 .gitignore 里。
- `logs/run/` 不链接 (纯文本), 也不进 git。`logs/dev/` **进 git**。

## 当前路线

见 `ROADMAP.md`。**必须有 AR baseline。** A2D 之前先有可用权重,
否则无法判断 DLM 的输出是「变差了」还是「本来就差」。
