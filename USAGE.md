# USAGE — 命令记录

## 首次设置

```sh
# 1. 软链接: data/ -> /mnt/extra/data, checkpoints/ -> /mnt/extra/ckpt
./scripts/setup_symlinks.sh

# 2. 建 .venv 并装 ROCm 10.0 的 torch + HF 栈 (几个 GB, 慢, 约 10-20 分钟)
./scripts/setup_env.sh

# 3. 装 dllm 包 (--no-deps, 见 USAGE 的「为什么 --no-deps」)
./scripts/install_dllm.sh

# 4. 冒烟自检 —— 确认硬件数字没变
./test/01-tokenizer-mask.py
```

## 环境

```
.venv/                 uv 建的, Python 3.14.7, 6.2G
torch                  2.12.0+rocm10.0.0  | hip 7.15.26333
transformers           4.57.0     ← dllm 的 pin, 不要升
peft                   0.17.1
accelerate             1.15.0     (peft 的硬依赖)
datasets               4.2.0      ← 装完 dllm 依赖要压回来, trl 会顶到 5.x
trl                    1.14.1     (只为让 dllm 的 rl/ 能 import, 本项目不用)
lm_eval                             (只为让 dllm 的 core/eval 能 import, 本项目不用)
tokenizers / safetensors / numpy / pyyaml / tqdm / matplotlib
包数                   108
GPU                    AMD Radeon 880M (gfx1150)
索引                    https://stable.repo.amd.com/rocm/whl-next/
```

### 五个必须知道的安装细节

**1. `--index-strategy unsafe-best-match` 是必需的。**
ROCm 索引只提供 `torch`, PyPI 只提供 HF 栈。uv 默认只从第一个含该包的索引取版本,
不跨索引, 所以不加这个 flag 会解析失败。`setup_env.sh` 已经写了。

**2. `transformers==4.57.0` 在 PyPI 上是 yank 状态** (原因: `Error in the setup
causing installation issues`)。wheel 本身可用, 已实测能装能跑。
**不要因为看到 yank 就升版本** —— 5.x 改了 `create_causal_mask` 签名,
`dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` 接不上。

**3. `dllm` 用 `--no-deps` 装。**
它的 `pyproject.toml` pin 了 `deepspeed==0.18.0` / `accelerate==1.11.0` / `wandb`,
本项目单卡小模型完全用不上, 而且 deepspeed 在 ROCm + Python 3.14 上大概率编译失败。
只装 `dllm` 本体, 缺什么补什么。

但 `dllm` 是**一个包把整棵依赖树拉进来**的设计: `dllm/__init__.py` 顶层
`from . import core, data, pipelines, utils`, 而 `pipelines/__init__.py` 又
`from . import a2d, bert, dream, editflow, fastdllm, llada, llada2, llada21, rl`。
所以 `import dllm` 会连带需要 `lm_eval` (拖 30 个包, +400MB) 和 `trl` ——
**本项目一个都不会用**, 但不装就 import 不了。`.venv` 因此从 5.8G 涨到 6.2G。

**4. `accelerate` 装, 但只装它一个。**
它不是「分布式训练专用」—— **`peft` 硬依赖 `accelerate>=0.21.0`**。
它本身是纯 Python, 不拖 deepspeed, 装上没有代价。

**5. 装完 dllm 依赖要把 `datasets` 压回 4.2.0。**
`trl` 的依赖解析会把 `datasets` 顶到 5.x, 而 `dllm` pin 的是 `4.2.0`。
`install_dllm.sh` 装完会强制回钉, 并验证 torch / transformers / datasets
三个关键版本都没被动。**只查 torch 和 transformers 会漏掉 datasets。**

## dllm 的上游 bug 和 patch

**`dllm` 和它自己 pin 的 transformers 不兼容。** 它 pin 了
`transformers==4.57.0`, 但三个 A2D 模型类（`llama`/`qwen2`/`qwen3`）都没定义
`config_class`; 而 4.57 的 `AutoModel.register`（`auto_factory.py:624`）
新增了一致性校验:

```python
if hasattr(model_class, "config_class") and model_class.config_class.__name__ != config_class.__name__:
    raise ValueError(...)
```

继承来的 `config_class` 是 `LlamaConfig`/`Qwen2Config`/`Qwen3Config`,
和 `A2D*Config` 对不上 → `import dllm.pipelines.a2d` 直接炸。

修法是 `scripts/patches/dllm-0001-a2d-config-class.patch`: 给三个类各补一行
`config_class = A2D*Config`。`install_dllm.sh` 自动打上。

**为什么 patch 而不是直接改 `~/Public/dllm`:** 那样这个仓库就只能在你自己
这台机器上复现。`install_dllm.sh` 走「从上游 clone 到**钉死的 commit** →
打本仓库的 patch → editable 装」, 换台机器跑一遍结果完全一致。

钉 commit 而不是用 `main` 分支名 —— 否则上游一提交, 你的环境就悄悄变了,
而且 patch 可能套不上。

```
scripts/patches/dllm-0001-a2d-config-class.patch    ← 进 git, 可复现
缓存位置: ~/.cache/ARnDlm/dllm              ← 不进 git
```

## 为什么不用系统 python 包

系统装的是 `python-pytorch-opt-rocm` 2.14.0+hip (ROCm 7.2), 和
`transformers` + `huggingface-hub 2.0.0` 有版本矛盾 (Arch 打包 bug,
详见 `logs/dev/01-local-state.md`)。干净的 `.venv` 隔离掉这个问题。

## 数据

已经准备好, 不需要重跑 (`data/` 是软链接):

```
data/raw/pretrain_t2t_mini.jsonl     1.24 GB
data/raw/sft_t2t_mini.jsonl          1.74 GB
data/prepared/minimind3-6400/
  tokens.bin                         665 MB, uint16, 332,495,324 token
  doc_lengths.npy                    int32, (1270238,)
  meta.json                          vocab=6400, n_docs=1270238
```

tokenizer 在 `tokenizer/tokenizer.json` (BPE, vocab 6400, 6108 merges)。

| | id | token |
|---|---|---|
| PAD | 0 | `<|endoftext|>` |
| BOS | 1 | `<|im_start|>` |
| EOS | 2 | `<|im_end|>` |
| **MASK (T1/T2)** | **27** | `<|buffer1|>` |
| **MASK (Qwen3-0.6B)** | **151669** | 空闲 slot, `convert_ids_to_tokens` 返回 `None` |

## 跑测试

```sh
.venv/bin/python test/01-tokenizer-mask.py
.venv/bin/python test/02-data-pipeline.py
.venv/bin/python test/03-lora-inject.py
.venv/bin/python test/04-dllm-loss-check.py
.venv/bin/python test/05-repetition-rate.py
```

索引见 `test/README.md`。测试**不切目录也能跑** (没用 `from _model import`)。
`03`/`04` 需要先装好 `peft` / `dllm`。

## 看训练曲线

trainer 每次 log 写两份, 都是程序产物, 落在 `logs/run/<实验名>/`:

```
summary.json     唯一真相, 追加写, 每行一个 JSON 对象
metrics.csv      同样内容的表格版, 方便直接丢进 Excel
```

```sh
.venv/bin/python scripts/plot.py logs/run/example/hf-ar-example
```

SwanLab 是**可选**的。想用先 export 令牌 —— 令牌在 `~/.config/zsh/dev.zsh` 里
但被注释掉了:

```sh
export SWANLAB_API_KEY=...
```

没 export 时训练脚本要正常跑通, 只是不上传。

## 先学哪一步

```sh
ar-hf  →  ar-hf-sft  →  ar-hf-sft-dllm_a2d
```

**逐步索引在 [`example/teach/README.md`](example/teach/README.md)**，9 个文件按依赖排好。
`LEARN.md` 是在独立工作区动手写实践版时的入口。

## 跑实验版（`example/`，29 格全通）

**两条命令跑完全部 29 格。** 都支持 `--cells-only`（跳过单元测试）和 `--quick`
（20 步小语料，不是 RESULTS.md 引用的那次）。

```sh
# 一半：预训练 / A2D / Direct-DLM / LoRA   10 格 + C1..C6 + decode-bench (约 25 min)
bash example/smoke_all.sh

# 另一半：SFT 列 + DPO 列                   19 格 + C7          (约 25 min)
bash example/smoke_sft.sh

# 共享机器上加 --cells-only：跳过单元测试。它们是故意重 CPU 的
# （一节在 CPU 上训 20 步模型，另一节重新 tokenize 1.7 GB 语料）。
bash example/smoke_all.sh --cells-only
bash example/smoke_sft.sh --cells-only

# 单个测试文件 (13 个, 失败非零退出 —— 不是 pytest, 环境里没装)
.venv/bin/python example/tests/test_sft_flatten.py

# 单格调试 (Tier 0, 200 步, 几十秒)
.venv/bin/python example/base/hf/train_ar.py
.venv/bin/python example/base/hf/train_dlm_sft.py --cell ar-3.2m-hf-sft
```

**decode 步数的两个语义**（OPEN-3）：`--dlm-steps` 是**每块**步数（本项目的约定），
`--dlm-total-steps` 是**全序列**预算，会按块数摊薄（`ceil(total / num_blocks)`，
与 dllm 内部一致）。`--dlm-total-steps 8` 在 4 块下等于 `--dlm-steps 2`。

配置从 YAML 来，**代码里没有字面量**：

```sh
configs/example/tier0.yaml          # 实验版, 已填
configs/practical/tier1.yaml       # 实践版, 还没有
```

## 测吞吐

```sh
.venv/bin/python scripts/bench_throughput.py --seq-len 512 --batch 8
.venv/bin/python scripts/bench_throughput.py --sweep --steps 100 --warmup 50
```

⚠️ **这张 880M 是核显，桌面和别的会话都在抢它。** 同一配置离散度可达 2.7 倍。
测之前 `rocm-smi --showuse` 确认卡空，同一配置跑 3 次报离散度，预热 ≥ 50 步。
方法学见 `notes/05-measurement.md`。

## 准备数据

```sh
# MiniMind 全量 (约 22 GiB, 下载约 6.6 MB/s, 可断点续传)
.venv/bin/python scripts/download_minimind.py --list
.venv/bin/python scripts/download_minimind.py

# HF 三个数据集 (choices: tinystories / openwebtext / wikipedia-zh)
.venv/bin/python scripts/download_data.py tinystories --max-docs 10000
.venv/bin/python scripts/download_data.py openwebtext
.venv/bin/python scripts/download_data.py wikipedia-zh

# tokenize 成训练格式
.venv/bin/python scripts/prepare_data.py data/raw/minimind_dataset/pretrain_t2t_mini.jsonl \
    --output data/prepared/minimind3-6400 --append-eos
```

**`datasets` 4.2.0 在本机不能用**（Python 3.14 + dill 0.4.0 报
`Pickler._batch_setitems() takes 2 positional arguments but 3 were given`），
`download_data.py` 会自动回退到 parquet backend。不要动 pin。

⚠️ **SFT / DPO / RL 数据用 `conversations` 格式，喂给 `prepare_data.py` 会静默
产出 0 token。** 已下载但还不能用，见 `notes/06-datasets.md`。

## 跑实践版（B 版，尚未实现）

B 版目录目前只有 docstring。写完之前下面这些命令都不存在：

```sh
# 计划中的入口（尚不存在，写的时候按example/teach/ 的顺序来）
.venv/bin/python -m training.ar_hf.pretrain       --config configs/practical/tier1.yaml
.venv/bin/python -m training.ar_native.pretrain   --config configs/practical/tier1.yaml
.venv/bin/python -m training.a2d_minimind.train   --ckpt checkpoints/<name>/final
.venv/bin/python -m training.a2d_dllm.train       --ckpt checkpoints/<name>/final
.venv/bin/python -m training.direct_dlm.pretrain  --config configs/practical/tier1.yaml
```

写代码的顺序在 `example/teach/00-common-base.md` → `13-tier1.md`，每个文件都写了验收标准。

**长任务用非阻断subagent 启动，让 subagent 监督进程并在完成时回报**
（见 `AGENTS.md` 的「运行」）。

## 换机器 / 链接断了

```sh
./scripts/setup_symlinks.sh
./scripts/setup_env.sh
./scripts/install_dllm.sh
```

换机器后 `logs/dev/02-hardware-perf.md` 的数字全部作废, 重测。
