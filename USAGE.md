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
.venv/                 uv 建的, Python 3.14.7
torch                  2.12.0+rocm10.0.0  | rocm 10.0.0
transformers           4.57.0     ← dllm 的 pin, 不要升
peft                   0.17.1
accelerate             (peft 的硬依赖)
datasets               4.2.0
tokenizers / safetensors / numpy / pyyaml / tqdm / matplotlib
索引                    https://stable.repo.amd.com/rocm/whl-next/
gfx1150 = Radeon 880M (Strix Point / Ryzen AI 9 365)
```

### 三个必须知道的安装细节

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

**4. `accelerate` 装, 但只装它一个。**
它不是「分布式训练专用」—— **`peft` 硬依赖 `accelerate>=0.21.0`**。
它本身是纯 Python, 不拖 deepspeed, 装上没有代价。

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

tokenizer 在 `src/base/tokenizer.json` (BPE, vocab 6400, 6108 merges)。

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
.venv/bin/python scripts/plot.py logs/run/t1_pretrain
```

SwanLab 是**可选**的。想用先 export 令牌 —— 令牌在 `~/.config/zsh/dev.zsh` 里
但被注释掉了:

```sh
export SWANLAB_API_KEY=...
```

没 export 时训练脚本要正常跑通, 只是不上传。

## 训练

还没写。各路线的模块等做到那一步再填 (见 `src/README.md` 的开工顺序):

```sh
# T1a  transformers 包
.venv/bin/python -m src.transformer.pretrain --config configs/t1_pretrain.yaml
.venv/bin/python -m src.transformer.lora     --ckpt checkpoints/t1/final.pt

# T1b  原生 pytorch
.venv/bin/python -m src.minimind.pretrain   --config configs/t1_pretrain_native.yaml
.venv/bin/python -m src.minimind.sample      --ckpt checkpoints/t1_native/final.pt

# T2   A2D (两条路必须对拍)
.venv/bin/python -m src.a2d.train            --ckpt checkpoints/t1_native/final.pt
.venv/bin/python -m src.dllm_a2d.train       --ckpt checkpoints/t1_native/final.pt

# T3   直接用 dllm 包训
.venv/bin/python -m src.dllm.pretrain        --config configs/t3_dllm_pretrain.yaml
```

**长任务用非阻断 subagent 启动, 让 subagent 监督进程并在完成时回报。**
(见 `memo-more.md`)

## 换机器 / 链接断了

```sh
./scripts/setup_symlinks.sh
./scripts/setup_env.sh
./scripts/install_dllm.sh
```

换机器后 `logs/dev/02-hardware-perf.md` 的数字全部作废, 重测。
