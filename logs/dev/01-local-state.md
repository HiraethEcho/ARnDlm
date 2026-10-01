# 本机状态

## 环境

```
.venv/                 uv 建, Python 3.14.7
torch                  2.12.0+rocm10.0.0  | rocm 10.0.0
transformers           4.57.0   ← dllm 的 pin (PyPI 上 yank, wheel 可用, 已实测)
peft                   0.17.1
accelerate             (peft 的硬依赖)
GPU                    AMD Radeon 880M (gfx1150, Strix Point / Ryzen AI 9 365)
系统 python             torch 2.14.0+hip (ROCm 7.2) + transformers 5.17.0 —— **坏的, 不用**
```

**uv 环境可行性已验证 (2026-10-01)**:
`uv pip install --dry-run` 单次解析成功, torch 2.12.0+rocm10.0.0 (cp314 /
gfx1150) + transformers 4.57.0 + peft 0.17.1 + datasets 4.2.0 全部共存。
必需 flag: `--index-strategy unsafe-best-match` —— ROCm 索引只有 torch,
PyPI 只有 HF 栈, uv 默认不跨索引取包, 不加就报 unsatisfiable。

**踩过的坑**:

1. **`peft` 硬依赖 `accelerate>=0.21.0`。** 一开始以为 accelerate 是
   「分布式训练才用得上」而砍掉了, 结果装 peft 时 uv 顺手要把整个 torch
   重解一遍 (多下 1.4GB CUDA 包)。accelerate 本身是纯 Python, 不带
   deepspeed, 装上没有代价 —— **显式装上**。
2. **`transformers==4.57.0` 在 PyPI 上是 yank 状态**
   (`Error in the setup causing installation issues`)。wheel 本身可用,
   已实测能装能跑 (Qwen3 forward/backward/generate 全通)。
   **不要因为 yank 就升版本** —— 5.x 改了 `create_causal_mask` 签名,
   `dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` 接不上。

## /mnt/extra 布局

```
/mnt/extra                     ext4
├── data/
│   ├── raw/
│   │   ├── pretrain_t2t_mini.jsonl   1.24 GB
│   │   └── sft_t2t_mini.jsonl        1.74 GB
│   └── prepared/minimind3-6400/
│       ├── tokens.bin         665 MB, uint16, 332,495,324 token
│       ├── doc_lengths.npy    int32, (1270238,) mean 261.8 max 11802
│       └── meta.json          vocab_size=6400, n_docs=1270238,
│                              total_tokens=332495324, val_frac=0.01
├── ckpt/                      checkpoints/ 链到这里
├── hf/hub/
│   ├── models--Qwen--Qwen3-0.6B/       T2c 的目标, 已下载
│   └── datasets--openai--gsm8k/
└── LAYOUT.md
```

### 语料里最短的文档是退化碎片

`len < 16` 的文档有 **613 篇 (0.05%)**, 内容长这样:

```
'CCertainly'   '200Hello'   'HHello'   'meHello'   '2023Hello'
```

**数据本身是好的** —— 随机和中位长度文档 decode 出来都是通顺中文。
这一点要注意的地方: **按长度过滤数据会优先留下这批碎片**, 而不是好样本。
`test/02-data-pipeline.py` 早期版本就是拿「最短 200 篇」拼块来 decode,
结果看起来像数据损坏, 其实是选错了样本。

## 软链接

```
data        -> /mnt/extra/data
checkpoints -> /mnt/extra/ckpt
```

由 `scripts/setup_symlinks.sh` 创建, 全在 .gitignore 里。`.gitignore` 里
必须写 `/data` 而不是 `/data/` —— git 的带斜杠写法只匹配目录, 而软链接
对 git 是个文件, 匹配不到, 结果软链接本身会被提交进仓库。

## 环境故障 / 待办

- **SwanLab 令牌在 `~/.config/zsh/dev.zsh:46`, 但被注释掉了** ——
  `SWANLAB_API_KEY` 没有 export。训练脚本要能在没 export 时正常跑通。
- **`dllm` 未安装。** `./scripts/install_dllm.sh` 还没跑过。
- **`~/Public/dllm/` 是 dLLM 包** (pip 名 `dllm`, 不是 `dlm`)。
  `~/Public/dlm/` **不存在** —— memo 里写错了。
  `dllm` 的 A2D **只实现了 llama / qwen2 / qwen3 三个架构**。
- `~/Public/minimind-dlm/` 不存在。「minimind-dlm」指的是
  <https://jingyaogong.github.io/space/dllm_intro> 这篇博客 +
  minimind discussion #618, 知识已收进 `notes/03-minimind-dlm.md`。

## 项目当前进度

**scaffold 已建, 训练未开始。** `src/` 七个目录全是空的 (只有 docstring)。

```
已建:
  AGENTS.md / USAGE.md / ROADMAP.md / README.md
  requirements.txt / .gitignore
  configs/{README.md, base.yaml}
  notes/{01,02,03-minimind-dlm,04,05}
  logs/{dev,run}/.gitkeep
  scripts/{setup_env.sh, setup_symlinks.sh, install_dllm.sh, plot.py}
  example/README.md                      (A/B 契约, 代码待 T1a/T2a)
  test/{README.md, 01..05}               01/02/04/05 已跑通, 03 待 peft
  src/{base,utils,transformer,minimind,a2d,dllm-a2d,dllm}/__init__.py
  src/base/tokenizer.json                6400 BPE (sha256 71f32c68…)
```

**没做**: `.venv` 未创建 · `dllm` 未装 · `configs/` 只有 base.yaml ·
`src/` 没有任何实现 · 没训过任何模型。

## 下一步

1. `./scripts/setup_env.sh` + `./scripts/install_dllm.sh` (要下 2-3GB, 约 10-20 分钟)
2. 冒烟测试测吞吐, 再定参数量 / batch / seq_len —— **先测后定, 不要反过来**
3. T1a: `src/transformer/`
