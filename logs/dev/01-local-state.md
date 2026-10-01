# 本机状态

## 环境 (2026-10-01 实测)

```
.venv/                 uv 建, Python 3.14.7, 5.8G
torch                  2.12.0+rocm10.0.0  | hip 7.15.26333
transformers           4.57.0   ← dllm 的 pin (PyPI 上 yank, wheel 可用, 已实测)
peft                   0.17.1
accelerate             1.15.0    (peft 的硬依赖)
datasets               4.2.0     ← 装完 dllm 依赖要压回来 (trl 会顶到 5.x)
trl                    1.14.1    (只为让 dllm 的 rl/ 能 import, 本项目不用)
lm_eval                           (只为让 dllm 的 core/eval 能 import, 本项目不用)
包数                   108 · .venv 6.2G
datasets               4.2.0
tokenizers             0.22.2
GPU                    AMD Radeon 880M (gfx1150), cuda.is_available() = True
系统 python             torch 2.14.0+hip (ROCm 7.2) + transformers 5.17.0 —— **坏的, 不用**
```

**ROCm 后端实测能算**: 512×512 matmul 跑在 GPU 上, 与 CPU 参考最大误差 `1.2e-4`
(fp32 正常范围)。

> ⚠️ **torch 张量默认在 CPU 上。** 建完模型不写 `.to('cuda')` / `.cuda()`,
> 它就在 CPU 上跑 —— 而 `torch.cuda.is_available()` 照样是 True,
> 看不出任何异常。做任何性能测量前先确认 device。

**uv 环境可行性已验证**: `uv pip install --dry-run` 单次解析成功,
torch 2.12.0+rocm10.0.0 (cp314 / gfx1150) + transformers 4.57.0 + peft 0.17.1
+ datasets 4.2.0 全部共存。必需 flag: `--index-strategy unsafe-best-match` ——
ROCm 索引只有 torch, PyPI 只有 HF 栈, uv 默认不跨索引取包, 不加就报
unsatisfiable。装完耗时 12 分钟 (主要是 ROCm wheel 下载)。

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
3. **`multiprocess/connection.py` 会报两条 `SyntaxWarning: 'return' in a
   'finally' block`。** Python 3.14 新告警, 无害, 不用管。

## 基准模型尺寸 (已用解析式验证)

Qwen3 架构 + vocab 6400 (tied embedding), `r=8` LoRA 只挂 attention 四个投影:

| | hidden | layers | Q/KV | head_dim | inter | base params | LoRA 可训 |
|---|---|---|---|---|---|---|---|
| 迷你 | 512 | 8 | 8/4 | 64 | 1536 | — | — |
| **中** | **576** | **12** | **8/4** | **72** | **1728** | **51,478,272** (51.48M) | 387,072 (**0.75%**) |
| 偏大 | 640 | 10 | 8/4 | 80 | 1920 | — | — |

**51,478,272 是实测值, 与逐层解析展开精确吻合** (attention 无 bias,
embed tied 只计一次)。ROADMAP 里预测的「51.5M」得到印证。

LoRA 可训参数 = 12 层 × 32,256, 也在解析式上精确吻合。

**但尺寸还没最终定** —— 先跑冒烟测试测吞吐, 再回头定参数量 / batch /
seq_len。这四个数是耦合的。

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

## dllm: 上游 bug + patch + 一包拉整棵树

### 和它自己 pin 的 transformers 不兼容

`dllm` pin 了 `transformers==4.57.0`, 但三个 A2D 模型类都没定义 `config_class`。
而 4.57 的 `AutoModel.register`（`transformers/models/auto/auto_factory.py:624`）
新增了一致性校验:

```python
if hasattr(model_class, "config_class") and model_class.config_class.__name__ != config_class.__name__:
    raise ValueError("... is not consistent with the config class you passed ...")
```

继承来的 `config_class` 是 `LlamaConfig` / `Qwen2Config` / `Qwen3Config`,
和 `A2D*Config` 对不上 → `import dllm.pipelines.a2d` 在**模块顶层**就炸。

修法: `scripts/patches/dllm-0001-a2d-config-class.patch`, 三个类各加一行
`config_class = A2D*Config`。已实测打上后 import 通过。

**为什么不直接改 `~/Public/dllm`**: 那样这个仓库只能在你自己机器上复现。
`scripts/install_dllm.sh` 走「从上游 clone 到**钉死的 commit** → 打本仓库的
patch → editable 装」。

```
scripts/patches/dllm-0001-a2d-config-class.patch    进 git
~/.cache/ARnDlm/dllm                        缓存, 不进 git
上游: https://github.com/ZHZisZZ/dllm.git
钉在: ca176752fbceec49c6b4777a2c18ae88e4eb10ed
```

用 commit 而不是 `main` 分支名 —— 上游一提交, 环境就静默变了, 而且 patch
可能套不上。

### 一个包把整棵依赖树拉进来

```
dllm/__init__.py        from . import core, data, pipelines, utils
  └─ pipelines/__init__.py
       from . import a2d, bert, dream, editflow, fastdllm, llada, llada2, llada21, rl
```

于是 `import dllm` 连带需要:

| 包 | 被谁拉入 | 本项目用吗 |
|---|---|---|
| `lm_eval` | `core/eval/base.py` | ❌ 不用。拖 30 个包 (scipy/sklearn/pandas/nltk/lxml), +400MB |
| `trl` | `pipelines/rl/grpo/trainer.py` | ❌ 不用。RL 强化学习, 和本项目无关 |
| `tyro` `omegaconf` `rich` `torchmetrics` | a2d + core 直接依赖 | ✅ 间接用到 |

`.venv` 因此从 **5.8G / 46 包** 涨到 **6.2G / 108 包**。

### ⚠️ `trl` 会把 `datasets` 顶到 5.x

装完 `trl` 后 `datasets` 被依赖解析从 **4.2.0 升到 5.0.1**, 而 `dllm`
pin 的是 `4.2.0`。已验证压回 4.2.0 后 `trl` 和 `dllm` 都照常 import。

`install_dllm.sh` 装完会强制回钉, 并验证 torch / transformers / datasets
**三个**关键版本。**只查前两个会漏掉 datasets。**

### 其他坑

- **`git apply --3way` 把改动放进暂存区**, 所以在那个 clone 里
  `git diff` 看着是空的 —— 要 `git diff --cached` 才看得到。
  幂等检测走的是 `git apply --check -R`, 不受影响, 但别拿 `git diff`
  判断 patch 有没有打上。
- **`dllm` 的 submodule `lm-evaluation-harness` 没有初始化**, 所以
  `lm_eval` 只能从 PyPI 装包。
- **`/tmp` 里有 dlm 会话遗留的 `bottleneck.py`**。只要 python 的 sys.path
  里出现 `/tmp` (比如 `python /tmp/foo.py`, 它会把脚本目录插到最前),
  这个 stub 就会遮蔽真的 `bottleneck` 包, 然后 pandas 的可选依赖探测
  会报 `ImportError: Can't determine version for bottleneck`。
  **别在 `/tmp` 里直接跑 python。**

## 项目当前进度

**scaffold 已建并提交 (`1ba36ca`), 训练未开始。** `src/` 七个目录全是空的
(只有 docstring)。

```
已建:
  AGENTS.md / USAGE.md / ROADMAP.md / README.md
  requirements.txt / .gitignore
  configs/{README.md, base.yaml}
  notes/{01,02,03-minimind-dlm,04,05}
  logs/{dev,run}/.gitkeep
  scripts/{setup_env.sh, setup_symlinks.sh, install_dllm.sh, plot.py}
  example/README.md                      (A/B 契约, 代码待 T1a/T2a)
  test/{README.md, 01..05}               5 个全部冒烟通过
  src/{base,utils,transformer,minimind,a2d,dllm-a2d,dllm}/__init__.py
  src/base/tokenizer.json                6400 BPE (sha256 71f32c68…)
```

**没做**: `configs/` 只有 base.yaml · `src/` 没有任何实现 ·
没训过任何模型 · 没测过吞吐。

## 下一步

1. **冒烟测试测吞吐** —— 参数量 / batch / seq_len 都等这个结果再定
2. T1a: `src/transformer/pretrain.py`

## 2026-10-01 环境极小冒烟检查

使用现有 `.venv/bin/python`，未安装或升级依赖，未下载模型。

| 项目 | 实测 |
|---|---|
| Python | 3.14.7 |
| torch | 2.12.0+rocm10.0.0 |
| torch.version.hip | 7.15.26333 |
| GPU | AMD Radeon 880M Graphics，CUDA API 可用 |
| transformers / tokenizers | 4.57.0 / 0.22.2 |
| peft / accelerate | 0.17.1 / 1.15.0 |
| datasets / numpy / safetensors | 4.2.0 / 2.5.3 / 0.8.0 |
| dllm | import、A2D 注册、MDLMTrainer import 通过 |
| LinearAlphaScheduler | weight(0.5) = 1.999996 |

Qwen3 AR 和 dllm A2DQwen3 均使用 vocab=64、hidden=32、intermediate=64、1 layer、2 Q heads、1 KV head、head_dim=16、tied embeddings、eager attention。各模型 11,392 参数，输入 batch=1、长度=6；CPU/GPU forward 均得到有限的 (1, 6, 64) logits。

验证范围仅覆盖导入、scheduler 和 forward。未运行 backward、optimizer、完整 Trainer、采样器或 checkpoint 恢复，不据此宣称完整训练 pipeline 已验证。当前环境无需为上述能力重装；后续阶段见 PLAN.md。
