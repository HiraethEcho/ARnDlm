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
│   │   ├── sft_t2t_mini.jsonl        1.74 GB
│   │   ├── minimind_dataset/         ← 2026-10-01 加, 11 个 jsonl + 2 个 symlink
│   │   ├── tinystories.jsonl         8.7 MB, 10,000 篇
│   │   ├── openwebtext.jsonl         48.6 MB, 10,000 篇
│   │   └── wikipedia-zh.jsonl        36.1 MB, 10,000 篇
│   └── prepared/
│       ├── minimind3-6400/           主实验数据 (SPEC 3.3)
│       ├── minimind-subsets/         之前切的分片, 当前无 pipeline 引用
│       ├── tinystories-6400/         3,110,358 tok / 10,000 docs
│       ├── openwebtext-6400/         17,174,450 tok / 10,000 docs
│       └── wikipedia-zh-6400/        38,571,096 tok / 10,000 docs
├── ckpt/                      checkpoints/ 链到这里
├── hf/hub/
│   ├── models--Qwen--Qwen3-0.6B/       T2c 的目标, 已下载
│   ├── datasets--openai--gsm8k/
│   ├── datasets--roneneldan--TinyStories/
│   └── datasets--jingyaogong--minimind_dataset/   hf_hub_download 的 cache
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

## 数据集补齐 (2026-10-01)

### `datasets` 4.2.0 在本机彻底不可用

`load_dataset(streaming=True)` 直接报:

```
TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given
```

`.venv` 里是 `dill 0.4.0`, 而 `datasets` 4.2.0 还在用 dill<0.3.9 的旧签名。
**不动 pin** (dill 降级会连带动 `datasets`/`trl` 的解析), 改走
`scripts/download_data.py --backend parquet`: `hf_hub_download` 拉 parquet 分片
+ `pyarrow` 流式读列。实测三个数据集全部走通, `--backend auto` 会先试
`datasets` 再白屏报错回退到 parquet。

### minimind_dataset 全量下载 —— 已完成, 大小逐个对齐

`scripts/download_minimind.py` (新) → `data/raw/minimind_dataset/`,
共 22 GiB 磁盘占用, 11 个 `.jsonl` 全部下载成功。
**每个文件的字节数与 HF `repo_info(siblings[].size)` 逐个比对一致**,
无 `.incomplete` / `.lock` 残留, 前 2000 行 JSON 全部可解析。

```
agent_rl.jsonl        82,036,930     2,001+ 行
agent_rl_math.jsonl   18,372,683     2,001+
dpo.jsonl             53,653,322     17,166
lora_exam.jsonl       24,650,716     2,001+
lora_identity.jsonl        22,789        91
lora_medical.jsonl     34,002,385     2,001+
pretrain_t2t.jsonl  8,275,074,893     8,468,827
rlaif.jsonl           23,754,740     19,502
sft_t2t.jsonl     14,096,018,369     5,109,432
pretrain_t2t_mini.jsonl -> ../pretrain_t2t_mini.jsonl   (symlink)
sft_t2t_mini.jsonl      -> ../sft_t2t_mini.jsonl        (symlink)
```

`data/raw/` 下已有的 `pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl`
**用 symlink 接进新目录而不是重下** —— `prepared/minimind3-6400/meta.json`
记录的 `source_file` 指向原路径, 重下会让两份同 sha 文件并存。

实测下载速度约 **6.6 MB/s** (24.5 GiB 用了 ~63 min)。hf-mirror 是这个速度,
按它估算: 单看 `sft_t2t.jsonl` 14 GiB 就要 ~36 min。

**下次长下载不要挂 subagent**: `dl-minimind` 跑了 63 min 后被 provider
abort (`opencode/space-bunny-free` 超时/中断), 子进程一起没了, 结果也没
返还。而 `hf_hub_download` 本身断点续传, 重跑就接上 —— 长任务应该由主会话
直接跑 (`ctx_execute` 配长 timeout), 或者分小文件多次跑, 不值得赌 subagent
能活到最后。

### 对话数据不能直接喂 `prepare_data.py`

`dpo/rlaif/agent_rl/sft_t2t` 是 `conversations` 格式, 没有 `text` 键。
`read_text()` 会静默跳过 (`if isinstance(text, str)`) —— **不报错, 只是
0 token**。展平成单条 text 需要 chat template, 属于 B 版实现。
详见 `notes/06-datasets.md` 第 2 节。

### 6400 BPE 对中文是超标的

实测 `wikipedia-zh` 达到 **1.07 token/char**, 比字符数还多; minimind 混合语料
1.46 chars/token。选 `max_seq_len` 时必须按 chars/token 换算。

## 吞吐冒烟测试 (2026-10-01)

`scripts/bench_throughput.py` (新)。**第一轮作废, 第二轮 (卡空闲时) 有效。**

### 第一轮: 机器被抢, 结果作废

同配置 (seq=512, batch=16, 28,452,352 参数, bf16, sdpa) 三次独立跑:

| 跑次 | 步数 | train-only tok/s | ms/step |
|---|---:|---:|---:|
| sweep | 100 | 8,824 | 928 |
| repeat a | 400 | 6,054 | 1,353 |
| repeat c | 400 | **3,251** | **2,520** |

**2.7 倍离散度。** 所以 `02-hardware-perf.md` 里写的
「run-to-run 方差 ±20%」在这张卡上是错的 —— 那个 ±20% 是卡空闲时测的。
一旦有别的会话在跑, 方差是 2-3 倍级。

**当时的三方共享方**:

| 共享方 | 证据 |
|---|---|
| 桌面合成器 | `niri` + `brave --type=gpu-process` 常驻 (PID 357557) |
| **另一个 pi 会话** | PID 354003, 14:44 启动, cwd 同样是 `/home/revo/Projects/ARnDlm` |
| 本会话自己的编译 | 前 ~25 步落在预热相里 |

`rocm-smi` 在测量期间报 `GPU use 1%` 但 `VRAM 93%` 且 `No KFD PIDs` ——
即显存被桌面端占着而当前没有计算进程。**「没进程」不等于「卡是空的」。**
复测前 `GPU use 0%`, 离散度就降到 1.4%。

### 第二轮: 有效结果

模型: `Qwen3ForCausalLM`, hidden 512 · 8 层 · Q8/KV4 · head_dim 64 ·
inter 1536 · vocab 6400 · tied embedding = **28,452,352 参数**。
数据 `prepared/minimind3-6400`, packing 定长块零 padding, bf16 + AdamW + sdpa,
不开 gradient checkpointing。计时只包 forward+loss+backward+step,
两侧 `torch.cuda.synchronize()`, warmup 50 步丢弃。

| seq/batch | 跑次 | train-only tok/s | wall tok/s | ms/step | final_loss | peak GB |
|---|---:|---:|---:|---:|---:|---:|
| 512/16 | a | 7,748 | 6,903 | 1057.2 | 5.5486 | 2.69 |
| 512/16 | b | 7,583 | 6,724 | 1080.3 | 5.5486 | 2.69 |
| 512/16 | d | 7,793 | 6,893 | 1051.2 | 5.5486 | 2.69 |
| 512/8 | c | 7,932 | 7,025 | 516.4 | 6.0373 | 1.45 |

**512/16 三次离散度 1.4%** (7583-7793)。`peak_gb` 随 batch 线性 (1.45 → 2.69)。

### 结论

| 项 | 实测 |
|---|---|
| 稳态吞吐 (28.45M, seq 512, B8) | **≈ 7,700-8,000 tok/s** |
| 跑完 332M token (minimind3-6400) | **11.6 h** |
| step 外开销 (组batch / H2D / python) | **0.09%** —— 可忽略 |
| 跑 20M token | 45 min |

⚠️ 第一版这张表的「11.9 h (train-only) / 16.8 h (wall)」里 **wall 那个数是错的** ——
`bench_throughput.py` 的 `wall_tps` 分子只数计时步的 token, 分母却含预热,
违反分子分母范围一致 (notes/05 Bug 2)。修后 wall = 8,067 tok/s ≈ train。

1. **batch 8 和 16 吞吐一样** (7932 vs 7708, 差 2.8% 在噪声内)。
   ms/step 精确翻倍 (516 → 1055) → 加 batch 只加开销不加算力。
   与 `02-hardware-perf.md` 里「B>8 无收益」一致, 现在在本机复现。
   → **batch 取 8**, 显存少一半, 吞吐不亏。
2. 这 7,700 不能和老项目的 3,700-4,700 直接比 —— 老数字是双向 attention +
   64M 参数 + 可能有其他负载, 三个变量都不同。
3. `final_loss` 三次完全相同 (5.5486) —— 计算路径确定, 变的只是墙钟。

### 还没测的 (不能下结论的部分)

| 问题 | 状态 |
|---|---|
| 哪个 kernel 慢 (attention fwd/bwd 逐项) | ❌ 本轮只报整体 |
| 换 SDPA backend 有没有用 | ❌ |
| packing vs padding 的 1.71x | ❌ 没测对照组 |
| 跨文档 attention 污染 | ❌ packing 直接首尾相接, 无 attention 隔离 |
| 参数量扫描 (28M → 51M → 更大) | ❌ 只测了 28M 一个点 |

### 脚本自身的 bug (已修)

1. **sweep 在一个进程里跑所有配置**, `max_memory_allocated()` 从没重置
   → `peak_gb` 单调递增 (0.52 → 5.15), 全是假的。修法: 每个配置前
   `reset_peak_memory_stats()`。
2. **warmup=25 不够**, 前 ~25 步落在编译/预热相。修法: 默认提到 50 步。

### 测吞吐前的 checklist

1. `rocm-smi --showuse` 确认 `GPU use` 接近 0%;
2. 确认没有别的 GPU 任务 (另一个 pi 会话 / 浏览器 GPU 进程);
3. 同一配置跑 **3 次** 报离散度, 不是跑一次;
4. warmup ≥ 50 步;

### test/01 · test/02 的 tokenizer 路径已修 (2026-10-01)

两个脚本都写的是 `ROOT / "src" / "base" / "tokenizer.json"`, 但 scaffold 改版后
tokenizer 已经搬到根目录 `tokenizer/tokenizer.json` —— `src/` 目录不存在。
两个脚本都会在 `Tokenizer.from_file` 处报「找不到文件」。

**这类错误危险在于它发生在验证环节**: 测试全绿给人「数据管线验过了」的错觉,
实际上一次都没跑通。修的是路径常量, 不改验证逻辑。修后两个都 PASS:

| test | 结果 |
|---|---|
| `01-tokenizer-mask.py` | vocab 6400 · mask id 27 (`<|buffer1|>`) 自然文本 56 token 内不出现 · 往返一致 · **编码速度 1,485k tok/s** |
| `02-data-pipeline.py` | 332,495,324 token 三文件自洽 · packing T=512 拼 65 篇首尾全 BOS/EOS · PAD 区全 0 · random/median doc decode 通顺 |

**编码 1,485k tok/s vs 训练 7,700 tok/s = 190 倍余量** → 数据组批永远不会成为
瓶颈, 与 `02-hardware-perf.md` 里旧项目测到的 0.01% 一致。`--num_workers`
在这套配置下可以直接不管。

⚠️ `test/02` 验的是 **token 排列** (BOS/EOS/PAD 位置), **不验 attention 隔离**。
packing 把多篇文档首尾相接时没有 document mask, 模型能看见上一篇的尾巴 ——
与 MiniMind 的做法一致, 但对 DLM 的 block 去噪可能有影响, 还没验。见上表
「还没测的」。

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
  scripts/download_data.py            (加了 parquet backend)
  scripts/download_minimind.py        (新, minimind_dataset 全量)
  scripts/prepare_data.py
  example/README.md                      (A/B 契约, 代码待 T1a/T2a)
  test/{README.md, 01..05}               5 个全部冒烟通过
  src/{base,utils,transformer,minimind,a2d,dllm-a2d,dllm}/__init__.py
  src/base/tokenizer.json                6400 BPE (sha256 71f32c68…)
```

**没做**: `configs/` 只有 base.yaml · `src/` 没有任何实现 ·
没训过任何模型 · 没测过吞吐。

**数据已就位**: minimind3-6400 (332M tok, 主实验) +
tinystories/openwebtext/wikipedia-zh 各 10,000 篇已 tokenize;
minimind 全量 11 个 jsonl / 22 GiB 下载完成并校验。

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
