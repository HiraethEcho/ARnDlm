# STATE — 当前能跑什么（2026-10-07，main `0bfe901`）

一句话：**`-example.py` 全部可跑，`x.py` 全部还是骨架。** 本文只记事实与能力边界；
契约在 SPEC.md，计划在 PLAN.md，施工顺序在 ROADMAP.md。

## 1. 两个版本的关系

| | `x.py`（骨架） | `x-example.py`（参考实现） |
|---|---|---|
| 内容 | 4–6 行 docstring，0 行代码 | 完整实现 + 每块功能注释 |
| 谁写 | 你 | AI |
| 验收 | 填完与 `-example` 同 seed 同输入**逐位对拍** | 各 `test/NN-*.py` |
| 现在能干嘛 | 不能跑 | **能下载数据、能训练、能评测、能 REPL** |

`x-example.py` 之间用 importlib 桥互相加载（文件名有连字符，不能 import）——
这只是临时手段，x.py 填完后桥全部拆掉换正常 import（见 ROADMAP 最后一格）。

## 2. 数据能力

### 已就位（`/mnt/extra/data`，磁盘余 409 G）

| prepared 目录 | token | 文档/块 | 用途 |
|---|---:|---:|---|
| `minimind3-6400` | 332,495,324 | 1,270,238 docs | **tier1 主实验**语料 |
| `wikitext-6400` | 194,509,960 | 29,444 docs | 英文书面语（2026-10-07 新加） |
| `wikipedia-zh-6400` | 38,571,096 | 10,000 docs | 中文百科 |
| `openwebtext-6400` | 17,174,450 | 10,000 docs | 英文网络语料 |
| `tinystories-6400` | 3,110,358 | 10,000 docs | **smoke 默认语料** |
| `sft-mini-6400` | 524,288（4,096 块 × 128） | 2,166 段对话 | SFT / DLM-SFT；supervised 260,590 |

raw（jsonl 原档）：`minimind_dataset/` 11 个文件（pretrain_t2t 7.9 G、sft_t2t 13.4 G、
dpo.jsonl 51 M/17,166 对、rlaif、agent_rl×2、lora_×3）、`tinystories.jsonl`、
`openwebtext.jsonl`、`wikipedia-zh.jsonl`、`wikitext-103.jsonl` + parquet 分片。

### 能做的转换（仓内脚本）

```sh
# ① 普通语料: jsonl({"text":...}) -> prepared 三件套（tokens.bin/doc_lengths.npy/meta.json）
.venv/bin/python common/prepare-example.py <in.jsonl> <out_dir>

# ② SFT 数据: jsonl({"conversations":[...]}) -> 预切块 + assistant_spans.npy
.venv/bin/python common/sft-prepare-example.py <in.jsonl> <out_dir> --max-blocks 4096

# ③ wikitext-103 下载（走 hf-mirror 直链 parquet; 需先有 parquet 分片）
.venv/bin/python scripts/make_wikitext_jsonl.py <out.jsonl> <parquet_dir>
```

**不能做**：仓内没有 minimind 数据集的下载器（数据已在盘上，来源见 notes/06）。
`common/prepare-example.py` 不联网——它只吃本地 jsonl。

## 3. 训练能力

入口只有一个：`src/training/main-example.py`。`--stage` × `--impl` 的合法组合：

| stage | impl | 起点 | 数据 | 产物名例 |
|---|---|---|---|---|
| `pretrain` | `trans` / `native` | 随机初始化（或 `--base` 共享 init） | 普通语料 | `ar-3.2m-trans-smoke` |
| `sft` | `trans` / `native` | `--base` AR 权重 | sft-mini-6400 | `ar-3.2m-trans-sft-smoke` |
| `a2d` | `mini` / `dllm` | `--base` SFT/AR 权重 | 普通语料；`--sft-data` 则 assistant 区间 | `ar-3.2m-trans-sft-mini_a2d-smoke` |
| `dlm` | `trans` / `dllm` | **随机初始化**（Direct-DLM） | 普通语料 | `dlm-3.2m-trans-smoke` |
| `dpo` | `trans` / `native` | `--base` SFT 权重 | `--dpo-data`（默认上游 dpo.jsonl） | `ar-3.2m-trans-sft-dpo-smoke` |

- **规模**：smoke = 3,213,056 参数（已实测）；tier1 = 26,879,488（解析式已验证，
  **未训练过**，预算 ≈ 11.6 h/次 pretrain）。
- **步数**：`--steps N` 任意；缺省取 config（smoke 1024、tier1 81,174）。
  已验证跑到 800 步（smoke）、20 步（tier1 未跑）。
- **实测吞吐**（smoke 尺寸，仅参考，SPEC 6.4 规定不引用为性能结论）：
  pretrain ~2.1 万 tok/s、SFT ~5.4 万、A2D ~1.3-5 万、DPO ~1.7 万。
- **恢复**：native 侧 C4 逐位等价（test/07）；trans 侧 HF resume 亦逐位等价（test/07 [6]）。
- **G1 同起点**：`--dump-init DIR` 存初始权重 → 两个 impl 都 `--base DIR`，step-0 loss 差 0.0。

## 4. 训练到哪一步（smoke 实测锚点）

| 格子 | 800 步结果 | 四门槛 |
|---|---|---|
| AR-SFT trans / native | assistant eval 4.78 / 4.58 | t1/2/3/4 全过 |
| G3 原路线（SFT→A2D→SFT）× mini/dllm | 5.42 / 5.35 | t2 有 1 个 prompt ≈0.9 |
| G3 新路线（A2D→SFT）× mini/dllm | 6.65 / 6.72 | t2 失败 |
| native 底座链 | 6.55 | 全过 |
| Direct-DLM→SFT（trans/dllm） | 6.50 | t2 失败（0.66-0.75） |
| DPO（trans/native，1,000 对 × 200 步） | loss 0.693→低 | margin acc 0.700 / 0.645 |

完整表：`RESULTS-smoke.md`（`scripts/collect_results.py` 生成，`--gate` 可作硬门禁）。
**结论一句话**：smoke 上 AR 系能 chat，DLM 系 loss 降但复读率偏高——SPEC 6.4 明说
rep-4 不能给实现排名，正式结论要 tier1。

## 5. 能看/能评

```sh
# 四门槛报告（含逐格 chat_report.json）
.venv/bin/python scripts/collect_results.py            # 汇总 RESULTS-smoke.md
.venv/bin/python scripts/collect_results.py --gate     # 任一门槛 ❌ -> exit 1

# REPL 人肉看输出（两种采样器）
.venv/bin/python src/eval/repl-example.py --ckpt checkpoints/ar-3.2m-trans-sft-smoke \
    --mode chat --sampler ar            # AR 模型: generate
.venv/bin/python src/eval/repl-example.py --ckpt checkpoints/ar-3.2m-trans-mini_a2d-smoke \
    --mode chat --sampler dlm           # DLM/A2D 模型: 双向块状揭示

# 16 个测试（全部应 PASS，从仓库根跑）
for t in test/0*.py test/1*.py; do .venv/bin/python "$t" | tail -1; done
```

## 6. 方便调的参数（configs/*.yaml）

| 段 | 可自由调 | 契约冻结（改=改契约） |
|---|---|---|
| `model` | hidden_size / num_layers / num_heads / num_kv_heads / head_dim / intermediate_size / context_length / rope_theta / rms_norm_eps | vocab_size 必须 6400（= tokenizer）；num_heads % num_kv_heads == 0 |
| `train` | steps / batch_size / lr / dlm_lr / min_lr_ratio / warmup_steps / weight_decay / grad_clip / beta1 / beta2 / seed / log_every / loss_window | — |
| `data` | prepared_dir / max_blocks（0=不限量） | — |
| `runtime` | dtype（bfloat16 / float16 / float32） | eval 恒 fp32（对拍硬要求） |
| `tokens` | — | pad/bos/eos/mask = 0/1/2/27 与 tokenizer 绑定 |

CLI 层额外可调：`--steps`、`--name`、`--data`、`--dpo-pairs`、`--sft-data`、`--base`、`--dump-init`。
**加新键会报错**（config 加载器一次列全未知键），不会静默忽略。

## 7. 还不能做的（诚实清单）

1. **x.py 全部是空的**——本文档说的每项能力都由 `-example.py` 提供。
2. **tier1 一次没跑过**（26.9M 只在解析式与 C1 守门上验过）。
3. **qwen3-0.6b LoRA 行没做**（G4 的一部分；peft/accelerate 已在环境里）。
4. **DPO 只到 smoke**（1,000 对 × 200 步）；RL 只做 DPO，无 PPO/GRPO。
5. **rep-4 门禁在 smoke 档不达标**（这是现象不是 bug，tier1 才判）。
6. native 与 HF 的同 seed 初始权重**不逐位相等**（用 `--base` 共享 init 绕过）。
7. ~~`src/training/dlm.py` 没有 `dlm-example.py` 对~~ —— **2026-10-14 已补**：
   `src/training/dlm-example.py`（Direct-DLM 独立成格）。补的过程中修掉一个真 bug：
   旧代码里 `--stage dlm --impl trans` 静默走了 dllm loss（分支写成
   `if impl == "mini" ... else dllm`），所以「两条 impl 轨迹逐位相同」是假象。
   修后实测：trans eval 5.0192 / dllm 5.0691（相对差 1.0%）。
8. ~~DPO 没有专项测试~~ —— **2026-10-14 已补** `test/16-dpo.py`（19 项：shift 语义
   逐位、policy==ref -> loss==ln2、margin 正负方向、lr×0.1、pair 相邻映射、数据格式）。
   补测试时修掉 `load_dpo_pairs` 的 `max_pairs` 语义 bug（数的是**行**不是 pair：
   旧 `--dpo-pairs 1000` 实际只训 500 对）；重跑后 DPO 实测 margin acc
   trans 0.700 / native 0.645（各 1,000 对 × 200 步）。
9. 无 CI、无 push 脚本；`logs/run/` 与 `checkpoints/` 不进 git。
