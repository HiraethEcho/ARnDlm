# ARnDlm

同一 tokenizer、词表、数据集与 venv 下比较 **AR 语言模型**与 **DLM（扩散语言模型）**，
两者都训到能对话。学习与可测量复现，不追求 SOTA。

- **契约**：[SPEC.md](SPEC.md) —— 五组对比、命名规则、C1–C5 验收、测量口径
- **计划**：[PLAN.md](PLAN.md) —— 阶段表与验收归属
- **使用**：[USAGE.md](USAGE.md) —— 配环境、备数据、跑训练、评测
- **知识**：[notes/](notes/README.md)（事实笔记）· [logs/dev/](logs/dev/README.md)（开发笔记）

## 目录

布局契约在 SPEC §7，此处同款：

```
AGENTS.md  SPEC.md  PLAN.md      # 契约与计划
notes/  logs/dev/                # 事实笔记 / 开发笔记（logs/run/ 不进 git）
common/                          # 底座：每条实现 × 每个阶段都用
  config.py seed.py device.py tokenizer.py(+tokenizer.json)
  data.py prepare.py metrics.py
configs/                         # smoketest.yaml / tier1.yaml
scripts/                         # 环境（setup_env / install_dllm / setup_symlinks + patches）
src/                             # 格子专属代码
  models/  trans.py + native/（包，接口对齐 trans）   # G1 的两条实现
  training/ main.py ar_trans.py ar_native.py state.py
            sft.py dlm.py a2d_dllm.py a2d_mini.py dpo.py
  eval/    chat.py（能 chat 四门槛 + rep-4）
data -> /mnt/extra  checkpoints -> /mnt/extra   # 大文件软链
.venv                            # uv 环境
```

代码成对：`x.py` 骨架（人填实现）＋ `x-example.py` 参考实现（对拍基准）。

## Qwen3：用 transformers 官方实现

`Qwen3Config -> Qwen3ForCausalLM`，不自己写 attention/MLP。与 GPT-2 的区别全由 config
驱动，这些是逐位对拍（C1）能成立的前提：

| 组件 | 位置 | 小模型实参（tier1） |
|---|---|---|
| RoPE | 位置编码 | `rope_theta 10000.0` |
| RMSNorm | 全部 norm（含 QK-Norm） | `rms_norm_eps 1.0e-6` |
| SwiGLU | MLP | `intermediate_size 1408` |
| GQA | `num_kv_heads < num_heads` | 8 头 / 4 KV 头 |
| head_dim 独立参数 | 不必 = hidden/heads | 128（= hidden/heads 时是巧合不是依赖） |
| 无 bias | 大部分线性层 | — |
| tied embedding | 省一份 LM head | true |

字段全部显式写入 `Qwen3Config`（不依赖库默认值——默认值随版本变）。

## 两档配置（configs/）

| | smoketest | tier1 |
|---|---|---|
| 用途 | 验 pipeline 能跑、loss 能降 | 正式对照（AR vs DLM 同 token 流） |
| 形状 | 256 / 2 层 / 2 头 / 1 KV / hd 128 / inter 768 / ctx 128 | 512 / 8 层 / 8 头 / 4 KV / hd 128 / inter 1408 / ctx 512 |
| 参数量 | 3,213,056 | 26,879,488 |
| 数据 | minimind3-6400 前段（max_blocks 4096） | 全量一遍（81,174 步 @ B8） |

tier1 的数字是实测（7,995 train-only tok/s，332M token ≈ 11.6 h，`logs/dev/02-hardware-perf.md`），
不是估计。换尺寸只动 config：`head_dim` 独立让 10M–70M 都能拼出合法形状。

## 数据流水线

语料是 **minimind 数据集**（中文，`common/prepare.py` 下载/整理）：

1. tokenize 成 token 流（6400-vocab BPE，`common/tokenizer.json`）
2. 每篇文档 `[BOS] … [EOS]` 包裹，拼成长流后按 ctx 切定长块
3. 落 `data/prepared/minimind3-6400/`：`tokens.bin` + `doc_lengths.npy` + `meta.json`
4. 三件文件加载时互相对账；MASK（id 27）不允许出现在正文

vocab_size 必须等于 tokenizer 词表，加载时断言。

## 训练入口

全部格子走 `src/training/main.py`：读 config → build（model / loss / sample 三元组）→ 训练 → 验证 → 采样 → 存盘。

- `--impl trans`：transformers `Trainer` 走 AR pretrain，训练配置在 `TrainingArguments`
- `--impl native`：`src/models/native/` 纯 torch 包在同一条 main 上跑，换芯片不换代码
- SFT / A2D / DLM 是 main 的不同 stage（`--stage`），a2d 用 `--impl dllm|mini` 切对照
- 续训：step / optimizer / 批次位置 / RNG 快照一起恢复（C4 要求逐步等价）

日志每步落 `logs/run/<名字>/metrics.csv`，结束写 `summary.json`（唯一真相）；
吞吐报 train-only 与 wall-clock 两个口径（SPEC §6.4）。

## 采样

采样在训练循环里调用（不是回调魔法）：固定 prompt → `encode(bos=True)` → `generate` →
decode 写进 summary。rep-4 复读率对所有 DLM 结论必报（SPEC §6.4：复读时 loss 照样降）。

## 最快上手

```sh
./scripts/setup_env.sh && ./scripts/install_dllm.sh && ./scripts/setup_symlinks.sh
python -m src.training.main --tier smoketest --impl trans --steps 20
```

GPU 约 1 分钟：loss 从 ~8.9（ln 6400）往下走、checkpoint 与 summary 生成。
正式跑法见 [USAGE.md](USAGE.md)。
