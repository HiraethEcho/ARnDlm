# ROADMAP

三条线。**每条线开工前先写模块 docstring 桩, 写完再写实现。**

约束 (全部来自 AGENTS.md, 不重复解释): 单 venv / transformers 4.57.0 /
Qwen3 架构 / vocab 6400 / 同一份 minimind 数据 / 目标 ~50M 但**待冒烟测试后定**。

---

## T1 · 训 AR 基准模型

**为什么先做这个**: 没有 AR baseline 就无法判断 DLM 是「变差了」还是「本来就差」。

### T1a · 用 `transformers` 包训  →  `src/transformer/`

| | |
|---|---|
| 产出 | 一个能续训的 AR checkpoint |
| 后训练 | `peft` LoRA / SFT, 目标是能对话 |
| 验收 | 生成的文本通顺; loss 曲线下降; 两种吞吐都报出来 |

关键决策点: **用 `Trainer` 还是自己写 loop。** `Trainer` 快但学不到东西;
自己写 loop 要处理 grad_accum / clip / scheduler / resume。

### T1b · 原生 pytorch 实现  →  `src/minimind/`

| | |
|---|---|
| 产出 | 与 T1a 同一架构、同一超参的 torch 原生实现 |
| 参考 | `~/Public/minimind` —— 训练循环、packing、dataset 都看它怎么写 |
| 验收 | **与 T1a 输出逐 token 一致** (同一 batch 同一权重, logits 对齐到 1e-5) |

架构 = **Qwen3**, 但尺寸自己定。见 `notes/04-qwen-architecture.md` 的三个坑:
`head_dim` 不等于 `hidden_size / n_heads`、GQA repeat 方向、QK-Norm。
vocab 6400 意味着 tied embedding, **传给 optimizer 的参数必须 `id()` 去重**。

参数量候选 (tied embedding, vocab 6400):

| | hidden | layers | Q/KV | head_dim | inter | 参数量 |
|---|---|---|---|---|---|---|
| 迷你 | 512 | 8 | 8/4 | 64 | 1536 | 28.5M |
| 中 | 576 | 12 | 8/4 | 72 | 1728 | 51.5M |
| 偏大 | 640 | 10 | 8/4 | 80 | 1920 | 53.3M |

**先跑冒烟测试定 batch / seq_len / 参数量, 再填 `configs/`。** 别反过来。

---

## T2 · 对基准模型做 A2D

A2D = 把 AR 模型改成双向注意力 + 扩散训练目标。改动集中在三处:
因果 mask → 双向、加噪 `corrupt()`、loss 换成加权 CE。

### T2a · minimind-dlm 手动实现  →  `src/a2d/`

**「minimind-dlm」在本项目里就是这个意思**: 按
`notes/03-minimind-dlm.md`（源自 https://jingyaogong.github.io/space/dllm_intro）
的思路手写 A2D。

| | |
|---|---|
| 参考 | `notes/03-minimind-dlm.md` + minimind discussion #618 |
| 产出 | 自己的 `convert.py` / `corruption.py` / `loss.py` / `sample.py` |
| 验收 | 与 T2b 在同权重同 seed 下 loss 一致 (差 < 1%) |

**除以 `p_mask` 的 importance reweighting 不能省。** 见 `notes/03-minimind-dlm.md`。

### T2b · 用 `dllm` 包  →  `src/dllm-a2d/`

| | |
|---|---|
| 入口 | `dllm/pipelines/a2d/convert.py` + `dllm/core/trainers/` |
| 硬约束 | **`dllm` 的 A2D 只实现了 `llama` / `qwen2` / `qwen3` 三个架构**, 见 `dllm/pipelines/a2d/models/` |
| 验收 | 与 T2a 在同一 checkpoint 上跑出同一个 loss 值 |

想知道 A2D 到底改了什么: 直接 diff
`dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` 和 transformers 4.57 的
`modeling_qwen3.py` —— 前者是后者拷过来改的。

### T2c · 对现成的开源模型做 A2D

| | |
|---|---|
| 目标 | `Qwen3-0.6B` (已下载在 `/mnt/extra/hf/hub/models--Qwen--Qwen3-0.6B`) |
| 代码位置 | 复用 `src/a2d/` 或 `src/dllm-a2d/`, 不开新目录 —— 只是权重来源不同 |
| 为什么要做 | 自训的 ~50M 和开源的 600M 差一个数量级。A2D 方法能不能 scale, 只有这一条能回答 |
| 注意 | Qwen3-0.6B 的 mask token 用 **`id 151669`** (267 个空闲 slot 之一, `convert_ids_to_tokens` 返回 `None` 可验证) |

---

## T3 · 直接用 `dllm` 包训 DLM  →  `src/dllm/`

| | |
|---|---|
| 产出 | 从零 (或从开源权重) 训的 DLM, 不自己写模型代码 |
| 与 T2b 的区别 | T2b 拿 AR 权重转成 DLM 只微调; T3 直接按 DLM 目标训, 不经过 AR |
| 验收 | **报 repetition rate, 不只报 loss** |

---

## 数据 / tokenizer (三条线共用)

```
data -> /mnt/extra/data
  prepared/minimind3-6400/
    tokens.bin         665 MB, uint16, 332,495,324 token
    doc_lengths.npy    int32, (1270238,) mean 261.8 max 11802
    meta.json          vocab=6400, n_docs=1270238
  raw/
    pretrain_t2t_mini.jsonl   1.24 GB
    sft_t2t_mini.jsonl        1.74 GB
```

```
tokenizer   src/base/tokenizer.json   BPE, vocab 6400, 6108 merges
            PAD 0 <|endoftext|>  BOS 1 <|im_start|>
            EOS 2 <|im_end|>     MASK 27 <|buffer1|>
```

**332M tokens 跑满是 19-25 小时。** 冒烟测试定完 batch/seq_len 再决定缩多少数据。

---

## 状态

| | 状态 |
|---|---|
| scaffold | ⬜ 未开始 |
| T1a | ⬜ |
| T1b | ⬜ |
| T2a | ⬜ |
| T2b | ⬜ |
| T2c | ⬜ |
| T3 | ⬜ |
