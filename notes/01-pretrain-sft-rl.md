# Pretrain / SFT / RL — 三个训练阶段

## 全景

| 阶段 | 全称 | 数据格式 | 学到什么 |
|---|---|---|---|
| Pretrain | 预训练 | `{"text": "..."}` | 语言、语法、知识。**但不会听话** |
| SFT | Supervised Fine-Tuning | `(user, assistant)` 对话对 | 学会「你问, 我答」 |
| RLHF / DPO | 偏好对齐 | (chosen, rejected) | 学会「哪个回答更好」 |

---

## SFT 是什么

**SFT = Supervised Fine-Tuning (监督微调)。**

**核心: SFT 的 loss 函数和 pretrain 完全一样, 都是 next-token prediction。**
变的只有两件事 —— 数据格式, 和 loss mask 的位置。

### 变化 1: 套上 chat template

```python
<|im_start|>user
什么是神经网络?<|im_end|>
<|im_start|>assistant
神经网络是一种模仿大脑处理方式的计算模型...<|im_end|>
```

### 变化 2: 只在 assistant 的回答上算 loss

```python
# input_ids = [<im_start>, user, 什么, 是, 神经网络, ?, <im_end|>, <im_start|>, assistant, 神经网络, 是, ...]
# labels    = [   -100,   -100, -100, -100,  -100,    -100,     -100,       -100,     -100,   神经网络, 是, ...]
#                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 只有这里算 loss

shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))
#                                              ^^^^ ignore_index=-100 的位置自动被忽略
```

**所以 SFT 本质是「换一种数据格式的继续 pretraining」, 没有任何新东西。**

这就是为什么一个 SFT 后的权重可以直接拿去做 A2D 转换 —— 它同时具备
语言能力和对话格式。

---

## Rollout 顺序

```
Pretrain → SFT → Chat (AR baseline) → A2D → Chat (DLM)
                              ↑                  ↑
                        改造前的对照组      同一个模型的两种形态
```

**必须有 AR baseline。** 没有对照组, 就无法判断 DLM 的输出是「变差了」
还是「本来就差」。

---

## 三个必须分清的概念

| 词 | 含义 | 容易混淆 |
|---|---|---|
| **pretrain** | 在原始文本上做 next-token prediction | — |
| **SFT** | 在对话数据上做 next-token prediction, 只算回答部分 | **不是新 loss** |
| **post-train** | 泛指 pretrain 之后的一切 (SFT + RLHF/DPO) | 口语里常被当成 SFT 的同义词 |
| **A2D** | 把 AR 模型改成 DLM (改架构 + 改目标) | **不是 post-train, 是架构改造** |

**A2D 不是 post-training。** 它换的是模型的目标函数和 attention mask,
属于架构级改造。见 `03-minimind-dlm.md`。
