# configs/ — 配置层

## 加载顺序

```
base.yaml  →  <实验>.yaml   (后者覆盖前者)
```

`base.yaml` **不包含任何模型结构维度** (hidden_size / n_layers / n_heads /
vocab_size / max_seq_len ...)。那些属于各实验配置。
base 只放「所有实验都一样」的东西: 优化器、调度、频率、路径。

## 实验配置清单

对应 `ROADMAP.md` 的三条线。

| 文件 | 用途 | 状态 |
|---|---|---|
| `base.yaml` | 全局默认 | ✅ |
| `t1_pretrain.yaml` | T1 AR 预训练 (transformers 版) | ⬜ |
| `t1_pretrain_native.yaml` | T1b AR 预训练 (torch 原生版) | ⬜ |
| `t1_sft_lora.yaml` | T1 LoRA / SFT 后训练 | ⬜ |
| `t2_a2d_native.yaml` | T2a A2D 微调 (手动实现) | ⬜ |
| `t2_a2d_dllm.yaml` | T2b A2D 微调 (dllm 包) | ⬜ |
| `t2_a2d_qwen3b.yaml` | T2c 对 Qwen3-0.6B 做 A2D | ⬜ |
| `t3_dllm_pretrain.yaml` | T3 用 dllm 包从零训 DLM | ⬜ |

## 写配置时的规矩

**模型维度写在实验配置里, 不写在 `base.yaml`, 更不硬编码在代码里。**

自训 (T1) 和改造现成模型 (T2c) 的模型形状不同, 但优化器、调度、
日志频率这些是共享的 —— 这就是 base 存在的意义。

**参数量先不钉死。** 冒烟测试 (见 `ROADMAP.md`) 测出实际吞吐后再填
`hidden_size` / `n_layers` / `max_seq_len` / `batch_size`。
这四个数是耦合的, 先拍数字再测是本末倒置。

## 记一件事

> 几个月后你会忘了 `t2_a2d_dllm.yaml` 是干嘛的。

所以每套配置**跑完之后**, 在这个文件里补一行: 跑了什么、结果如何、
有什么坑。**没跑过的配置标 ⬜, 别让它看起来像能用的。**

## mask token 不同

| 线 | mask token id |
|---|---|
| T1 / T2a / T2b (自训, vocab 6400) | **27** = `<|buffer1|>` |
| T2c (Qwen3-0.6B, vocab 151936) | **151669** (267 个空闲 slot 之一) |

T3 用 `dllm` 的默认配置, 别手改。
