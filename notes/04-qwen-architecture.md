# Qwen 架构

Qwen2 / Qwen3 都是 LLaMA 系的 pre-norm RMSNorm + RoPE + SwiGLU + GQA。
差异集中在 `head_dim` 和 QK-Norm。

## 实测 config

| | Qwen3-0.6B | Qwen2.5-0.5B | Qwen2-0.5B |
|---|---|---|---|
| `model_type` | `qwen3` | `qwen2` | `qwen2` |
| `hidden_size` | 1024 | 896 | 896 |
| `num_hidden_layers` | 28 | 24 | 24 |
| `num_attention_heads` (Q) | 16 | 14 | 14 |
| `num_key_value_heads` (K,V) | 8 | 2 | 2 |
| `head_dim` | **128 (config 显式)** | **64 (config 省略)** | **64 (config 省略)** |
| Q projection | 1024 → **2048** | 896 → 896 | 896 → 896 |
| K/V projection | 1024 → 1024 | 896 → **128** | 896 → 128 |
| GQA ratio | 2:1 | **7:1** | 7:1 |
| `intermediate_size` (SwiGLU) | 3072 | 4864 | 4864 |
| `rope_theta` | 1e6 | 1e6 | 1e6 |
| `vocab_size` | 151936 | 151936 | 151936 |
| `max_position_embeddings` | 40960 | 32768 | 131072 |
| `tie_word_embeddings` | True | True | True |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 |
| `sliding_window` | None | 32768 (未启用) | 131072 (未启用) |

数据来源: `https://hf-mirror.com/<model>/raw/main/config.json`

---

## 坑 1 — `head_dim` 不一定在 config 里

Qwen2 系列**省略** `head_dim`, 必须自己算:

```python
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
```

Qwen3-0.6B 是特例: `hidden_size=1024` 但 `head_dim=128` 且 16 heads,
所以 Q projection 输出是 **2048 ≠ hidden_size**。

**写死 `q_proj: hidden → hidden` 在 Qwen3 上直接形状错。**

---

## 坑 2 — GQA 的显存计算用错 head 数

K/V heads 少于 Q heads, 多头共享同一组 KV:

```
kv_cache = 2 * n_layers * num_key_value_heads * head_dim * seq_len
```

**用 `num_key_value_heads`, 不是 `num_attention_heads`。**
Qwen2.5-0.5B 的 7:1 意味着 KV 只有 Qwen3-0.6B 的 1/4。

实现时注意方向 —— 要把 KV **扩展**到 Q 的 head 数:

```python
r = n_attention_heads // n_key_value_heads     # 注意分子是 Q
if r > 1:
    k = k.repeat_interleave(r, dim=1)
    v = v.repeat_interleave(r, dim=1)
```

写反成 `n_kv // n_q` 会得到 0, GQA 静默失效, 然后在 SDPA 处报
`tensor a (16) must match tensor b (8)`。

---

## 坑 3 — QK-Norm 只有 Qwen3 有

Qwen3 在 attention 里对 Q 和 K **各加一个 per-head `RMSNorm(head_dim)`**:

```python
q = self.q_norm(self.q_proj(h).view(B, T, nH, hd))    # 注意 norm 在 view 之后
k = self.k_norm(self.k_proj(h).view(B, T, nKV, hd))
```

Qwen2 没有。**这是 Qwen2 ↔ Qwen3 唯一真正的不兼容点。**
(相比之下 `is_causal` 那个改动根本不算架构差异 —— 改一行 flag 就行。)

Q/K 的 `head_dim` 相同 (都是 `head_dim`), 只是 head **个数**不同,
所以 RoPE 的 cos/sin 表可以共用一张。

---

## 其余相同部分

- `RMSNorm` **pre-norm**, `eps = 1e-6`
- `RoPE`, `theta = 1e6`
- `SwiGLU` FFN: 三个矩阵 `gate` / `up` / `down`
- `tie_word_embeddings = True` → `lm_head.weight` 就是 `embed_tokens.weight`
