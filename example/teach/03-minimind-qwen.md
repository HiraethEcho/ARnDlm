# Step 3 — `models/minimind_qwen/`：纯 torch 的 Qwen-mini

**你要写的**：`models/minimind_qwen/`（RoPE · RMSNorm · SwiGLU · GQA · QK-Norm · tied embedding）
**实验版参考**：`example/base/native/model.py`
**验收**：**C1** —— 灌入行1 的权重后，同 batch logits `max|Δ| ≤ 1e-5`。

```sh
.venv/bin/python example/check_ab.py c1 --layer-report
```

这是整个项目最核心的一步。C1 会把 RoPE 位置、GQA 头复制、QK-Norm 位置、
tied embedding 这四类 bug 全部逼出来。

---

## 3.1 名字必须和 transformers 一模一样

不是风格问题 —— 是 **C1 能成立的前提**。行1 的 checkpoint 要能直接
`load_state_dict` 进你的模型，中间没有重命名表。

| transformers | 你必须这么命名 |
|---|---|
| `model.embed_tokens.weight` | 同 |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | 同 |
| `model.layers.{i}.self_attn.{q_norm,k_norm}.weight` | 同 |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | 同 |
| `model.layers.{i}.input_layernorm.weight` / `post_attention_layernorm.weight` | 同 |
| `model.norm.weight` / `lm_head.weight` | 同 |

自检：

```python
set(hf.state_dict()) == set(your_model.state_dict())     # 必须 True
```

## 3.2 attention 里的顺序

```
project → q_norm/k_norm (在 head_dim 上) → RoPE → GQA 复制 → sdpa → o_proj
```

四个错一个就废：

| 错 | 后果 |
|---|---|
| RoPE 在 QK-Norm 之前 | C1 差 1e-1 起步 |
| QK-Norm 作用在 hidden_size 而不是 head_dim | 形状对不上，直接报错（这个反而好） |
| GQA 用 `repeat` 而不是 `repeat_interleave` | **不报错**，模型照常训练，只是每个 query head 配错了 kv head |
| o_proj 之前没 reshape 回 [B,T,H·D] | 形状错 |

GQA 那一行值得展开：`q` 有 2 个 head，`k`/`v` 只有 1 个。
正确做法是 `kv` head 0 服务 q head 0 和 1：

```python
x[:, :, None].expand(B, kv, n_rep, T, D).reshape(B, kv * n_rep, T, D)   # = repeat_interleave
torch.repeat_interleave(x, n_rep, dim=1)                               # 等价
```

**用 `repeat`（`x.repeat(1, n_rep, 1, 1)`）会把两个 q head 都接到同一个 kv 上吗？**
不会 —— `repeat` 是 `[kv0, kv0, kv1, kv1]`，`repeat_interleave` 是 `[kv0, kv0, kv1, kv1]`…
在 `kv=1` 时两者一样，**在 `kv=2` 时完全不同**。
Tier 0 的 `kv_heads=1` 恰好掩盖了这个 bug，所以对拍要额外构造一个 `kv=2` 的配置。

## 3.3 RMSNorm 的精度

`modeling_qwen3.py:50-54`：

```python
input_dtype = x.dtype
x = x.to(torch.float32)
var = x.pow(2).mean(-1, keepdim=True)
x = x * torch.rsqrt(var + eps)          # eps = 1e-6
return self.weight * x.to(input_dtype)  # 权重乘法在原 dtype
```

方差在 **fp32** 里算，不是输入 dtype。bf16 下两者能差 1e-3。

## 3.4 RoPE 的 inv_freq 要逐位对齐

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
```

`arange` 的 dtype 要一样（int64 再转 float），否则除法结果在小数第 8 位上不同，
C1 就过不了 1e-5。

`emb = cat((freqs, freqs), dim=-1)`，然后 `cos`/`sin` 各自 `[None, T, head_dim]`，
apply 时 `unsqueeze(1)` 变成 `[1, 1, T, head_dim]` 去广播 `[B, H, T, D]`。

## 3.5 ⚠ 相同 seed ≠ 相同初始权重

实验版实测：两个模型都传 `seed=42`，`max|Δ| = 2.7`。

原因：`transformers` 在 `post_init()` 里初始化，本地实现在 `apply()` 里，
**消耗 RNG 的顺序不同**。

```python
hf = build_hf(cfg, seed=42)
nat = NativeQwen3(cfg)          # 不要指望它和 hf 一样
nat.load_state_dict(hf.state_dict())   # 必须显式灌权重
```

这条对 A2D 也成立：转换永远靠**灌权重**，不靠「两边初始化方式一样」。

## 3.6 双向开关是同一个模型

A2D 第一步就是 `is_causal=True → False`（`notes/03-minimind-dlm.md` §8）。
权重一个都不动：

```python
F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and T > 1)
```

`T > 1` 的判断要留着：长度为 1 时 `is_causal=True` 在部分后端会报形状错。

验证开关真的生效（实验版测试就是这么做的）：

| 检查 | 期望 |
|---|---|
| 改最后一个 token，看前面的 logits | causal：**完全不变**（0.0） |
| 同上，`is_causal=False` | 变化 > 1e-3 |

## 3.7 参数解析式

用 `example/tests/test_hf_ar.py::analytic_params` 的公式，
Tier 0 = **3,213,056**（tied 只算一次，`state_dict` 25 项 / 实存 24 个张量）。

## 3.8 排错流程

C1 失败时**不要**从头读代码，跑分层报告：

```sh
.venv/bin/python example/check_ab.py c1 --random --layer-report
```

```
embed_tokens     0.000e+00
rope cos/sin     0.000e+00
layer 0          5.960e-07
layer 1          8.345e-07
lm_head          2.265e-06
```

第一个明显偏离的层就是 bug 所在。实验版实测全绿时的量级是 **1e-7 ~ 1e-6**
（fp32 累加顺序造成的，不是语义差异）。