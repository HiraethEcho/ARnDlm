# Step 7 — `diffusion/sampler.py`

**你要写的**：`diffusion/sampler.py`
**实验版参考**：`example/common/a2d.py::sample_dlm` + `gumbel_argmax`
**验收**：同 seed 同参数同权重，实验版和你的输出**逐 token 相同**。

```sh
.venv/bin/python example/tests/test_a2d.py      # 第 [3] 节
```

---

## 7.1 推理三步（`notes/03` §5）

```
初始化:  [prompt 固定] [M M M M ...]
每一步:  1. 对**全序列** forward
         2. 算每个 MASK 位置的置信度
         3. 揭示 top-k, k = 剩余 MASK 数 / 剩余步数   ← 动态算
```

## 7.2 三个关键点

### 关键点 1：每步揭示数是动态的

```python
n_unmask = max(1, math.ceil(n_active / steps_left))
```

不是固定 top-k。固定值在最后几步会一次揭太多，质量下降。

预算要**按序列算**，不能按整个 batch 算。实验版第一版就是按 batch 汇总的
`remaining`，batch=1 时看不出来，batch>1 就错了。

### 关键点 2：每步全序列 forward，没有 KV cache

双向注意力要用全部已知上下文。当前 block 的预测依赖后面还没揭示的位置，
所以「只算当前 block」是错的。这也是 DLM 推理比 AR 慢的根源
（`notes/03` §5，双向比因果慢 38%）。

### 关键点 3：只有当前 block 能被揭示

```python
conf = confidence[b, start:end].clone()
conf[~active] = -inf          # 非 MASK 位置冻结
```

即使别的 block 里有个位置置信度更高，这一步也不能动它。
**这就是 `block_size` 成为「速度 / 质量」连续旋钮的原因**
（`notes/03` §6：`block_size = max_new_tokens` 最快但依赖步数，
`block_size = 1` 退化成 AR）。

## 7.3 采样用 Gumbel-Max

```python
u = torch.rand(...).clamp_(1e-20, 1.0)
gumbel = -log(-log(u))
perturbed = logits / temperature + gumbel
token = argmax(perturbed)
```

比 softmax+multinomial 在低精度下更稳（`notes/03` §7）。
`temperature=0` 直接走 argmax。

⚠️ **置信度要重新算**：

```python
probs = softmax(logits)                 # 原始 logits
confidence = probs.gather(-1, token)    # 不是 perturbed 的分数
```

用 perturbed 分数当置信度，揭示顺序就由噪声决定，而不是由模型决定 ——
采样还是随机的，但「按置信度揭示」这个机制整个失效了。

## 7.4 top_k 在 argmax 之前

```python
logits = top_k_filter(logits, top_k)    # 截断候选空间
token, conf = gumbel_argmax(logits, temperature)
```

先 top_k 再加 Gumbel 噪声。被截掉的 logit 变成 `-inf`，加上噪声仍是 `-inf`。

## 7.5 block_size 的四个取值

| block_size | 行为 | Tier 0 实测 |
|---|---|---|
| = gen_len (32) | 全序列一次去噪 | 最快 |
| 8 | 分块并行去噪 | 本项目默认 |
| 1 | 逐 token AR（退化） | 最慢，但最接近 AR |

`steps=1` 时整个 block 一次填满；`steps=16 > block_size` 时也会终止
（多出来的步数在 block 填满后直接退出）。这两个边界 实验版测试都覆盖了。

## 7.6 坑

| 坑 | 症状 |
|---|---|
| 置信度用 perturbed 分数 | 揭示顺序变成随机，机制失效但不报错 |
| 揭示预算按 batch 算 | batch=1 正常，batch>1 每条序列揭示数不对 |
| 不冻结非当前 block | `block_size` 失去意义，等于全序列同时揭示 |
| 只 forward 当前 block | 质量下降，看起来像「步数不够」 |
| 截断到 EOS 后当空 = 坏了 | 200 步的 smoke 里 EOS 早现，文本为空很常见 |

## 7.7 对拍方式

```
同权重 + 同 prompt + temperature=0 + 同 block/steps
  →  输出必须逐 token 相同
```

**用 `temperature=0` 对拍**。有温度时两边各自的随机数流不同，
输出必然不同，差异不代表 bug。

实验版测试就是这么做的：`check("temperature=0 is deterministic", ...)`。