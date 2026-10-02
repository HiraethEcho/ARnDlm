# Step 4 — `training/ar_native/`：同一套循环，原生模型

**你要写的**：`training/ar_native/`
**实验版参考**：`example/base/native/train_ar.py`
**验收**：与行1 **同 seed 同数据**，逐步 loss 一致（fp32，`< 1e-6`）。

---

## 4.1 为什么行2 也要写训练脚本

行1 和行2 的差别**只有模型**。数据、批次顺序、调度、优化器、计时、存盘
全部共用（来自 step 0 的 `utils/`）。

如果行2 另写一套循环，那么 C2/C3 的 loss 差就有两个来源：模型 or 循环。
实验版两行共用 `common/loop.py`，就是为了让差值**只能**来自模型。

## 4.2 入口签名要对齐

```
example/base/native/train_ar.py --steps 200 --batch-size 8 --max-blocks 4096
```

`--max-blocks` 是为了 smoke 快速验流程：4096 块 × 128 = **524,288 token**，
200 步 × 8 × 128 = 204,800 token，只吃语料前 ~20%。

## 4.3 实验版实测基准（你可以对着对）

| | 行1 hf | 行2 native |
|---|---|---|
| loss 首 → 末 | 8.4414 → 4.6970 | 8.4165 → 4.6932 |
| eval loss | 4.6356 | 4.6283 |
| train-only tok/s | 46,942 | 47,703 |
| wall tok/s | 40,633 | 42,502 |
| peak GB | 0.214 | 0.214 |
| rep-4 | 0.1724 | 0.3276 |

两条 loss 首值不同（8.4414 vs 8.4165）**是对的**：两边的随机初始化不同
（见 step 3.5），初始 loss 本来就不该相等。200 步后收敛到同一区间，
C1 在训完的权重上是 **6.91e-06**。

**吞吐逐次运行会变 ±5%**（共享核显，见 `ISSUES.md` OPEN-5 的同类现象），
表里是一次具体运行的值；loss、eval loss、rep-4 在同配置下是**逐位可复现的**。

rep-4 不同（0.17 vs 0.33）也是对的：采样用了各自模型的随机数。
**rep-4 是分布性质，不是对拍指标。**

## 4.4 吞吐：手写不等于慢

行2 反而快 **1.4%**。Tier 0 只有 3.2M 参数、ctx 128，
transformers 的 python 层开销和融合算子省下来的差不多。
正式规模（50M+、ctx 512）时结论可能反过来 —— 到时候重测，别沿用。

**这也是为什么小模型冒烟不能推断正式模型的性能。**

## 4.5 采样：没有 KV cache 的写法

行2 不用 `generate`，每步重跑整个前缀：

```python
for _ in range(max_new_tokens):
    window = prefix[:, -max_context:]
    logits = model(window).logits[0, -1].float()
    values, indices = torch.topk(logits, top_k)
    probs = torch.softmax(values / temperature, dim=-1)
    nxt = indices[torch.multinomial(probs, 1)]
```

- `window = prefix[:, -max_context:]`：**必须裁**。不裁的话序列会超过
  `max_position_embeddings`，RoPE 位置 id 越界。
- `logits[0, -1]` 只取最后一个位置 —— 前面位置的 logits 白算了，
  但在 Tier 0 上重跑整个前缀比维护 cache 简单得多，也快。
- `temperature=0` 走 `argmax`，不要用 `multinomial` 配 `temperature=0`
  （会除零或 NaN）。

正式规模再上 KV cache；DLM 行**永远不需要**（`notes/03` §5 关键点 2）。

## 4.6 坑

| 坑 | 症状 |
|---|---|
| 采样不裁窗口 | 超过 128 之后 RoPE 位置越界，输出变乱码 |
| 忘了 `.float()` 取 logits | top-k 在 bf16 上选错 token，分布悄悄变窄 |
| 采样后没恢复 `train()` | 后续 step 在 eval 模式下跑 |
| 用 rep-4 做 实验版/实践版 对拍 | 它是分布性质，两次采样本来就不一样 |
| 拿 Tier 0 的吞吐推断正式规模 | 见 4.4 |

## 4.7 验收命令

```sh
.venv/bin/python example/base/native/train_ar.py --compare-hf
```

末尾会打印两行：

```
verify  OK  max|dlogit| 0.00e+00
C1      OK  hf-ar-example vs minimind-ar-example  max|dlogit| 6.91e-06  (tol 1e-5)
```