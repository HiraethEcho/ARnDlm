# Step 6 — `diffusion/loss.py`

**你要写的**：`diffusion/loss.py`
**实验版参考**：`example/common/a2d.py::a2d_loss`
**验收**：**C2** —— 与 `dllm` 的 loss 相对差 `< 1%`；同时与闭式解对拍。

```sh
.venv/bin/python test/04-dllm-loss-check.py     # 闭式对拍（已有）
.venv/bin/python example/tests/test_a2d.py      # 第 [2] 节
```

---

## 6.1 公式

```
L = Σ_i [ 1_{masked(i)} · CE(f(x_t)_i, x_0(i)) / p_mask ] / N_valid
```

```python
nll = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1),
                      reduction="none").view_as(x0)
weight = (1.0 / (t + EPS)).unsqueeze(1)     # [B,1] 广播到 [B,L]
loss = (nll * weight * masked).sum() / n_valid
```

## 6.2 三个必须避开的写法

`test/04` 已经把这三个反例算出来了（实验版实测 Δ 值，vocab=8、logits×3）：

| 写法 | Δ | 为什么还能训 |
|---|---|---|
| **不除以 `p_mask`** | **2.30** | loss 曲线照样下降，只是偏向大 t（遮得多的样本） |
| **分母用被遮数量** | **7.26** | loss 随 mask 比例漂移，曲线形状变了但仍然下降 |
| **`p_mask` 当 per-token** | loss 直接变 9.6 | 权重在样本内混开 |

这三个是最阴险的一类 bug：**数值在合理范围，曲线在下降，模型什么都没学到。**
`notes/03` 的作者自己也报了负面结果（loss 正常，输出大量重复）。

## 6.3 分母是「有效 token 数」，不是「被遮数量」

```python
loss = per_token.sum() / valid.sum().clamp_min(1)
```

`valid = labels != -100`。纯文本预训练时每块都是满的，`valid` 就是全部位置。
但 SFT 场景下 prompt 区 label 是 -100，**分母就不能包含 prompt**。

实验版 `a2d_loss` 保留了 `valid` 参数就是为了这个：SFT 时传
`valid = (labels != -100)`。

## 6.4 ε = 1e-6，不是 0

`dllm/core/schedulers/alpha.py:88` 硬编码 1e-6。你的实现必须一样：
t 是 `U[0,1)`，抽到 t < 1e-6 的概率约 1e-6，**不 clamp 就会偶发 NaN**，
而且一崩就是一整个 run，没法复现。

## 6.5 比 loss 必须 fp32

bf16 下两套实现的舍入差比要测的差异还大。C2/C3 一律：
固定批次 + 固定 seed + **fp32**（`Loop.eval_loss` 强制 fp32 就是这个原因）。

## 6.6 坑

| 坑 | 症状 |
|---|---|
| 分母用 masked 数 | 曲线下降，分布学错 |
| 不做 `clamp_min(1)` | 全 padding 时除零 |
| `weight` 形状 `[B,L]` | 权重在样本内混开 |
| 用 `view` 而不是 `reshape` | logits 非连续，`view` 报错 |
| CE 在 bf16 下算 | 与 dllm 对拍永远对不上 |

## 6.7 C2 的做法

```
同 AR checkpoint  →  同 batch（同 fixed_batches）  →  同 seed
     ↓                                                  ↓
hand-written a2d_loss                          dllm MDLMTrainer
     ↓                                                  ↓
       loss_a ───────────── 相对差 < 1% ─────────── loss_b
```

注意两边 loss 的量纲：**A2D loss 不能和 AR loss 比**（cell 3 的 eval 4.97
比 cell 1 的 4.64 高，是正常现象，见 step 6.8）。

## 6.8 一个容易误读的数

cell 1 的 eval CE 是 **4.6356**，cell 3 的 A2D eval 是 **4.9723**，
A2D 看起来「更差」。

**这是两个不同的量。** A2D 目标除以 t，小 t 样本（只遮几个位置）的权重
是 1/t，最大到 1000，所以整体数值被小 t 样本抬高。

**能比的只有同一目标下的两个实现**（C2/C3）。跨目标比 loss 没有意义。