# Step 5 — `diffusion/corruption.py` + `scheduler.py`

**你要写的**：`diffusion/corruption.py`（加噪）、`diffusion/scheduler.py`（α(t) 与权重）
**实验版参考**：`example/common/a2d.py`（`sample_t` / `corrupt`）
**验收**：同 seed 同 batch，实验版和你的 mask 位置**逐位相同**。

```sh
.venv/bin/python example/tests/test_a2d.py      # 第 [1] 节
```

---

## 5.1 四步训练（`notes/03` §4）

```
Step 1  t ~ U[0,1)                      每个样本一个标量
Step 2  按比例 t 随机 mask 部分位置
Step 3  双向 forward，预测所有被遮位置
Step 4  只在 MASK 位置算 loss
```

## 5.2 t 是 per-sample 标量，不是 per-token

这是最容易写错的地方，而且写错了**不报错**：

```python
# ✓  每个样本一个 t
t = torch.rand(B)                       # [B]
probs = torch.rand(B, L)
masked = probs < t[:, None]             # 广播成 [B, L]

# ✗  每个位置一个 t —— 每个位置的 mask 概率都是 U[0,1)
t = torch.rand(B, L)
masked = torch.rand(B, L) < t
```

实验版实测（`test_a2d.py` 第 [1] 节）：per-token 版本下每个样本的 mask 率
**均匀分布在 [0,1)**，而 per-sample 版本 mask 率 ≈ t，最大偏差 0.10
（128 个位置的抽样噪声量级）。

## 5.3 加噪就是替换，没有连续噪声

文本没有自然的连续噪声空间，MASK 就是离散噪声（`notes/03` §3）。

```python
x_t = torch.where(masked, torch.full_like(x0, MASK_ID), x0)
```

**没有加高斯噪声、没有乘缩放系数。** 这类扩散的「前向过程」只有一步。

## 5.4 MASK token 的 id

本项目用 **27**（`<|buffer1|>`），复用预留 token，**不扩词表**（`SPEC 3.2`）。

扩词表会改变 embedding 形状，A2D 转换时从 AR checkpoint 灌权重就全部对不上 ——
而且症状是「loss 一直下不去」，不会报错。

`test/01-tokenizer-mask.py` 已经验证自然文本里 56 token 内不出现 27。

## 5.5 scheduler：linear 与 cosine

`dllm` 有两个 α 调度（`dllm/core/schedulers/alpha.py`）：

| 调度 | α(t) | p_mask = 1-α(t) | weight |
|---|---|---|---|
| Linear | `1 - t` | `t` | `-α'(t)/(1-α(t)+ε) = 1/(t+ε)` |
| Cosine | `cos(πt/2)` 平方形式 | `1 - α(t)` | 同式 |

本项目全部用 **linear**，所以 `p_mask = t`、`weight = 1/t`。

写 scheduler 时注意两点：

1. **ε 是硬编码的 1e-6**（`alpha.py:88`）。t→0 时 `1/t` 会爆，ε 兜底。
   你的实现必须用同一个 ε，否则小 t 样本上的 loss 差几个数量级。
2. **不要引入 ELBO / log-SNR / weighting schedule**（`notes/03` §3 明确说了没有）。
   MDLM 系的加权就是 `1/p_mask` 这一项。

## 5.6 坑

| 坑 | 症状 |
|---|---|
| t 写成 per-token | 不报错，模型学到「遮一半」而不是「按 t 遮」 |
| MASK 扩词表 | loss 高且不降 |
| 忘记 `t.clamp_min(1e-3)` | t=0 时 `1/t` 爆，loss 变 NaN |
| 用同一个 generator 连续两次「复现」 | generator 会前进，两次结果不同（要用两个同 seed 的） |
| 在 CUDA 上传 CPU generator | `RuntimeError` |

## 5.7 验收

你的 `corrupt()` 拿同一个 `torch.Generator().manual_seed(7)` 跑两次，
输出必须逐位相同；且 `t=0` 时 mask 数为 0，`t=0.999` 时 mask 率 > 0.99。