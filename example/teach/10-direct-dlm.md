# Step 10 — `training/direct_dlm/`

**你要写的**：`training/direct_dlm/`
**实验版参考**：`example/base/hf/direct_dlm.py` + `example/base/hf/train_direct_dlm.py`
**验收**：能训练、能采样、能存盘；eval loss 与 A2D 格的差**记录**下来（不设阈值）。

---

## 10.1 这个格子只有一个变量

矩阵第三列 Direct-DLM 的意思不是「新方法」，而是**同一个 MDLM 目标，
但不从 AR 权重出发**。

所以正确写法是**复用** `training/a2d_minimind/` 的 loss 和 sampler，
只换初始化：

```python
# 不要复制 loss —— 复制之后两格会各自漂移，
# 而这一对格存在的全部意义就是「只有一个变量不同」
from base.hf.direct_dlm import make_direct_loss     # 实验版：import 而非 copy
```

实践版对应：`training/direct_dlm/` import `training/a2d_minimind/` 的模块。

## 10.2 学习率要换

| 格 | 起点 | lr |
|---|---|---|
| A2D | 训好的 AR 权重 | **1e-4** |
| Direct-DLM | 随机初始化 | **3e-4** |

用 A2D 的 1e-4 训 200 步，随机初始化的模型基本没动 —— 曲线看起来「在降」，
其实只是从 log(V) 掉了一点点。

**这是 Direct-DLM 格子最容易写成「跑通了但什么也没发生」的地方。**

## 10.3 记录初始指纹

```python
fingerprint = torch.cat([p.detach().reshape(-1)[:8] for p in model.parameters()])
```

存进 `config.json`。两个都声称 `seed=42` 的 Direct-DLM 运行，
如果指纹不同，那它们根本不是同一个实验 —— 而这件事在 loss 曲线上完全看不出来。

实验版记录的值（Tier 0, seed 42）就存在
`checkpoints/example/dlm-3.2m-hf/config.json`。

## 10.4 实验版实测：AR 权重值多少

| | loss 首 → 末 | eval | rep-4 |
|---|---|---|---|
| `hf-minimind-a2d` | 5.8107 → 5.2759 | **4.9723** | 0.7586 |
| `hf-direct-dlm` | 8.4999 → 5.8585 | **5.6427** | 0.7931 |

AR 权重在这个规模上值 **0.67 nat**，而且 from-scratch 的模型复读明显更严重。

**这不矛盾**（`notes/03` §11）：A2D 继承的是 AR 的**机制**，从零训要自己学机制。
在 200 步 / 20 万 token 的规模上，差距只会更大。

## 10.5 不要从这个数推正式规模

5.64 vs 4.97 是 **205k token** 的结果。要回答「A2D 到底值不值」，
需要在正式规模上各训到收敛再比 —— 见 `teach/13`。

smoke 阶段这一格只回答：**从零训的 pipeline 能不能跑通**。

## 10.6 坑

| 坑 | 症状 |
|---|---|
| 复制 A2D 的 loss | 两格漂移，对比失效 |
| 用 1e-4 | 200 步等于没训，loss 只从 8.7 掉到 8.4 |
| 忘了双向 attention | 这一列的模型变成 AR，方法名骗人 |
| 不记初始指纹 | 两次「同 seed」的运行不可比，且看不出来 |
| 拿 0.67 nat 外推 | 规模依赖，这个数只在 20 万 token 上成立 |

## 10.7 验收命令

```sh
.venv/bin/python example/tests/test_direct_dlm.py
.venv/bin/python example/base/hf/train_direct_dlm.py
```