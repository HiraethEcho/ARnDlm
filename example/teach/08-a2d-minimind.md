# Step 8 — `training/a2d_minimind/`

**你要写的**：`training/a2d_minimind/`
**A 版参考**：`example/base/hf/a2d_minimind.py` + `example/base/native/a2d_minimind.py`
**验收**：**C2/C3** —— 与 dllm 的 loss 相对差 `< 1%`（A 版实测 **7.6e-08**）。

---

## 8.1 这个目录有多薄

A 版两个 base 行的 A2D 训练脚本一共 ~60 行，因为难的四块已经拆出去了：

| 部分 | 在哪 | 步骤 |
|---|---|---|
| 加噪 / 加权 loss / 采样器 | `common/a2d.py` | step 5-7 |
| 训练循环 / 存盘 / 指标 | `common/loop.py` 等 | step 0 |
| 双向 attention 的开法 | 每行 5 行 | step 3 |

`training/a2d_minimind/` 要做的是**接线**：模型 → loss → 循环 → 采样 → 存盘。

## 8.2 A2D 转换只有两行有效代码

```python
# 行1（transformers）：4D 全零 mask 就是「不要因果 mask」
def dlm_forward(model):
    def logits_fn(input_ids):
        mask4d = torch.zeros(B, 1, T, T, ...)
        return model(input_ids, attention_mask=mask4d).logits
    return logits_fn

# 行2（原生）：一个开关
logits = model(x, is_causal=False)
```

- **词表不加 MASK**（用预留的 27，见 step 5.4）
- **权重一个都不动**
- 4.57 没有 `layer.self_attn.is_causal` 这个字段。那个是 5.x 的写法，
  本项目继承来的 `~/Projects/dlm/example/hf_dllm.py` 照 5.17 写的，**不能抄**。

## 8.3 loss 的 t 分布要和 dllm 一致

```python
# 我们（notes/03 §4 的纯 U[0,1)，clamp 兜底）
t = torch.rand(B).clamp_min(1e-3)

# dllm（MDLMConfig.time_epsilon = 1e-3）
t = eps + (1 - eps) * torch.rand(B)
```

两者 `E[1/t]` 差 **0.1%**，本身可以忽略。但 **C2 对拍要求两次抽到同一个 t**，
所以对拍时必须用同一个公式 + **同一个全局 seed**，且 RNG 消耗顺序一致：

```
dllm:  rand(B)  → t          rand(B, L) → mask
我们:  rand(B)  → t          rand(B, L) → mask      ← 顺序一致
```

A 版 C2 就是这么做的，并且**额外验证两次 corruption 真的相同**，
而不是只看两个相近的 loss 数字。

## 8.4 A 版实测（两行、两个 checkpoint、同一份数据）

| | hf-minimind-a2d | hf-dllm-a2d |
|---|---|---|
| loss 首 → 末 | 5.8107 → 5.2759 | 5.7631 → 5.2761 |
| eval loss | 4.9723 | 4.9765 |
| train-only tok/s | **46,073** | **28,708** |
| rep-4 | 0.7586 | 0.0000 |

两个 loss 收敛到同一区间，这是 A2D 目标学对了的证据。

**eval loss 比 AR 的 4.6356 高是正常的**（见 step 6.8），两者不能比。

**吞吐差 40% 的来源已定位**（`check_ab.py c2 --bench`）：

```
手写 loss   14.31 ms      dllm loss   22.95 ms      1.60x
```

差的 8.6 ms 是 `MDLMTrainer.compute_loss` 里的 python 开销
（transpose 版 cross_entropy、多余的 mask 张量、meter 更新），
**不是模型差异，也不是 attention kernel 差异**。

## 8.5 rep-4 的差异不要解释

`0.7586` vs `0.0000` —— 两次采样用的是不同解码器、不同 RNG 流，
**rep-4 是分布性质，不是对拍指标**。32 个 token 的样本上它噪声极大。

要用 rep-4 判断「DLM 有没有复读」，得跑长生成（几百 token）+ 多个 prompt，
smoke 阶段只要求「报了」。

## 8.6 坑

| 坑 | 症状 |
|---|---|
| 抄 5.x 的 `is_causal` 后门 | 4.57 里字段不存在，或改了没用 |
| A2D 里加 MASK token | embedding 形状变了，AR 权重灌不进去 |
| C2 不控制 RNG | 两个 loss 差 20%，然后花两天找 bug |
| 把 A2D eval loss 和 AR eval loss 比 | 结论「A2D 更差」，实际上不可比 |
| 把 rep-4 当对拍指标 | 追一个本来就会变的数 |

## 8.7 验收命令

```sh
.venv/bin/python example/check_ab.py c2 --bench
# [c2] PASS rel_diff=7.561e-08 loss_ours=6.306 loss_dllm=6.306 same_corruption=True
#      ours_ms=1.431e+01 dllm_ms=2.295e+01 dllm_overhead_x=1.60e+00

.venv/bin/python example/check_ab.py c5
# [c5] PASS missing=0 unexpected=0 mismatched=0 tensors=24
```