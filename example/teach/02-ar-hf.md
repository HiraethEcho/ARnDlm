# Step 2 — `training/ar_hf/`：自回归训练循环

**你要写的**：`training/ar_hf/`
**实验版参考**：`example/base/hf/model.py::ar_loss` + `example/common/loop.py`
**验收**：同 seed 同数据，你的 loss 曲线和 实验版逐步一致（fp32，`< 1e-6`）。

---

## 2.1 这一步的接口

```
training/ar_hf/
  train.py       # 入口，参数与 example/base/hf/train_ar.py 对齐
  loss.py        # ar_loss(model, batch) -> (loss, extra)
```

训练循环本体（optimizer / 调度 / 计时 / 存盘）属于 step 0 的 `utils/`。
**如果 `utils/` 还没写完，先用 实验版的 `common.loop` 跑通 loss 这一步。**

## 2.2 loss 就三行，但含义要清楚

```python
inputs, targets = batch[:, :-1], batch[:, 1:]
logits = model(inputs).logits
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
```

- **移位**：`[..., :-1]` 当输入，`[..., 1:]` 当标签。每块 `ctx-1` 个预测。
- **reshape 而不是 view**：`logits` 是非连续张量，`view` 会报错，
  `reshape` 在必要时自动拷贝。`F.cross_entropy` 需要二维输入。
- **不除以有效长度**：这里每块都是满的（packing 丢掉了尾巴），不存在 PAD。
  如果你改成 padding，**必须**把 PAD 位置的 label 置 -100，否则模型在学着预测 PAD。

## 2.3 必须同时报的两个吞吐

`notes/05-measurement.md` Bug 1 的形状：

```python
t0 = time.perf_counter()
loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()                  # 不 synchronize 你量的是排队时间
step_seconds = time.perf_counter() - t0
meter.add(n_tokens, step_seconds)         # train-only 分母只放这一段
```

**实验版自己也踩了**：最初 `summary` 在 eval / sample 之前组装，
wall-clock 于是和 train-only 三个数字位完全一样 —— 就是 Bug 1 描述的形状。
现在 eval 和采样在计时区间内，summary 最后组装。

## 2.4 计时区间里不要放别的东西

| 内容 | 放不放进 train-only |
|---|---|
| forward / backward / optimizer.step | ✅ 放 |
| validation loss | ❌ |
| 采样生成 | ❌ |
| 存 checkpoint | ❌ |

`common/loop.py::_train_step` 用前后两个 `cuda.synchronize()` 把这段夹住。
少一个 → 计时混入上一个 step 的排队时间，ms/step 会随队列深度漂移。

## 2.5 LR 调度

```
warmup:  lr · (step+1) / warmup          (线性)
之后:    cosine 从 lr 降到 min_lr_ratio · lr
```

**「跑一半」不能靠改 `steps` 实现** —— 改了调度就变了，续训和一次跑完的曲线对不上。
要提前停就加 `max_steps`（`Loop.fit(max_steps=...)`），实验版就是这么做的，
也是 C4 能成立的前提。

## 2.6 optimizer 参数组

```python
decay     = [p for p in ... if p.dim() >= 2]   # 矩阵
no_decay  = [p for p in ... if p.dim() <  2]   # RMSNorm / bias
```

按 `id()` 去重再分组（`SPEC 3.1`）。tied embedding 在 `parameters()` 里已经去重，
但如果你的实现用 `named_parameters(remove_duplicate=False)` 或直接遍历 `state_dict`，
就会把同一张量交给 AdamW 两次 —— 等效学习率翻倍。

## 2.7 坑

| 坑 | 症状 |
|---|---|
| 只报一个吞吐 | 无法区分「训练慢」和「验证/采样贵」 |
| 用 epoch 而不是步数封顶 | 332M token 会跑一整天，smoke 根本跑不完 |
| AMP 下把 bf16 权重直接给 optimizer | 200 步看不出问题，10 万步后更新量被吃掉 |
| 恢复时只 load 权重 | 批次顺序从头开始，曲线对不上（实验版真机测过，见 `0d3e5ce`） |
| AMP 下比两套实现的 loss | bf16 舍入差比要测的差异还大，对拍要 fp32 |

## 2.8 验收命令

```sh
# 实验版跑一遍拿基准数
.venv/bin/python example/base/hf/train_ar.py

# 逐步 loss 一致性（fp32）
#   同 seed -> 同初值 -> 同批次 -> loss[0..199] 逐个比
```

实验版 Tier 0 实测：loss `8.4414 -> 4.6970`，eval `4.6356`，
train-only `46,942` / wall `40,633` tok/s，峰值 `0.214 GB`，rep-4 `0.1724`。
完整表见 `RESULTS.md`（吞吐逐次会变，loss 类数字逐位可复现）。