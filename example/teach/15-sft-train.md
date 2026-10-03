# Step B2 — SFT 训练：AR objective 与 DLM objective 共用一份代码

**你要写的**：`training/sft_ar_hf/`（AR objective）和 `training/sft_a2d_hf/`（DLM objective）
**实验版参考**：`example/base/hf/train_sft.py`、`example/base/hf/train_dlm_sft.py`、`example/common/losses.py::sft_ar_loss`
**验收**：`ar-3.2m-hf-sft` loss `9.4364 -> 6.4502`、eval `6.2659`；C7 对拍 `max_abs_loss_delta=0.000e+00`。

**先决条件**：Step B1 的 span 必须含 assistant 的 EOS，否则模型学不会停。
这一条曾经漏掉，loss 曲线完全正常，采样却每次跑满 64 token、零个 EOS——
见 `example/ISSUES.md` DONE-12。

**关键设计**：`--base` 切换权重来源，**objective 由脚本本身决定**。
两个 SFT 脚本结构一样，只有 loss 不同、加载 checkpoint 的路径不同。
不要写成两个各写一遍的脚本——它们迟早会漂移，而漂移看起来像模型 bug。

---

## B2.1 为什么 cell 名里 objective 和 base 是分开的

```
ar-3.2m-hf-sft            base = Qwen-mini-HF,      objective = AR
ar-3.2m-hf-mini_a2d-sft   base = Qwen-mini-HF,      objective = DLM  (上游是 A2D 后的权重)
ar-3.2m-native-sft        base = Qwen-mini-MiniMind, objective = AR
```

`base` 回答「用哪份权重」，`objective` 回答「按什么目标训」。
G3 要比较的就是同一对 base 下两种 objective 的差别，所以这两个维度必须正交。

上游 checkpoint 名**由程序带着**（`train_dlm_sft.py` 的 `CELLS` 字典），
不要让脚本从自己的 cell 名反推。反推一旦遇到命名规律之外的格子就崩，
而且崩之前会静默加载错权重。

## B2.2 assistant-only loss：三处都是承重的

```python
ids, labels = batch[0], batch[1]        # [B, L] 两个通道
inputs, targets = ids[:, :-1], labels[:, 1:]
n_target = int((targets != IGNORE).sum())
if n_target == 0:
    return model(inputs).logits.sum() * 0.0, {...}   # 保住计算图
logits = model(inputs).logits
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                       ignore_index=IGNORE)
```

1. **target 来自 labels 通道，不是 ids 通道。** 用 ids 就是在教模型
   复述刚喂给它的 token。批这个张量时两个通道必须一起移位，
   否则 label 的位置整体错一位。
2. **`reduction='mean'` 的分母是「这个 batch 里被监督的 token 数」**。
   先自己数一遍（`n_target`），别指望 `F.cross_entropy` 报给你。
   如果按 sample 求平均再对 batch 求平均，一个监督占比 10% 的块
   会和 90% 的块等权——**这正是「按样本加权」的经典 bug**。
3. **全忽略的 batch 必须特判。** 全 `-100` 时 `F.cross_entropy` 返回 nan，
   一步就把整个 run 毁掉，而 loss 曲线在前几百步**看起来完全正常**。
   乘 `0.0` 保图：梯度是全零，但参数仍在 optimizer 里，形状也对。

## B2.3 对照 `ar_loss`：差在哪

| | `ar_loss`（A2D） | `sft_ar_loss` |
|---|---|---|
| target | `batch[:, 1:]` | `labels[:, 1:]` |
| ignore_index | 无（全监督） | `IGNORE = -100` |
| 分母 | `targets.numel()` | `n_target`（被监督的个数） |
| 全忽略 | 不可能发生 | 必须特判返回 0 |

改动只有这三处。**其余（调度、计时、存盘、optimizer 分组）与 Step A2 完全相同。**
这就是把训练循环放进 `utils/` 的回报。

## B2.4 DLM objective 的 SFT 版

`ar-3.2m-{hf,native}-sft-{mini,dllm}_a2d` 用的 loss 就是 Step C2 的
A2D loss，只是喂进去的 batch 带 assistant mask。

**掩码只在 assistant span 内采样**，prompt 区域永远可见——
否则模型学的是「补全任意前缀」，不是「回答问题」。
这是 SFT 与 pretrain-A2D 唯一的差别，掩码调度本身不变。

## B2.5 坏 checkpoint 检测

SFT 之后最容易出的事是**复读**。所以每个 SFT cell 结束都会算
`rep-4`（连续 4-gram 重复率）并写进 `RESULTS.md`。

单块 32 token / 2 prompts 的 rep-4 离散度极大，**不能用来排名**
（见 `ISSUES.md` OPEN-6）。它的作用是**报警**：
从 0.17 跳到 0.30 以上说明退化了。

## B2.6 坑

| 坑 | 症状 |
|---|---|
| target 用 ids 通道 | loss 降得比正常更快，模型输出复述输入 |
| 全忽略 batch 不特判 | loss 某一步变 nan，之后全废，而曲线此前看着正常 |
| 采样 prompt 用 `f"user\\n{p}\\nassistant\\n"` | 前缀与训练语料差一个 EOS（turn 用 EOS 分隔，不用换行）。长度相同、token 不同、不报错 |
| 只换 builder 不换 sampler | native 模型没有 `.device`，或采样走了另一套 prompt |
| 按 sample 求平均 | 监督占比低的块权重被放大，跨实现对拍永远对不上 |
| labels 和 ids 分别移位 | 位置差一格，loss 只降不报错 |
| DLM 掩码采到 prompt 上 | 评测输出不像在回答问题 |
| 续训只 load 权重不 load 优化器 | 批次顺序从头开始，曲线对不上（见 Step A2） |

## B2.7 验收命令

```sh
# AR objective
.venv/bin/python example/base/hf/train_sft.py              # ar-3.2m-hf-sft
.venv/bin/python example/base/native/train_sft.py          # ar-3.2m-native-sft

# DLM objective，--cell 决定上游是哪个
.venv/bin/python example/base/hf/train_dlm_sft.py --cell ar-3.2m-hf-mini_a2d-sft

# C7：同权重同 batch，两个 base 的 loss 逐位比对
.venv/bin/python example/check_ab.py c7

.venv/bin/python example/tests/test_sft_loss.py
.venv/bin/python example/tests/test_sft_cells.py
```

## B2.6 C7 对拍到底证明了什么

**不要把它读成「两个 cell 的 eval 相同」。** 两个 cell 是两次独立训练：

| cell | loss | eval | peak |
|---|---|---:|---:|
| `ar-3.2m-hf-sft` | 9.4364 -> 6.4502 | 6.2659 | 0.214 GB |
| `ar-3.2m-native-sft` | 9.4190 -> 6.4344 | 6.2475 | 0.214 GB |

eval 差 0.018 是**正常的 run-to-run 差异**（两条链路的 kernel 调度顺序不同，
累积顺序不同，fp32 之外还有 GPU 归约顺序的差异）。

C7 证明的是更强的一件事：**把两份权重拷进同一进程、喂同一批 batch，
逐 batch 的 loss 差为 0.000e+00**。也就是说两个 base 在数学上是同一个函数，
差异只来自训练过程的浮点路径。**这是「G1：两种 AR 写法是不是同一个模型」的答案。**

下一步：Step C1 掩码与调度 → C2 A2D loss → C4 `training/a2d_dllm/`。
