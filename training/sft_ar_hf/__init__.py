"""sft_ar_hf — 自回归 SFT: 只在 assistant 区算 loss。

对应教学计划 Step B2 (`example/teach/15-sft-train.md`)。
上游是 `ar_hf/` 训出来的 AR checkpoint, 下游是 `a2d_dllm/`。

    data.py              读 data/sft/ 落盘的 (tokens, assistant_spans)
    loss.py              sft_ar_loss(model, batch) -> (loss, extra)
    train.py             入口

**这一步是 G3 的必需对照组。** 没有它就无法判断 DLM 的输出是「变差了」
还是「本来就差」——参见 AGENTS.md。

loss 和 `ar_hf/` 的差只有三处, 其余(调度 / 计时 / 存盘 / optimizer 分组)完全相同:

| | ar_hf | sft_ar_hf |
|---|---|---|
| target | batch[:, 1:] | labels[:, 1:] |
| ignore_index | 无(全监督) | -100 |
| 分母 | targets.numel() | 被监督的 token 数 |

**三处都是承重的**:
- target 来自 labels 通道, 不是 ids 通道 —— 用 ids 等于教模型复述刚喂进去的 token。
- labels 和 ids **一起**移位, 分别移位会差一格。
- reduction='mean' 的分母是「这个 batch 里被监督的 token 数」。按 sample
  求平均再对 batch 求平均, 会让监督占比 10% 的块和 90% 的块等权。
- **全忽略的 batch 必须特判返回 0**。全 -100 时 F.cross_entropy 返回 nan,
  一步毁掉整个 run, 而 loss 曲线在坏掉之前看起来完全正常。

依赖: transformers 4.57.0, ../../models/hf_qwen, ../../utils, ../data"""
