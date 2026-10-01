"""dllm — 直接用 dllm 包训练 DLM。

对应 ROADMAP.md 的 T3。这条线不自己写任何模型代码。

    pretrain.py          用 dllm 的 Trainer 从零训 (或从开源权重续)
    sample.py            dllm 的采样
    evaluate.py          接到 ../utils/metrics.py 的重复率上

与 dllm-a2d/ 的区别:
    dllm-a2d/  拿 AR 权重转成 DLM, 只微调
    dllm/      直接按 DLM 目标训练, 不经过 AR

**核心指标是重复率, 不是 loss。** DLM 失败最典型的表现就是复读而 loss
照样下降。见 notes/03-minimind-dlm.md。

依赖: dllm, ../base
"""
