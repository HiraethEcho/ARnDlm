"""minimind — 原生 pytorch 训练 / 后训练, 不依赖 transformers。

对应 PLAN.md 的 T1b。参考 ~/Public/minimind 的训练循环、packing、dataset。

    config.py           Qwen3 的 config (hidden / layers / heads / head_dim ...)
    attention.py        Qwen3Attention: GQA + QK-Norm
    block.py            decoder layer
    mlp.py              SwiGLU
    model.py            完整模型
    init.py             权重初始化
    pretrain.py         训练循环 (fwd/bwd/grad_accum/clip/scheduler/resume)
    sft.py              后训练
    sample.py           AR 采样

**验收标准是「与 transformer/ 的输出逐 token 一致」**, 不是「loss 差不多」。
同 batch 同权重下 logits 要对齐到 1e-5。

三个坑见 notes/04-qwen-architecture.md:
head_dim 不等于 hidden/n_heads · GQA repeat 方向 · tied 权重要 id() 去重。

依赖: torch (原生), ../base
"""
