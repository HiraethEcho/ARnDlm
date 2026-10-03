"""hf_qwen — 调 transformers 包构造 Qwen3, 不自己写模型。

对应教学计划 Step A1 (`example/teach/01-hf-qwen.md`)。**不依赖 minimind。**

    config.py            从 MiniMind 的 6400 词表 tokenize + Qwen3Config
    model.py             build_model(cfg, seed) -> Qwen3ForCausalLM
    load.py              从 HF 或本地 safetensors 载权重

这里**什么都不写**。第一版直接用 `transformers.AutoModelForCausalLM` +
`Qwen3Config`, 因为第一版的目的不是「手写模型」, 是「先有一个 AR baseline」。
没有 baseline, A2D 出来的模型变差了你也不知道是 DLM 的问题还是它本来就差。

**为什么模型结构用包、训练循环不用。** 结构是抄一遍就错的东西
(RoPE 缩放、GQA repeat 方向、head_dim != hidden/n_heads、QK-Norm 在哪一步),
抄对了没人夸, 抄错了 loss 曲线照样降。训练循环相反——它决定你怎么得出结论,
而每个结论都要你自己算一遍才作数。见 `training/ar_hf/`。

**验收**: 与 `models/minimind_qwen/` 在**同权重同 batch** 下逐 token logits 对齐到
1e-5。这是 C1, 不是「loss 差不多」。

依赖: transformers 4.57.0, ../utils
坑: tie_word_embeddings=True 时 embed_tokens.weight 和 lm_head.weight
是同一个对象, 传给 optimizer 前必须按 id() 去重, 否则等效 lr 翻倍。"""
