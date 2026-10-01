"""transformer — 调用 transformers 包做训练 / 后训练。

对应 PLAN.md 的 T1a。产出是 AR baseline。

    pretrain.py         AutoModelForCausalLM + 自己的 loop 或 Trainer
    sft.py              chat template + 只在 assistant 区算 loss
    lora.py             peft LoRA 后训练 (target 只挂 attention 投影)
    sample.py           generate()

用 transformers 而不是手写, 是为了先把「AR baseline 存在」这件事解决掉。
有了 baseline 才能判断 DLM 是「变差了」还是「本来就差」。

坑: tie_word_embeddings=True 时 embed_tokens.weight 和 lm_head.weight 是
同一个对象, 传给 optimizer 前必须按 id() 去重, 否则 lr 翻倍。

依赖: transformers 4.57.0, peft, accelerate, ../../base
"""
