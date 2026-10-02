# RESULTS — A 版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-ar-example | Qwen-mini-HF | AR | 3213056 | 6 | 8.7901 -> 8.5614 | 9,097 | 4,971 | 8.3379 | 0.0000 | 1,024 tok, Tier 0 |
| minimind-ar-example | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 47,417 | 42,168 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| hf-minimind-a2d-example | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 45,563 | 40,142 | 4.9723 | 0.8103 | from hf-ar-example, A2D lr 1e-04, block 8, steps 8 |
| hf-dllm-a2d-example | Qwen-mini-HF | dLLM-A2D | 3213056 | 200 | 5.7631 -> 5.2761 | 28,838 | 27,113 | 4.9765 | 0.0000 | from hf-ar-example, dllm MDLMTrainer, block 8, steps 8 |
| hf-direct-dlm-example | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.4999 -> 5.8585 | 36,160 | 32,428 | 5.6427 | 0.7759 | random init, lr 3e-04, block 8, steps 8 |
| native-minimind-a2d-example | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 200 | 5.7642 -> 5.2846 | 48,888 | 43,273 | 4.9692 | 0.3966 | from minimind-ar-example, lr 1e-04, block 8, steps 8 |
| native-dllm-a2d-example | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 200 | 5.7136 -> 5.2850 | 27,838 | 25,881 | 4.9728 | 0.4138 | from minimind-ar-example, dllm loss on native torch, block 8 |
| native-direct-dlm-example | Qwen-mini-MiniMind | Direct-DLM | 3213056 | 200 | 8.4655 -> 5.8544 | 48,147 | 42,644 | 5.6351 | 0.6897 | random init, lr 3e-04, block 8, steps 8 |
| qwen3-0.6b-minimind-a2d-lora-example | Qwen3-0.6B | MiniMind-DLM-A2D | 598071296 | 200 | 13.5244 -> 4.9977 | 686 | 654 | 4.8166 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok |
| qwen3-0.6b-dllm-a2d-lora-example | Qwen3-0.6B | dLLM-A2D | 598071296 | 200 | 12.9805 -> 5.0049 | 491 | 474 | 4.8013 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok, dllm loss |
