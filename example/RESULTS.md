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
| native-minimind-a2d-example | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 4 | 5.7713 -> 5.7713 | 6,815 | 3,436 | 5.9271 | 0.0476 | from minimind-ar-example, lr 1e-04, block 8, steps 8 |
| native-dllm-a2d-example | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 4 | 5.7586 -> 5.7586 | 4,678 | 2,849 | 5.8889 | 0.0476 | from minimind-ar-example, dllm loss on native torch, block 8 |
| native-direct-dlm-example | Qwen-mini-MiniMind | Direct-DLM | 3213056 | 4 | 8.2819 -> 8.2819 | 6,915 | 3,607 | 8.5616 | 0.0000 | random init, lr 3e-04, block 8, steps 8 |
| qwen3-0.6b-minimind-a2d-lora-example | Qwen3-0.6B | MiniMind-DLM-A2D | 598071296 | 200 | 12.1802 -> 10.0348 | 714 | 666 | 10.4773 | 0.9138 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok |
| qwen3-0.6b-dllm-a2d-lora-example | Qwen3-0.6B | dLLM-A2D | 598071296 | 200 | 12.1953 -> 4.5071 | 494 | 477 | 4.0900 | 0.6897 | LoRA r8 q/k/v/o, 0.3835% trainable, 6,528 tok, dllm loss |
