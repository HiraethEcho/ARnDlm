# RESULTS — 实验版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ar-3.2m-hf | Qwen-mini-HF | AR | 3213056 | 200 | 8.4414 -> 4.6970 | 48,138 | 41,863 | 4.6356 | 0.1724 | 524,288 tok, Tier 0 |
| ar-3.2m-native | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 48,777 | 43,571 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| ar-3.2m-hf-mini_a2d | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 46,463 | 41,198 | 4.9723 | 0.8103 | from ar-3.2m-hf, A2D lr 1e-04, block 8, steps 8 |
| ar-3.2m-hf-dllm_a2d | Qwen-mini-HF | dLLM-A2D | 3213056 | 200 | 5.7631 -> 5.2761 | 25,882 | 24,623 | 4.9765 | 0.0000 | from ar-3.2m-hf, dllm MDLMTrainer, block 8, steps 8 |
| dlm-3.2m-hf | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.4999 -> 5.8585 | 47,355 | 41,908 | 5.6427 | 0.7759 | random init, lr 3e-04, block 8, steps 8 |
| ar-3.2m-native-mini_a2d | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 200 | 5.7642 -> 5.2846 | 49,579 | 43,848 | 4.9692 | 0.3966 | from ar-3.2m-native, lr 1e-04, block 8, steps 8 |
| ar-3.2m-native-dllm_a2d | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 200 | 5.7136 -> 5.2850 | 26,305 | 24,558 | 4.9728 | 0.4138 | from ar-3.2m-native, dllm loss on native torch, block 8 |
| native-direct-dlm-example | Qwen-mini-MiniMind | Direct-DLM | 3213056 | 200 | 8.4655 -> 5.8544 | 49,613 | 44,057 | 5.6351 | 0.6897 | random init, lr 3e-04, block 8, steps 8 |
| qwen3-0.6b-qwen3-0.6b-mini_a2d | Qwen3-0.6B | MiniMind-DLM-A2D | 598071296 | 200 | 13.5244 -> 4.9977 | 612 | 587 | 4.8166 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok |
| qwen3-0.6b-qwen3-0.6b-dllm_a2d | Qwen3-0.6B | dLLM-A2D | 598071296 | 200 | 12.9805 -> 5.0049 | 426 | 414 | 4.8013 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok, dllm loss |
| dlm-3.2m-dllm | Qwen-mini-HF | Direct-DLM | 3213056 | 6 | 8.5088 -> 8.1215 | 7,480 | 4,511 | 8.6530 | 0.0000 | random init, dllm MDLMTrainer, lr 3e-04, block 8 |
