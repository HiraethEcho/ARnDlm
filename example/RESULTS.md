# RESULTS — 实验版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ar-3.2m-hf | Qwen-mini-HF | AR | 3213056 | 200 | 8.4414 -> 4.6970 | 46,173 | 40,286 | 4.6356 | 0.1724 | 524,288 tok, Tier 0 |
| ar-3.2m-native | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 48,393 | 43,224 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| ar-3.2m-hf-mini_a2d | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 46,852 | 41,498 | 4.9723 | 0.8103 | from ar-3.2m-hf-example, A2D lr 1e-04, block 8, steps 8 |
| ar-3.2m-hf-dllm_a2d | Qwen-mini-HF | dLLM-A2D | 3213056 | 200 | 5.7631 -> 5.2761 | 28,519 | 26,892 | 4.9765 | 0.0000 | from ar-3.2m-hf-example, dllm MDLMTrainer, block 8, steps 8 |
| dlm-3.2m-hf | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.4999 -> 5.8585 | 44,949 | 40,068 | 5.6427 | 0.7759 | random init, lr 3e-04, block 8, steps 8 |
| ar-3.2m-native-mini_a2d | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 200 | 5.7642 -> 5.2846 | 49,155 | 43,585 | 4.9692 | 0.3966 | from ar-3.2m-native-example, lr 1e-04, block 8, steps 8 |
| ar-3.2m-native-dllm_a2d | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 200 | 5.7136 -> 5.2850 | 29,970 | 27,742 | 4.9728 | 0.4138 | from ar-3.2m-native-example, dllm loss on native torch, block 8 |
| dlm-3.2m-dllm | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.3498 -> 5.8577 | 28,142 | 26,121 | 5.6460 | 0.7586 | random init, dllm MDLMTrainer, lr 3e-04, block 8 |
| qwen3-0.6b-dllm_a2d | Qwen3-0.6B | dLLM-A2D | 598071296 | 200 | 12.9805 -> 5.0049 | 504 | 487 | 4.8013 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok, dllm loss |
| qwen3-0.6b-mini_a2d | Qwen3-0.6B | MiniMind-DLM-A2D | 598071296 | 200 | 13.5244 -> 4.9977 | 706 | 672 | 4.8166 | 0.7759 | LoRA r8 q/k/v/o, 0.3835% trainable, 76,160 tok |
