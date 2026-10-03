# RESULTS — 实验版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

> **`tokens` 列有两个数，不要混。**`steps` = 200 是本次实际跑的步数，
> 每步 `batch 8 × seq 128` → **实际训练 204,800 token**（每格 `summary.json`
> 的 `throughput.tokens` 就是这个）。
> `notes` 列里的「524,288 tok」是**数据切片的大小**（4096 块 × 128，
> 见 `common/data.py` 的 `n_tokens = n_blocks × block_size`），即可用上限，
> 不是消耗量。两个数都对，量的是不同的东西。

> **SFT 格的 `eval loss` 与预训练格不可比。** SFT 的 loss 只在 assistant 段
> （每块 128 token 里约 62 个）计算，预训练的 loss 在全部 token 上计算。
> 两者只在 SFT 之间比、或在预训练之间比，**不能跨列比大小**。
> SFT 行之间也不应直接比——两行从各自独立的 AR checkpoint 出发，差异来自上游。
> 要隔离 SFT 路径本身，用 `check_ab c7`：同一份权重进两个实现，差 0.000e+00。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ar-3.2m-hf | Qwen-mini-HF | AR | 3213056 | 200 | 8.4414 -> 4.6970 | 46,816 | 41,108 | 4.6356 | 0.1724 | 524,288 tok, Tier 0 |
| ar-3.2m-native | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 48,384 | 42,990 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| ar-3.2m-hf-mini_a2d | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 45,432 | 40,043 | 4.9723 | 0.8103 | from ar-3.2m-hf-example, A2D lr 1e-04, block 8, steps 8 |
| ar-3.2m-hf-dllm_a2d | Qwen-mini-HF | dLLM-A2D | 3213056 | 200 | 5.7631 -> 5.2761 | 30,078 | 28,026 | 4.9765 | 0.0000 | from ar-3.2m-hf-example, dllm MDLMTrainer, block 8, steps 8 |
| dlm-3.2m-hf | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.4999 -> 5.8585 | 46,603 | 40,738 | 5.6427 | 0.7759 | random init, lr 3e-04, block 8, steps 8 |
| ar-3.2m-native-mini_a2d | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 200 | 5.7642 -> 5.2846 | 49,366 | 43,659 | 4.9692 | 0.3966 | from ar-3.2m-native-example, lr 1e-04, block 8, steps 8 |
| ar-3.2m-native-dllm_a2d | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 200 | 5.7136 -> 5.2850 | 31,779 | 29,120 | 4.9728 | 0.4138 | from ar-3.2m-native-example, dllm loss on native torch, block 8 |
| dlm-3.2m-dllm | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.3498 -> 5.8577 | 30,403 | 27,809 | 5.6460 | 0.7586 | random init, dllm MDLMTrainer, lr 3e-04, block 8 |
| qwen3-0.6b-dllm_a2d | Qwen3-0.6B | dLLM-A2D | 598343680 | 200 | 12.9856 -> 5.0268 | 526 | 507 | 4.8238 | 0.7586 | LoRA r8 q/k/v/o, 0.3834% trainable, 76,160 tok, dllm loss |
| qwen3-0.6b-mini_a2d | Qwen3-0.6B | MiniMind-DLM-A2D | 598343680 | 200 | 13.5252 -> 5.0172 | 792 | 751 | 4.8397 | 0.7586 | LoRA r8 q/k/v/o, 0.3834% trainable, 76,160 tok |

> **最后两行已重跑**（2026-10-02，OPEN-1b 结清）。它们之前是**旧 mask 方案**
> （新增 `<|diff-mask|>` + `resize_token_embeddings(151670)`）的数，`params` 记的是
> 被 resize 掉的 266 行之后的 598,071,296。换成**复用 `151662` `<|fim_pad|>`**
> 后 embedding 不再变形，参数量回到 checkpoint 真实的 **598,343,680**。
>
> **不止 params 变了**：eval 4.8013 → **4.8238**、4.8166 → **4.8397**。
> MASK token 不同 → 加噪时替换掉的 embedding 行不同 → loss 会变。
> 这一点之前只是推测（`ISSUES.md` OPEN-1b 写「可能也会变」），现在有数了。

| ar-3.2m-hf-sft | Qwen-mini-HF | AR-SFT | 3213056 | 200 | 9.4364 -> 6.4502 | 45,493 | 40,889 | 6.2659 | 0.0000 | from ar-3.2m-hf-example, assistant-only loss, lr 1e-04 |
| ar-3.2m-native-sft | Qwen-mini-MiniMind | AR-SFT | 3213056 | 200 | 9.4190 -> 6.4344 | 45,794 | 41,906 | 6.2475 | 0.3966 | from ar-3.2m-native-example, assistant-only loss, native torch, lr 1e-04 |
| ar-3.2m-hf-mini_a2d-sft | Qwen-mini-HF | MINIMIND-A2D-SFT | 3213056 | 200 | 10.8812 -> 6.6375 | 45,038 | 39,514 | 6.3872 | 0.8621 | from ar-3.2m-hf-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-native-mini_a2d-sft | Qwen-mini-MiniMind | MINIMIND-A2D-SFT | 3213056 | 200 | 10.8871 -> 6.6492 | 45,723 | 41,880 | 6.4133 | 0.9828 | from ar-3.2m-native-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-hf-dllm_a2d-sft | Qwen-mini-HF | DLLM-A2D-SFT | 3213056 | 200 | 10.4530 -> 6.6382 | 29,097 | 27,368 | 6.3845 | 0.8621 | from ar-3.2m-hf-dllm_a2d-example, dllm MDLMTrainer.compute_loss, labels==-100 outside spans, lr 1e-04 |
| ar-3.2m-native-dllm_a2d-sft | Qwen-mini-MiniMind | DLLM-A2D-SFT | 3213056 | 200 | 10.4663 -> 6.6522 | 30,062 | 27,615 | 6.4125 | 0.8448 | from ar-3.2m-native-dllm_a2d-example, dllm MDLMTrainer.compute_loss on the native model, lr 1e-04 |
| ar-3.2m-hf-sft-dllm_a2d | Qwen-mini-HF | DLLM-A2D | 3213056 | 200 | 6.5488 -> 5.3370 | 28,195 | 26,597 | 5.0378 | 0.4655 | from ar-3.2m-hf-sft-example, dllm MDLMTrainer.compute_loss, G3 original route, lr 1e-04 |
| ar-3.2m-hf-sft-mini_a2d | Qwen-mini-HF | MINIMIND-A2D | 3213056 | 200 | 6.6252 -> 5.3353 | 45,565 | 40,184 | 5.0315 | 0.8621 | from ar-3.2m-hf-sft-example, hand-written A2D, G3 original route, lr 1e-04 |
| ar-3.2m-native-sft-mini_a2d | Qwen-mini-MiniMind | MINIMIND-A2D | 3213056 | 200 | 6.2622 -> 5.3218 | 47,774 | 42,232 | 5.0335 | 0.6724 | from ar-3.2m-native-sft-example, hand-written A2D on native torch, G3 original route, lr 1e-04 |
| ar-3.2m-native-sft-dllm_a2d | Qwen-mini-MiniMind | DLLM-A2D | 3213056 | 200 | 6.1908 -> 5.3229 | 29,147 | 26,932 | 5.0376 | 0.6724 | from ar-3.2m-native-sft-example, dllm MDLMTrainer.compute_loss on native torch, G3 original route, lr 1e-04 |
| qwen3-0.6b-mini_a2d-sft | Qwen3-0.6B | MiniMind-DLM-A2D-SFT | 598343680 | 200 | 7.1127 -> 4.1766 | 723 | 686 | 3.7559 | 0.6047 | from qwen3-0.6b-mini_a2d-example, LoRA r8 q/k/v/o, 0.3834% trainable, 179,614 supervised tok, Qwen tokenizer |
| ar-3.2m-hf-sft-mini_a2d-sft | Qwen-mini-HF | MINIMIND-A2D-SFT | 3213056 | 200 | 8.1965 -> 6.3050 | 45,922 | 40,774 | 5.9773 | 0.8776 | from ar-3.2m-hf-sft-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-hf-sft-dllm_a2d-sft | Qwen-mini-HF | DLLM-A2D-SFT | 3213056 | 200 | 7.9108 -> 6.3097 | 28,028 | 26,421 | 5.9776 | 0.1034 | from ar-3.2m-hf-sft-dllm_a2d-example, dllm MDLMTrainer.compute_loss, labels==-100 outside spans, lr 1e-04 |
| ar-3.2m-native-sft-mini_a2d-sft | Qwen-mini-MiniMind | MINIMIND-A2D-SFT | 3213056 | 200 | 8.2246 -> 6.3146 | 48,164 | 43,998 | 5.9713 | 0.6034 | from ar-3.2m-native-sft-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-native-sft-dllm_a2d-sft | Qwen-mini-MiniMind | DLLM-A2D-SFT | 3213056 | 200 | 7.9239 -> 6.3187 | 30,096 | 27,776 | 5.9709 | 0.0345 | from ar-3.2m-native-sft-dllm_a2d-example, dllm MDLMTrainer.compute_loss on the native model, lr 1e-04 |
| ar-3.2m-hf-sft-dpo | Qwen-mini-HF | AR-SFT-DPO | 3213056 | 200 | 0.8704 -> 0.7744 | 36,056 | 33,255 | - | 0.0000 | from ar-3.2m-hf-sft-example, DPO beta 0.1, 277/4000 pairs fit ctx 128, ref = frozen SFT |
| dlm-3.2m-hf-sft | Qwen-mini-HF | MINIMIND-A2D-SFT | 3213056 | 200 | 12.6040 -> 6.9338 | 45,484 | 40,106 | 6.6882 | 0.2069 | from dlm-3.2m-hf-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-native-sft-dpo | Qwen-mini-MiniMind | AR-SFT-DPO | 3213056 | 200 | 0.8853 -> 0.7623 | 37,042 | 34,494 | - | 0.9483 | from ar-3.2m-native-sft-example, DPO beta 0.1, 277/4000 pairs fit ctx 128, ref = frozen SFT |
| dlm-3.2m-dllm-sft | Qwen-mini-HF | DLLM-A2D-SFT | 3213056 | 200 | 12.1911 -> 6.9351 | 27,714 | 26,186 | 6.6878 | 0.9138 | from dlm-3.2m-dllm-example, dllm MDLMTrainer.compute_loss, labels==-100 outside spans, lr 1e-04 |
