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
| ar-3.2m-hf | Qwen-mini-HF | AR | 3213056 | 200 | 8.4414 -> 4.6970 | 47,096 | 41,046 | 4.6356 | 0.1724 | 524,288 tok, Tier 0 |
| ar-3.2m-native | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 46,428 | 41,402 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| ar-3.2m-hf-mini_a2d | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 45,110 | 40,145 | 4.9723 | 0.8103 | from ar-3.2m-hf-example, A2D lr 1e-04, block 8, steps 8 |
| ar-3.2m-hf-dllm_a2d | Qwen-mini-HF | dLLM-A2D | 3213056 | 200 | 5.7631 -> 5.2761 | 28,781 | 27,091 | 4.9765 | 0.0000 | from ar-3.2m-hf-example, dllm MDLMTrainer, block 8, steps 8 |
| dlm-3.2m-hf | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.4999 -> 5.8585 | 44,983 | 40,094 | 5.6427 | 0.7759 | random init, lr 3e-04, block 8, steps 8 |
| ar-3.2m-native-mini_a2d | Qwen-mini-MiniMind | MiniMind-DLM-A2D | 3213056 | 200 | 5.7642 -> 5.2846 | 48,448 | 42,900 | 4.9692 | 0.3966 | from ar-3.2m-native-example, lr 1e-04, block 8, steps 8 |
| ar-3.2m-native-dllm_a2d | Qwen-mini-MiniMind | dLLM-A2D | 3213056 | 200 | 5.7136 -> 5.2850 | 28,561 | 26,441 | 4.9728 | 0.4138 | from ar-3.2m-native-example, dllm loss on native torch, block 8 |
| dlm-3.2m-dllm | Qwen-mini-HF | Direct-DLM | 3213056 | 200 | 8.3498 -> 5.8577 | 28,989 | 26,862 | 5.6460 | 0.7586 | random init, dllm MDLMTrainer, lr 3e-04, block 8 |
| qwen3-0.6b-dllm_a2d | Qwen3-0.6B | dLLM-A2D | 598343680 | 200 | 12.9856 -> 5.0268 | 472 | 456 | 4.8238 | 0.7586 | LoRA r8 q/k/v/o, 0.3834% trainable, 76,160 tok, dllm loss |
| qwen3-0.6b-mini_a2d | Qwen3-0.6B | MiniMind-DLM-A2D | 598343680 | 200 | 13.5252 -> 5.0172 | 682 | 650 | 4.8397 | 0.7586 | LoRA r8 q/k/v/o, 0.3834% trainable, 76,160 tok |

> **最后两行已重跑**（2026-10-02，OPEN-1b 结清）。它们之前是**旧 mask 方案**
> （新增 `<|diff-mask|>` + `resize_token_embeddings(151670)`）的数，`params` 记的是
> 被 resize 掉的 266 行之后的 598,071,296。换成**复用 `151662` `<|fim_pad|>`**
> 后 embedding 不再变形，参数量回到 checkpoint 真实的 **598,343,680**。
>
> **不止 params 变了**：eval 4.8013 → **4.8238**、4.8166 → **4.8397**。
> MASK token 不同 → 加噪时替换掉的 embedding 行不同 → loss 会变。
> 这一点之前只是推测（`ISSUES.md` OPEN-1b 写「可能也会变」），现在有数了。

| ar-3.2m-hf-sft | Qwen-mini-HF | AR-SFT | 3213056 | 200 | 9.4496 -> 6.5408 | 46,151 | 40,644 | 6.3639 | 0.4310 | from ar-3.2m-hf-example, assistant-only loss, lr 1e-04 |
| ar-3.2m-native-sft | Qwen-mini-MiniMind | AR-SFT | 3213056 | 200 | 9.4158 -> 6.5252 | 46,374 | 42,477 | 6.3462 | 0.0517 | from ar-3.2m-native-example, assistant-only loss, native torch, lr 1e-04 |
| ar-3.2m-hf-mini_a2d-sft | Qwen-mini-HF | MINIMIND-A2D-SFT | 3213056 | 200 | 10.9498 -> 6.7526 | 46,370 | 40,968 | 6.4958 | 0.8448 | from ar-3.2m-hf-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-native-mini_a2d-sft | Qwen-mini-MiniMind | MINIMIND-A2D-SFT | 3213056 | 200 | 10.9032 -> 6.7901 | 46,888 | 42,992 | 6.5279 | 0.9655 | from ar-3.2m-native-mini_a2d-example, hand-written A2D, assistant spans only, lr 1e-04 |
| ar-3.2m-hf-dllm_a2d-sft | Qwen-mini-HF | DLLM-A2D-SFT | 3213056 | 200 | 10.4933 -> 6.7602 | 29,149 | 26,741 | 6.4880 | 0.8621 | from ar-3.2m-hf-dllm_a2d-example, dllm MDLMTrainer.compute_loss, labels==-100 outside spans, lr 1e-04 |
| ar-3.2m-native-dllm_a2d-sft | Qwen-mini-MiniMind | DLLM-A2D-SFT | 3213056 | 200 | 10.5050 -> 6.7670 | 29,531 | 27,801 | 6.5088 | 0.9138 | from ar-3.2m-native-dllm_a2d-example, dllm MDLMTrainer.compute_loss on the native model, lr 1e-04 |
| ar-3.2m-hf-sft-dllm_a2d | Qwen-mini-HF | DLLM-A2D | 3213056 | 200 | 6.4912 -> 5.3596 | 29,709 | 27,808 | 5.0685 | 0.2759 | from ar-3.2m-hf-sft-example, dllm MDLMTrainer.compute_loss, G3 original route, lr 1e-04 |
| ar-3.2m-hf-sft-mini_a2d | Qwen-mini-HF | MINIMIND-A2D | 3213056 | 200 | 6.5755 -> 5.3436 | 45,271 | 40,345 | 5.0425 | 0.7069 | from ar-3.2m-hf-sft-example, hand-written A2D, G3 original route, lr 1e-04 |
| ar-3.2m-native-sft-mini_a2d | Qwen-mini-MiniMind | MINIMIND-A2D | 3213056 | 200 | 6.3013 -> 5.3257 | 48,699 | 43,032 | 5.0410 | 0.6897 | from ar-3.2m-native-sft-example, hand-written A2D on native torch, G3 original route, lr 1e-04 |
| ar-3.2m-native-sft-dllm_a2d | Qwen-mini-MiniMind | DLLM-A2D | 3213056 | 200 | 6.1886 -> 5.3196 | 28,429 | 26,336 | 5.0283 | 0.6207 | from ar-3.2m-native-sft-example, dllm MDLMTrainer.compute_loss on native torch, G3 original route, lr 1e-04 |
| qwen3-0.6b-mini_a2d-sft | Qwen3-0.6B | MiniMind-DLM-A2D-SFT | 598343680 | 200 | 7.0291 -> 4.2466 | 697 | 662 | 3.8151 | 0.1897 | from qwen3-0.6b-mini_a2d-example, LoRA r8 q/k/v/o, 0.3834% trainable, 175,649 supervised tok, Qwen tokenizer |
