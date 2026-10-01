# RESULTS — A 版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-ar-example | Qwen-mini-HF | AR | 3213056 | 200 | 8.4414 -> 4.6970 | 46,192 | 40,650 | 4.6356 | 0.1724 | 524,288 tok, Tier 0 |
| minimind-ar-example | Qwen-mini-MiniMind | AR | 3213056 | 200 | 8.4165 -> 4.6932 | 46,649 | 41,837 | 4.6283 | 0.3276 | 524,288 tok, Tier 0, native torch |
| hf-minimind-a2d-example | Qwen-mini-HF | MiniMind-DLM-A2D | 3213056 | 200 | 5.8107 -> 5.2759 | 45,304 | 39,295 | 4.9723 | 0.5172 | from hf-ar-example, A2D lr 1e-04 |
