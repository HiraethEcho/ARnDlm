# RESULTS — A 版 smoke 实测

由 `example/common/report.py` 自动追加，不要手改。

Tier 0 · 200 steps · tinystories-6400 · 只验 pipeline，不看模型能力。

| cell | base | method | params | steps | loss first->last | train-only tok/s | wall tok/s | eval loss | rep-4 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-ar-example | Qwen-mini-HF | AR | 3213056 | 200 | 8.44 -> 4.70 | 45,760.00 | 40,318.50 | 4.64 | 0.17 | 524,288 tok, Tier 0 |
