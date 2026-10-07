# smoke RESULTS（3.2M, 全格子 200→800 阶梯）

生成: collect_results.py 实测。t1 参照（同家族 AR-SFT）: trans: 4.7759; native: 4.5751

| 格子 | steps | assistant eval | rep-4(各prompt) | t1 ≤1.5x | t2 rep-4<0.5 | t3 格式 | t4 不回声 |
|---|---:|---:|---|---|---|---|---|
| ar-3.2m-trans-sft-smoke | 800 | 4.7759 | 0.00, 0.12, 0.00 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-native-sft-smoke | 800 | 4.5751 | 0.00, 0.08, 0.07 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-trans-sft-mini_a2d-sft-smoke | 800 | 5.4241 | 0.16, 0.25, 0.97 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-dllm_a2d-sft-smoke | 800 | 5.3479 | 0.30, 0.44, 0.92 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-mini_a2d-sft-smoke | 800 | 6.6497 | 0.67, 0.80, 0.79 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-dllm_a2d-sft-smoke | 800 | 6.7245 | 0.70, 0.23, 0.62 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-native-mini_a2d-sft-smoke | 800 | 6.5542 | 0.34, 0.36, 0.21 | ✅ | ✅ | ✅ | ✅ |
| dlm-3.2m-trans-sft-smoke | 800 | 6.4988 | 0.66, 0.36, 0.46 | — | ❌ | ✅ | ✅ |
| dlm-3.2m-dllm-sft-smoke | 800 | 6.4988 | 0.75, 0.43, 0.46 | — | ❌ | ✅ | ✅ |

_t1 = assistant eval loss / AR-SFT 参照 ≤ 1.5；t2 = rep-4 < 0.5；t3 = 回答非空且格式正确；t4 = 不回声。数据口径：sft-mini-6400 （52 万 token），smoke 只验管线不评判质量，200 步起步不达标追 400/800。_