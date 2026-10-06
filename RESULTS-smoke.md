# smoke RESULTS（3.2M, 全格子 200 步起步）

生成: collect_results.py 实测; t1 参照 = ar-3.2m-trans-sft-smoke (assistant eval 4.7759)

| 格子 | steps | assistant eval | rep-4(各prompt) | t1 ≤1.5x | t2 rep-4<0.5 | t3 格式 | t4 不回声 |
|---|---:|---:|---|---|---|---|---|
| ar-3.2m-trans-sft-smoke | 800 | 4.7759 | 0.00, 0.00, 0.10 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-native-sft-smoke | 800 | 4.6083 | 0.03, 0.00, 0.90 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-mini_a2d-sft-smoke | 200 | 5.3839 | 0.49, 0.52, 0.41 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-dllm_a2d-sft-smoke | 200 | 5.3728 | 0.15, 0.43, 0.15 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-mini_a2d-sft-smoke | 200 | 6.9359 | 0.74, 0.66, 0.77 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-dllm_a2d-sft-smoke | 200 | 6.9359 | 0.70, 0.85, 0.61 | ✅ | ❌ | ✅ | ✅ |
| dlm-3.2m-trans-sft-smoke | 200 | 9.4796 | 0.85, 0.85, 0.92 | ❌ | ❌ | ✅ | ✅ |
| dlm-3.2m-dllm-sft-smoke | 200 | 9.4796 | 0.92, 0.98, 0.92 | ❌ | ❌ | ✅ | ✅ |

_t1 = assistant eval loss / AR-SFT 参照 ≤ 1.5；t2 = rep-4 < 0.5；t3 = 回答非空且格式正确；t4 = 不回声。数据口径：sft-mini-6400 （52 万 token），smoke 只验管线不评判质量，200 步起步不达标追 400/800。_