# smoke RESULTS（3.2M, 全格子 200 步起步）

生成: collect_results.py 实测; t1 参照 = ar-3.2m-trans-sft-smoke (assistant eval 4.7759)

| 格子 | steps | assistant eval | rep-4(各prompt) | t1 ≤1.5x | t2 rep-4<0.5 | t3 格式 | t4 不回声 |
|---|---:|---:|---|---|---|---|---|
| ar-3.2m-trans-sft-smoke | 800 | 4.7759 | 0.26, 0.31, 0.00 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-native-sft-smoke | 800 | 4.6064 | 0.00, 0.09, 0.22 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-trans-sft-mini_a2d-sft-smoke | 800 | 5.4697 | 0.20, 0.13, 0.97 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-dllm_a2d-sft-smoke | 800 | 5.4312 | 0.33, 0.16, 0.92 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-mini_a2d-sft-smoke | 800 | 7.4324 | 0.72, 0.85, 0.53 | ❌ | ❌ | ✅ | ✅ |
| ar-3.2m-dllm_a2d-sft-smoke | 800 | 6.7895 | 0.28, 0.39, 0.38 | ✅ | ✅ | ✅ | ✅ |
| dlm-3.2m-trans-sft-smoke | 800 | 6.8143 | 0.31, 0.28, 0.18 | ✅ | ✅ | ✅ | ✅ |
| dlm-3.2m-dllm-sft-smoke | 800 | 6.8143 | 0.33, 0.33, 0.33 | ✅ | ✅ | ✅ | ✅ |

_t1 = assistant eval loss / AR-SFT 参照 ≤ 1.5；t2 = rep-4 < 0.5；t3 = 回答非空且格式正确；t4 = 不回声。数据口径：sft-mini-6400 （52 万 token），smoke 只验管线不评判质量，200 步起步不达标追 400/800。_