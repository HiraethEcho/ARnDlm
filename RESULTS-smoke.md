# smoke RESULTS（3.2M, 全格子 200 步起步）

生成: collect_results.py 实测; t1 参照 = ar-3.2m-trans-sft-smoke (assistant eval 4.7759)

| 格子 | steps | assistant eval | rep-4(各prompt) | t1 ≤1.5x | t2 rep-4<0.5 | t3 格式 | t4 不回声 |
|---|---:|---:|---|---|---|---|---|
| ar-3.2m-trans-sft-smoke | 800 | 4.7759 | 0.00, 0.00, 0.00 | ✅ | ✅ | ✅ | ✅ |
| ar-3.2m-native-sft-smoke | 800 | 4.6064 | 0.00, 0.04, 0.90 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-mini_a2d-sft-smoke | 800 | 5.4241 | 0.23, 0.36, 0.97 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-trans-sft-dllm_a2d-sft-smoke | 800 | 5.3479 | 0.38, 0.15, 0.93 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-mini_a2d-sft-smoke | 800 | 6.6916 | 0.70, 0.68, 0.73 | ✅ | ❌ | ✅ | ✅ |
| ar-3.2m-dllm_a2d-sft-smoke | 800 | 6.6816 | 0.46, 0.30, 0.33 | ✅ | ✅ | ✅ | ✅ |
| dlm-3.2m-trans-sft-smoke | 800 | 6.7232 | 0.44, 0.31, 0.21 | ✅ | ✅ | ✅ | ✅ |
| dlm-3.2m-dllm-sft-smoke | 800 | 6.7232 | 0.33, 0.39, 0.16 | ✅ | ✅ | ✅ | ✅ |

_t1 = assistant eval loss / AR-SFT 参照 ≤ 1.5；t2 = rep-4 < 0.5；t3 = 回答非空且格式正确；t4 = 不回声。数据口径：sft-mini-6400 （52 万 token），smoke 只验管线不评判质量，200 步起步不达标追 400/800。_