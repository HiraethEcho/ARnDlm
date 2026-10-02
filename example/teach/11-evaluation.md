# Step 11 — `evaluation/`

**你要写的**：`evaluation/`（生成 + 评测 + repetition rate）
**实验版参考**：`example/common/metrics.py` + `example/common/cell.py::finish_cell`
**验收**：重复率数值和 实验版一致；`results` 表的每一行都带 rep-4。

---

## 11.1 重复率是核心指标，不是附加项

`AGENTS.md` 写死了：**DLM 失败最典型的表现就是复读，而 loss 照样下降**
（`notes/03` §11 作者自曝的负面结果就是「loss 正常，输出大量重复」）。

所以 `common/report.py` 的 `rep4` 是**必填参数**，不是可选的额外项：
十行手抄的表格，就是十次「把 train-only 吞吐填进 wall-clock 那一列」和
「把重复率留空」的机会。

## 11.2 rep-n 的定义

```
rep-n = 1 - |unique n-grams| / |total n-grams|
```

按序列分别统计后合并。长度 < n 的序列**不参与**（返回 nan），
不是记 0.0 —— 把退化样本记成 0 会让坏掉的采样器看起来完美。

实验版 `test_common.py` 拿闭式解对拍过：

```
looped = list(range(8)) * 8      # 61 个 4-gram 窗口，8 个不同
        → rep4 = 1 - 8/61 = 0.8689   ← 实现给出 0.8689
```

## 11.3 rep-4 是分布性质，不是对拍指标

两行同样的权重、同样的解码参数，两次采样的 rep-4 也会不同：
采样有随机数，32 个 token 的样本噪声极大。

**它能回答**：这个模型是不是在复读。
**它不能回答**：实验版和 实践版谁更好。

实验版实测（Tier 0, 200 步）：

| 格 | rep-4 |
|---|---|
| ar-3.2m-hf | 0.1724 |
| ar-3.2m-native | 0.3276 |
| ar-3.2m-hf-mini_a2d | 0.7586 |
| dlm-3.2m-hf | 0.7931 |
| qwen3-0.6b-*-lora | 0.9138 |

读法：**Direct-DLM（0.79）比 A2D（0.76）复读略严重**，
**Qwen3-0.6B + LoRA 几乎全在复读（0.91）**。
行2 的 A2D 是 0.4655 —— 同一条路线换一个 base，差异就有 0.29，
这个量级的分散度本身就说明 32 token 的样本不能用来给实现排名。
方向和 `notes/03` 一致：A2D 继承机制，从零训要自己学机制，
LoRA 只有 0.38% 的参数可动。

## 11.4 要判断复读，长生成 + 多 prompt

smoke 阶段 32 token / 2 prompt 只能「报出来」，不能「判断」。
正式评测（`SPEC` Evaluation plan）：

- 固定 prompt：`Once upon a time` / `The future of AI is`
- 长度：几百 token
- 报：sampling steps、生成速度、output quality、**repetition rate**

## 11.5 两个吞吐都要报

`eval.repetition_ngram` 和 `report_train_only_tps` / `report_wall_tps`
在 `configs/base.yaml` 里是**默认开启**的，不是可选项。

实验版自己踩过：`summary` 在 eval/sample 之前组装，wall-clock 于是
和 train-only 三个数字位一样 —— 那就是 `notes/05` Bug 1 的形状，
自己复现了一遍（见 commit `2928fba`）。

## 11.6 验收命令

```sh
.venv/bin/python example/check_ab.py decode-bench   # 解码成本
cat example/RESULTS.md                              # 10 行，每行带 rep-4
```

`RESULTS.md` 由 `common/report.py` 写入，**不要手改**；
重跑同一格会覆盖旧行（不覆盖就会出现两行不同步数、看不出引用哪行）。