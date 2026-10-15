# ROADMAP — x.py 填写顺序

规矩（PLAN.md + AGENTS.md）：`x.py` 由你填；填完用**同 seed 同输入**与
`x-example.py` 逐位对拍，一致才算完。每个文件对应的验收测试在 `test/NN-*.py`，
先把测试当验收标准读一遍，再写实现。

骨架与参考实现**不在同一目录**：骨架在 `models/` `prepare/` `trainer/` `a2d/` `scripts/`，
参考实现集中在 `common/` 与 `src/`。下表两列都是全路径。

## 对拍怎么做（每个文件都适用）

```sh
# ① 基准: 用参考实现跑测试, 记下关键数字（末行带锚点与 IMPL）
.venv/bin/python test/06-c1-logits.py          # -> 2.146e-06  [IMPL=example]

# ② 候选: 同一条命令加 IMPL=x, 加载目标自动换成新布局里的 x.py
#    映射表在 test/_impl.py（参考实现路径 -> 骨架路径）；骨架还空着会直接报错
IMPL=x .venv/bin/python test/06-c1-logits.py

# ③ 数字逐位一致 = 过; 差一点先看 logs/dev-old/04 的踩坑清单
```

一天的顺序是**从底向上**：底座（`models/` config + tokenizer，`prepare/`）→ 模型实现 → `trainer/` 主干 → 各 stage → eval → 收尾。
下面每格的「依赖」列没满足就别开工（对拍会撞 import 错误）。

## 阶段 A · 底座（PLAN P1；骨架在 `models/` + `prepare/`）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| A1 | `models/config.py` | `common/config-example.py` | `test/01-config.py` 全过 | — | 半小时 |
| A2 | `prepare/seed.py` | `common/seed-example.py` | `test/02` 的 [1]-[3] | A1 | 15 分钟 |
| A3 | `prepare/device.py` | `common/device-example.py` | `test/02` 的 [4]-[6] | A1 | 15 分钟 |
| A4 | `models/tokenizer.py` | `common/tokenizer-example.py` | `test/03` | A1 | 半小时 |
| A5 | `prepare/data.py` | `common/data-example.py` | `test/04`（含四道自检负例） | A1 A4 | 一个半小时 |
| A6 | `prepare/metrics.py` | `common/metrics-example.py` | `test/05` | — | 半小时 |
| A7 | `prepare/prepare.py` | `common/prepare-example.py` | `test/08` | A1 A4 | 一小时 |
| A8 | `prepare/sft-prepare.py` | `common/sft-prepare-example.py` | `test/15` | A1 A4 | 一小时 |

**A1 的坑**：不要加 `from __future__ import annotations`（`dataclass.f.type` 会变
字符串，类型校验静默失效）；yaml 浮点必须 `3.0e-4` 形式；`lr_at` 的 warmup 是
**HF 语义**（step0 = 0，峰在 warmup 整点），不是 `(step+1)/warmup`。

**A5 的坑**：位置→文档映射用 `searchsorted(virt_start)`，用 `virt_end` 会让
第 0 篇整篇错位（test/04 的朴素对拍专抓这个）；批次顺序必须是 `f(seed, step)`
纯函数（C4 的根）。

**A8 已定**：SFT 数据生产者单独成格 `prepare/sft-prepare.py`（骨架已建），不并入
`prepare/prepare.py`；SPEC §7 与 §3.3 已同步。

## 阶段 B · models/（PLAN P2，C1 是整个项目的核心对拍）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| B1 | `models/model-trans.py` | `src/models/trans-example.py` | `test/06` 参数量 + [2] 直拷 | A1 | 半小时 |
| B2 | `models/model.py` | `src/models/native/model-example.py` | `test/06` 全部（C1 ≤1e-5、tier1 形状守门） | B1 | 半天 |

**B2 的坑**（都在 test/06 里被钉住）：QK-Norm 在 RoPE **之前**且作用在 head_dim 上；
GQA 用 repeat-interleave 语义；RMSNorm 方差在 fp32 里算；`tie` 必须在
`apply(_init_weights)` **之后**（否则 embed 被重画一次）；`is_causal=False` 是
A2D 唯一要动的开关，权重一个不动。

## 阶段 C · trainer 主干（PLAN P3）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| C1 | `trainer/state.py` | `src/training/state-example.py` | `test/07` | A2 | 半小时 |
| C2 | `trainer/ar_native.py` | `src/training/ar_native-example.py` | `test/07` 全过 + smoke 20 步 | B2 C1 | 一小时 |
| C3 | `trainer/ar_trans.py` | `src/training/ar_trans-example.py` | `test/07` [6]（HF resume）+ smoke 20 步 | B1 | 一小时 |
| C4 | `trainer/main.py` | `src/training/main-example.py` | 两条冒烟命令 + 命名合 SPEC §1.2 | C2 C3 | 一个半小时 |

**C3 的坑**：Trainer 对「有 `__len__` 的数据集」会静默换 RandomSampler——批次顺序
被洗掉；用 `IterableDataset` 且**不要定义 `__len__`**；`logging_steps` 在 smoke 步数
小于 `log_every` 时要降下来，否则没有 loss 点可看。

**C4 的坑**：`--name` 覆盖要强制 `-smoke` 后缀；A2D 默认名必须从 `--base` 的
stem 推出 impl 段，否则 G5 两个 base 撞名。

## 阶段 D · SFT（PLAN P5）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| D1 | `trainer/sft.py` | `src/training/sft-example.py` | `test/09`（T4b 区间 / T4a 还原 / T4f 加噪约束）+ 800 步 smoke | A5 A8 C4 | 一天 |

**D1 的坑**：span 的 header 与 body **分别编码再拼**（整条编码后减 header 长度
会错 1–2 token，静默）；采样前缀必须逐 token 拼 `[BOS] user\n{q} [EOS] assistant\n`，
不能写 f-string（轮分隔是 EOS 不是换行）。

## 阶段 E · A2D / DLM（PLAN P6）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| E1 | `a2d/a2d.py` | `common/a2d-example.py` | `test/12`（分母语义手算对照） | A6 | 一小时 |
| E2 | `a2d/a2d_mini.py` | `src/training/a2d_mini-example.py` | `test/13` [1][2][4]（C2/C3 + T4g） | B2 C2 E1 | 一小时 |
| E3 | `a2d/a2d_dllm.py` | `src/training/a2d_dllm-example.py` | `test/13` [3][5]+ 三路 C2/C3 0.0 | E2 | 一个半小时 |
| E4 | `trainer/dlm.py` | `src/training/dlm-example.py`（2026-10-14 已补：Direct-DLM 独立成格） | `dlm-3.2m-{trans,dllm}-smoke` 两条（trans 5.0192 / dllm 5.0691） | E2 E3 C4 | 一小时 |
| E5 | `trainer/dpo.py` | `src/training/dpo-example.py` | `test/16-dpo.py`（2026-10-14 已补：shift 逐位 / ln2 恒等 / margin 正负 / lr×0.1） | D1 C4 | 半天 |

**E3 的坑**：dllm 的 `compute_loss` 收不到 `attention_mask` 时会**静默走因果
forward**——必须注入 4D 全零 mask（test/13 [5] 专门钉这个）；`right_shift_logits`
保持 False，BOS 前置不会发生。

## 阶段 F · eval（PLAN P5/P7）

| # | 文件 | 参考实现 | 验收 | 依赖 | 预计 |
|---|---|---|---|---|---|
| F1 | `scripts/chat.py` | `src/eval/chat-example.py` | `test/10`（含 1.5 / 0.5 边界负例） | D1 E1 | 一小时 |
| F2 | `scripts/repl.py` | `src/eval/repl-example.py` | `test/11`（4 ckpt × 2 模式 + 多轮） | F1 | 一小时 |

## 阶段 G · 收尾（x.py 全部填完之后）

| # | 事项 | 说明 |
|---|---|---|
| G1 | 拆 importlib 桥 | `-example.py` 里的 `_load("…-example.py")` 全部换成正常 import；tests 的加载目标从 `-example` 换成新布局的骨架 |
| G2 | `scripts/collect_results.py` 换真模块 | 它现在 import 的是 example 模块 |
| G3 | 全测试回归 | `for t in test/0*.py test/1*.py; do .venv/bin/python "$t" \| tail -1; done` 全 PASS |
| G4 | smoke 全链复跑 | 用 x.py 版本重跑 `RESULTS-smoke.md` 的格子，与 example 版数字对拍 |
| G5 | 更新 STATE.md | 把「x.py 是空的」这一条去掉，其余能力表按实际改 |

## 时间预算参考（全部合计）

A 组 ≈ 5 小时 · B 组 ≈ 6 小时 · C 组 ≈ 4.5 小时 · D 组 ≈ 1 天 · E 组 ≈ 1.5 天 · F 组 ≈ 2 小时 · G 组 ≈ 2 小时。
**合计约 4 个工作日**（按每天 6 小时有效编码算）。
