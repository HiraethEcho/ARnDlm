# GOAL — 实验版全流程

**创建**：2026-10-02 21:48 CST　**状态**：待批准，未执行
**授权**：本文件获批后，本次长跑允许 AI 每个 milestone 自动 commit。不 push。

---

## 1. 目标

把 `example/`（实验版）从「10 格跑通」推进到 **SPEC 2.3 全部剩余格跑通**，
并让 `example/RESULTS.md` 里每一行都对得上磁盘上的 checkpoint。

**不做**：实践版（根目录七个目录）的任何实现代码。**不 push。**

---

## 2. 访谈结论（用户逐条拍板）

| # | 问题 | 决定 |
|---|---|---|
| Q1 | GPU | **自由使用**，不设限制 |
| Q2 | 范围 | 先做完 SFT 列 13 格 + `dlm-3.2m-dllm`；**若在截止时间前完成，继续做 DPO 列** → 后经澄清：**无条件全做** |
| Q3 | SFT 数据 | `sft_t2t_mini.jsonl`（1.74 GB / 905,718 行，已软链到 `minimind_dataset/`） |
| Q4 | `assistant_spans` 语义 | **只在 assistant 段算 loss**（标准 SFT） |
| Q5 | OPEN-2 `t` 分布 | **保持两套**，对拍时两边都用 `epsilon=1e-3` 并重新播种 |
| Q6 | OPEN-3 `steps` 语义 | **CLI 上两个参数都给**：`--dlm-steps`（每块）+ `--dlm-total-steps`（全序列预算） |
| Q7 | commit | **每个 milestone 自动 commit** |

---

## 3. 期限：无截止，做完为止；10:00 是后续任务的闸门

用户 2026-10-02 21:48 答复：「**一直做直到完成**」——M1–M9 全部做完，包括 DPO。

**但有一个条件链**（闸门时刻 = **2026-10-03 10:00 CST**）：

| 条件 | 任务 |
|---|---|
| 本文件 M1–M9 在 10:00 CST 前完成 | 用一个**不自动关闭的 subagent** 执行 `~/Projects/minimind/GOAL.md` |
| 上一个也在 10:00 CST 前完成 | 回头对本任务做一遍 review：查 bug + 完善教程 |

时钟基准：`date` 与 `TZ=Asia/Shanghai date` 同为 CST，本机无时区偏差。

---

## 4. 数据事实（2026-10-02 实测，不是推测）

### 4.1 `sft_t2t_mini.jsonl`

| 项 | 值 |
|---|---|
| 大小 / 行数 | 1,739,201,170 B / **905,718** 行 |
| 顶层结构 | `{"conversations": [ {...}, ... ]}` —— **没有 `text` 键** |
| 消息结构 | `{"role": "user"\|"assistant", "content": str}` |
| 采样 20,000 行 | 平均 7.2 条消息/对话，范围 2–16 |
| role 分布 | user 71,745 · assistant 71,669 |
| 是否有 system | **无**。20,000 条对话的首条消息全是 `user` |
| `reasoning_content` | **19.0% 的 assistant 消息带这个字段**（约 300 字符的 think 块） |

### 4.2 `notes/06-datasets.md` 需要更正的地方

记的是 `{"conversations": [{"role": "user", "content": ...}]}` —— **正确**。
但**漏了 `reasoning_content`**，而它是 19% 的 assistant 消息都有的字段。
展平脚本若按 `content` 单字段处理，会静默丢掉这 19% 的思考内容。

### 4.3 `reasoning_content` 怎么处理　→ **用户已定：方案 A**

三个选项，本文件按 **A** 写入计划（**用户 2026-10-02 已确认**）：

| | 做法 | 后果 |
|---|---|---|
| **A（已选）** | 拼进 assistant 的文本，`<thinking>...</thinking>` + `content`，**loss 覆盖它** | 保留 think 能力，与 MiniMind 上游一致 |
| B | 拼进文本但 **loss 不覆盖**（进 assistant span 的前缀部分） | 模型学不到 think |
| C | 完全丢弃 | 丢掉 19% 的内容，上游是当正常文本训的 |

**A 的已知代价**（写在这里而不是忘了）：think 块比答案长 10 倍
（298 vs 28 字符），算进 loss 会让 loss 主要由 think 驱动而不是最终答案。
所以 SFT 格的 eval loss 与预训练格**不可直接比大小**，只能在 SFT 之间比。

> 旁证：`~/Projects/minimind/GOAL.md` 里同一个数据集选的是**方案 C（去
> `reasoning_content`）**。两边口径不同，**两个项目的 loss 数字不可互引**。

---

## 5. Milestone 清单（14 项 + 收尾）

每个 milestone 走固定四步：**correct（先写能失败的测试）→ quality（去死代码）
→ iterate 2–3 轮 → record（写进文件）→ commit**。

### M1 · SFT 展平器　`example/common/sft.py`

产出 `prepared/sft-mini-6400/`：`tokens.bin` + `doc_lengths.npy` +
**`assistant_spans.npy`** + `meta.json`。

| 内容 | 验收 |
|---|---|
| `conversations` → 展平成 ChatML 风格文本 | `test_sft_flatten.py`：展平后 decode 回来能对上一条真实对话 |
| **每条 assistant 消息的 token 区间写进 spans** | spans 的区间 decode 出来必须是 assistant 的内容，不得含 user 段 |
| `reasoning_content` 按 §4.3 的 A 处理 | 19% 的消息带 think，测试要专门覆盖 |
| 只在 assistant 段算 loss | 人工构造「user 段 token 全对、assistant 段全错」的 batch，loss 必须非零 |
| tokenizer 用项目 6400 BPE，`--append-eos` | vocab 6400 · MASK 27 不得出现在语料里（沿用 `data.py` 的检查） |

**阻断**：§4.3 的 A/B/C 未点头前不动手。

### M2 · 带 loss mask 的训练循环

`example/common/loop.py` 加 span 加权的 CE。多边形：一个 batch 里不同样本的
assistant 比例不同，必须**按 token 数归一**，不能按样本数。

验收：`test_loss_mask.py` —— 全部 mask 掉的 batch loss = 0；部分 mask 时 loss
等于只在被 mask 位置上算的值。

### M3 · `ar-3.2m-{hf,native}-sft`　G3 对照（P0 ×2）

AR 预训练 → SFT。**这两格是后面所有 SFT 格的基线**，没有它们，G3 的
「先 SFT 还是先 A2D」无法回答。

### M4 · `ar-3.2m-{hf,native}-{mini,dllm}_a2d` + 各自 `-sft`　G3 新路线（P0 ×4）

**A2D → SFT**（新路线）。与 M5 的原路线构成 G3 的核心对照。

### M5 · `ar-3.2m-{hf,native}-sft-{mini,dllm}_a2d`　G3 原路线（P0 ×4）

**SFT → A2D**（原路线）。SPEC 说原路线在前，因为 Qwen3-0.6B 和 dllm 的 A2D
作用对象本来就是已经能 chat 的模型。

### M6 · `dlm-3.2m-dllm`　G2（P1 ×1，**无前置**）

代码已存在（`example/base/hf/direct_dlm_dllm.py` + `train_direct_dlm_dllm.py`），
`smoke_all.sh` 已占 cell 8，**但从没跑过**——最后一次全量 smoke 时 cell 8 还是
`native-direct-dlm`。本 milestone = 跑它 + 补 `smoke_final.log` 的证据。

### M7 · `qwen3-0.6b-mini_a2d-sft`　G4（P1 ×1）

行 3 分支。**先做 M1–M6 再做这格**：它最慢，且要用到 §4.3 的结论。

### M8 · 收尾：RESULTS.md 与代码对齐

- 最后两行 LoRA（旧 mask 方案）重跑刷新，删掉 §4.3 的注记
- 格名对齐（cell 8 与 qwen3 两格在 `4bc81bc` / `21d85c7` 里改过名，
  `RESULTS.md` 还是旧名）
- 每一行都要能在 `logs/run/example/` 找到对应的 `summary.json`

### M9 · DPO 列

`sft_t2t_mini.jsonl` **不含偏好对**（每条对话只有 user/assistant，没有 chosen/rejected）。
需用 `data/raw/minimind_dataset/dpo.jsonl`（17,166 行）。
**先确认那份的 schema，再决定做不做**——它比 SFT 多一层 pair 构造。

---

## 6. 已知阻断与开放问题

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | ~~截止时间未给~~（§3） | 决定 M9 做不做 | ✅ 已解：**无截止，全做**；闸门改为 2026-10-03 10:00 CST |
| 2 | ~~`reasoning_content` 怎么处理~~（§4.3） | 曾阻断 M1 | ✅ 已解：**方案 A**（loss 覆盖 think） |
| 3 | ~~超长对话截断策略~~（§8） | 曾阻断 M1 | ✅ 已解：**按整轮丢弃，不半句截断** |
| 3 | `RESULTS.md` 最后两行是旧 mask 方案 | 引用它的人 | M8 |
| 4 | OPEN-4 packing 无 attention 隔离 | DLM 会从上一篇尾巴取信息 | **本轮不处理**，但 RESULTS.md 要写明未验 |
| 5 | OPEN-5 Tier 0 吞吐不能外推 | 不能用 Tier 0 数字推正式规模 | 记档 |
| 6 | OPEN-6 rep-4 在 32 token/2 prompt 上分散度极大 | 不能用 rep-4 排名 | SFT 格照报，但注明不可用于排名 |
| 7 | `dlm-3.2m-dllm` 的 `native` 版是否删除 | SPEC 说「取代原 native 版」 | M6 时确认 |

---

## 7. 工作约定

- 每个 milestone 自动 commit（Q7 授权），最终报告列出全部 sha
- **不 push**
- 不改目标。超范围的问题问用户；超参数、命名、重构我自己决定
- **检查因为错误的原因通过时，我会说明是哪个原因**
- 每个 milestone 的决策写进文件，不留在对话里
- GPU 共享：跑之前 `rocm-smi --showuse` 确认卡空；不测吞吐就不受这条约束

### 2026-10-02 21:55 追加的三条

| # | 问题 | 答复 |
|---|---|---|
| 1 | 我做不完时，minimind 那个还做不做？ | **不管它的期限，就让它干**——闸门只管「先不先起它」，不管它几点完 |
| 2 | subagent 怎么用？ | **别管，就放着**——persistent 起步，不去设计它的生命周期 |
| 3 | review 轮要不要注入 bug 验测试？ | **不要** |

### 目标时刻

用户：「**最早明天 10 点前结束**」——即**目标 2026-10-03 10:00 CST 前完成 M1–M9**。
这是目标而非硬期限；晚于它也要做完，但完成后才起 minimind 的 subagent。

---

## 8. 规模（实验版定义即 Tier 0，不另行选择）

`configs/example/tier0.yaml`：hidden 256 · 2 层 · ctx **128** · vocab 6400 ·
batch 8 · 200 步 · 3,213,056 参数 · `tinystories-6400`。

⚠️ **ctx 128 短**——SFT 的对话展平后往往超过 128 token。
**用户已定：超长对话按完整轮次丢弃，不做半句截断**，
避免 assistant span 被切断、让「只算 assistant 段算 loss」这个约定失效。

实现细则（AI 自定，不问）：展平时按**消息**累积 token 数，一轮对话
（1 条 user + 1 条 assistant）整体入块；装不下当前轮就封块开新块。
最后只剩一个没有 assistant 尾巴的块丢掉——它的 loss 会是 0。