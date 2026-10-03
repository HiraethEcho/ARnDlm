# ISSUES — 暂存的问题

**规则**：解不了就记在这里（原因 / 影响 / 回头计划），继续做下一格，收尾时清理。
**状态**：`OPEN` / `DEFERRED`（明确不做）/ `DONE`（已解决，写上怎么解决的）。

## DONE-1b · 实验版已改为复用 151662，两格已重跑

**状态**：已结清（2026-10-02，M6 的全量 smoke 里重跑）。
参数量 598,071,296 → **598,343,680**；eval 4.8013 → **4.8238**、4.8166 → **4.8397**。
**不止 params 变了** —— mask token 不同 → 加噪时替换的 embedding 行不同 → loss 会变。
这一点之前只是推测，现在有数字。

已改的地方：

| 文件 | 改动 |
|---|---|
| `base/qwen3_0p6b/common_qwen.py` | `MASK_TOKEN = "<\|fim_pad\|>"` + `MASK_ID = 151662`；`load_tokenizer()` 不再add token，改成**断言它在词表里且 id 没变**；`build_a2d_lora()` 去掉 `resize_token_embeddings`，改成断言 embedding **行数未变且逐位相同** |
| `tests/test_lora_0p6b.py` | 断言改成 `mask == 151662`、`vocab == 151669`（原来断言 151670）、`embedding rows == 151936` |
| `base/qwen3_0p6b/lora_common.py` · `train_lora_dllm.py` · `base/hf/a2d_dllm.py` | docstring / 注释里的 151669 → 151662 |

`test_lora_0p6b.py` PASS（含手写 loss vs dllm loss 相对差 1.55e-05）。两格已重跑，
RESULTS.md 里现在带一条说明「最后两行已重跑」并给出变化前后的数字。

---

## DONE-1 · Qwen track 的 MASK id：复用 `151662` `<|fim_pad|>`

**决定 (用户拍板)**：复用已有的 `<|fim_pad|>` (151662)，**不新增 token、不 resize
embedding**。`SPEC.md` §3.2 已改。

**原问题**：`SPEC.md` 写「MASK 使用 id 151669」，实测越界——Qwen3-0.6B 的 tokenizer
`vocab_size = 151,669`，合法 id 是 0-151668。

**验证 (2026-10-01)**：

| 检查 | 结果 |
|---|---|
| `vocab_size` | 151,669 → 151669 确实越界 |
| 151662 | `<|fim_pad|>`，special 段真实存在 |
| `tinystories-6400` 中 id ≥ 151000 | **一个都没有** |
| `minimind3-6400` 中 id ≥ 151000 | **一个都没有** |

两个语料都没碰过任何 special token，所以复用不会与语料相撞。

**为什么复用比新增好**：

1. 不动 embedding —— 新增需 `resize_token_embeddings(151670)`，embedding 从
   checkpoint 的 151,936 行缩到 151,670 行，少 266 行。复用则参数量保持精确的
   **598,071,296**，实验版/实践版 对拍基线干净。
2. 对 LoRA 两格的可训练参数**影响为零** —— LoRA 只挂 attention 四个投影，
   backbone 冻结，embedding 本来就不训练。

**⚠️ 注意**：`<|fim_pad|>` 在 Qwen3 自己的预训练语料里可能出现过（Qwen 用了 FIM
格式），所以它**不是随机初始化的一行**。不影响正确性（A2D 只拿它当被遮位置的
标记），但不要当成「空白 token」期待。

**本文件下方代码的遗留**：`example/base/qwen3_0p6b/` 目前仍按「加 `<|diff-mask|>`
+ resize」实现（`common_qwen.py` 的 `MASK_TOKEN`、`lora_common.py` 的注释、
`tests/test_lora_0p6b.py` 的断言）。改不改见下方 OPEN-1b。

---

## OPEN-2 · 实验版 loss 与 dllm loss 的 t 分布不同源

**现象**：手写实现是 `t ~ U[0,1)`（`notes/03` §4），dllm 是
`t = ε + (1-ε)·U[0,1)`，`ε = 1e-3`（`MDLMConfig.time_epsilon`）。

**影响**：`E[1/t]` 相差 **0.1%**（6.91 vs 6.92），对拍时远小于 1% 的阈值；
但**对拍必须让两边抽到同一个 t**，所以 `check_ab c2/c3` 里两边都用
`epsilon=1e-3` 并重新播种。

**回头计划**：正式训练时要不要统一成 dllm 的写法？目前 实验版保持
「手写版 = notes/03 原式」+「对拍时对齐」，两种口径都在代码注释里写明。

---

## DONE-3 · decode 的 `steps` 语义：`--dlm-steps` 与 `--dlm-total-steps` 都已给

**现象**：dllm 的 `steps` 是**全序列**预算，会按 block 数摊薄
（`samplers/mdlm.py:131`：`steps = ceil(steps / num_blocks)`）；
`notes/03` §7 的语义是**每块**步数。同一个 `--dlm-steps 8`：
dllm 跑 8 次 forward，实验版跑 32 次。

**影响**：任何跨实现的解码速度对比，除非换算到同一工作量，否则测的是
「谁配置的步数少」。`check_ab decode-bench` 已经做了换算。

**回头计划**：实践版要不要统一成一个语义？建议保留 实验版口径（每块），
在 CLI 上写明，并在对拍脚本里换算。

---

## OPEN-4 · packing 没有 attention 隔离

**现象**：块由多篇文档首尾相接拼成，没有 document mask，模型能看见上一篇的尾巴。
与 MiniMind 的做法一致，**是刻意的**（test/02 和 实验版模块注释都写明了）。

**影响**：AR 无所谓；DLM 尤其可疑 —— 被遮的位置会从上一篇的尾巴取信息。
行3 的 Qwen track 同样如此。

**回头计划**：正式规模实验前做一次消融 —— 带 document mask vs 不带，
在固定 seed 下比 eval loss。先记着，别在 smoke 阶段改。

---

## DONE-8 · G3 的「尾部 DLM-SFT」已补齐，**G3 已有结论**

**SPEC 内部矛盾**：

| 位置 | 说法 |
|---|---|
| §2.3 pipeline 列 | `ar-sft-mini_a2d` = `pretrain(AR) → SFT → A2D`（**无**尾部 SFT） |
| G3 格子行 | 原路线：`ar-sft` → A2D → **DLM-SFT**（**有**尾部 SFT） |
| G3 注意 | 「**变量：顺序**」 |

**为什么这不是文字问题**。两条路线的当前终点**不可比**：

| 组 | A2D 阶段语料 | eval 测的是 | 实测 eval |
|---|---|---|---|
| 新路线 `{mini,dllm}_a2d-sft`（M4） | SFT 语料 | assistant 段上的 A2D loss | 6.488 – 6.528 |
| 原路线 `sft-{mini,dllm}_a2d`（M5） | **预训练**语料 | 全 token 上的 A2D loss | 5.028 – 5.068 |

两边 eval 的 token 集合不同（assistant 段 ~62/128 vs 全部 128），**差值毫无意义**。
5.03 比 6.50 好这个印象是假的。

**G3 的验收是「两条线的『能 chat』四条门槛都能报；差值就是顺序的代价」**，
而「能 chat」只能在 **SFT 语料** 上评。所以原路线**必须**再跑一次 SFT
（DLM-SFT），两条路线才落在同一个可比点上。

**两个选项**：

| | 做法 | 代价 |
|---|---|---|
| **O1** | 补 4 格尾部 DLM-SFT（`ar-3.2m-{hf,native}-sft-{mini,dllm}_a2d` 再 SFT 一次），SPEC §2.3 加这 4 格 | 4 × 200 步，Tier 0 约 2 分钟 |
| **O2** | 不补，G3 标记为「pipeline 跑通但结论待定」 | 矩阵里 4 个 ⬜ 变 ✅ 但 G3 无结论 |

**已按 O1 结清**（2026-10-02）：`train_dlm_sft.py` 的 `CELLS` 从 4 格扩到 8 格，
新增 `ar-3.2m-{hf,native}-sft-{mini,dllm}_a2d-sft` 四格，上游分别是 M5 的四格。

**结论**（两条路线终止于同一评测：assistant 段上的 A2D loss，同语料同目标）：

| 路线 | hf/mini | hf/dllm | native/mini | native/dllm |
|---|---:|---:|---:|---:|
| 新路线 A2D → SFT | 6.4958 | 6.4880 | 6.5279 | 6.5088 |
| 原路线 SFT → A2D → SFT | **6.1029** | **6.0851** | **6.0851** | **6.0485** |

**原路线优约 0.40 nat，四对全部同向。** 结论写进 `SPEC.md` G3。

## DEFERRED-10 · DLM 行不做 DPO（用户 2026-10-02 决定放弃）

**为什么 DLM 行做不了**：DPO 的目标是 `log pi(y|x)`（自回归因子的乘积）的差。
掩码扩散模型**同时对所有位置去噪**，不定义这个量。SPEC 自己的矩阵就写了
「DLM 无序列 logprob」。要做得先有一个 ELBO 形式化的 `log pi(y|x)`，
**那是研究问题，不是配置开关**。

**用户决定：放弃**。所以 `dlm-3.2m-{hf,dllm}-sft-dpo` 与 `qwen3-0.6b-*-sft-dpo`
**不做**，不是待办。AR 行两格已跑通（偏好准确率见 RESULTS.md）。

**历史记录**：

## OPEN-9 · loss 工厂的裸名 lambda（本次会话犯两次）

`lambda: minimind_loss` 求值得到**函数对象**，于是 Loop 把工厂当 loss 调
`factory(model, batch)` 拿到闭包，报 `cannot unpack non-iterable function
object`。本次在两个不同文件里各犯一次。

正确形状是 `lambda m: minimind_loss(m)`。`train_a2d_from_sft.py` 的
`build_parts` docstring 记了这条。

## DONE-7 · 方案 A 与 ctx 128 的冲突：用户接受 3% 存活率

**发现**：2026-10-02 写 SFT 展平器时实测。

用户拍板的 Q4 是「方案 A：把 `reasoning_content` 拼进 assistant 文本并**监督**它」。
但 `configs/example/tier0.yaml` 的 `context_length: 128`，而带 think 的消息中位长度
**276 token**：

| 采样 5,000 条对话 | 值 |
|---|---:|
| 消息总数 | 36,977 |
| 带 `reasoning_content` 的消息 | 7,100（19.2%） |
| 消息 token 长度中位 | 31 |
| 带 think 的消息长度中位 | **276** |
| ctx 128 装不下的消息 | 7,734（20.9%） |
| 其中带 think 的 | 6,890（全部消息的 18.6%） |

**后果**：按整轮丢弃（GOAL.md §8），带 think 的消息几乎**全部**被丢掉。
**方案 A 的收益为零，代价是丢掉 19% 的 assistant 消息。**

**代码现在跑在 A2**（保持 ctx 128，接受丢弃），丢弃量写在
`data/prepared/sft-mini-6400/meta.json` 的 `n_dropped_turns_with_thinking`
（4,096 块切片里 3,164 / 3,420），不是静默的。

**三个选项，等拍板**：

| | 做法 | 后果 |
|---|---|---|
| **A1** | SFT 格用 `context_length: 256` 或 384 | think 大部分能装下。**代价：Tier 0 的 ctx 是已完成的 10 格配置，改它让那 10 格的数字全部作废**（它们是 ctx 128） |
| **A2** | 保持 ctx 128，接受丢弃 | 数字干净，但 GOAL.md §4.3 的 A 变成空头支票 |
| **A3** | 装不下时只丢 think、保留答案 | 两种口径混在一份语料里，最难解释 |

**影响范围**：13 个 SFT 格的**数据内容**，不阻断代码。

---

## OPEN-6 · rep-4 在 32 token / 2 prompt 上分散度极大

同一份权重、同一套解码参数，行2 的 A2D 是 **0.4655**，行1 的 A2D 是 **0.7586** ——
换一个 base 就差 0.29，而两者的配方只差模型实现。

**影响**：任何用 rep-4 给 DLM 实现排名的做法都是错的。smoke 阶段只要求「报了」。

**回头计划**：正式评测用几百 token × ≥5 个 prompt，再谈比较（teach/11 §11.4）。

---

## OPEN-5 · Tier 0 的吞吐不能外推到正式规模

**现象**：原生 torch 模型在 Tier 0 比 transformers **快 5%**
（47.5k vs 45.2k tok/s）。小模型时 transformers 的 python 开销
大于其融合算子的收益。

**影响**：任何「手写一定更慢」的判断在这里都是错的。正式规模（50M+、ctx 512）
结论可能反过来。

**回头计划**：`teach/13` 阶段重测 Tier 1 阶梯。

---

## DEFERRED-1 · dllm 包的 `trl` / `lm_eval` 依赖

`import dllm` 会拉进整个 `pipelines/`（含 RL 和评测），`.venv` 从 46 包涨到 108 包。
本项目不用它们，但删不掉（`dllm/__init__.py` 顶层 import）。
已决定不动（`scripts/install_dllm.sh` 回钉了 datasets 4.2.0）。

## DEFERRED-2 · BD3LM 没有 实验版

`dllm/core/trainers/bd3lm.py` 和 `samplers/bd3lm.py` 都在，本项目矩阵里没有这一列。
不实现。