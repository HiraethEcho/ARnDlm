# ISSUES — 暂存的问题

**规则**：解不了就记在这里（原因 / 影响 / 回头计划），继续做下一格，收尾时清理。
**状态**：`OPEN` / `DEFERRED`（明确不做）/ `DONE`（已解决，写上怎么解决的）。

---

## OPEN-1 · SPEC 3.2 的 MASK id 151669 在 Qwen3-0.6B 的 tokenizer 里不存在

**现象**：该 tokenizer 的最大 id 是 **151668**，151669 无效。
A 版按「新加一个 token」处理：加 `<|diff-mask|>` 后它正好拿到 151669。

**附带影响**：`resize_token_embeddings(151670)` 同时把 embedding 从
checkpoint 的 **151936 行**缩到 151670 行，掉了 266 行 = **272,384 参数**。
这 266 行 tokenizer 永远吐不出来，所以不影响正确性，但参数量对不上时
很容易被当成 bug 查半天。

**影响**：行3 两格（cells 9/10）。

**回头计划**：B 版写 `training/a2d_dllm/` 时二选一，并写进注释：
1. 沿用 A 版（加 token + resize），接受参数量变化；
2. 复用已有的未使用 special token（如 `<|fim_pad|>` 151662），**不改 embedding**。
   更干净，但要确认该 token 在 Qwen3-0.6B 的训练数据里从未出现。

**未决**：SPEC 该改成「先加 token 再用 151669」，还是「复用 151662」。

---

## OPEN-2 · A 版 loss 与 dllm loss 的 t 分布不同源

**现象**：手写实现是 `t ~ U[0,1)`（`notes/03` §4），dllm 是
`t = ε + (1-ε)·U[0,1)`，`ε = 1e-3`（`MDLMConfig.time_epsilon`）。

**影响**：`E[1/t]` 相差 **0.1%**（6.91 vs 6.92），对拍时远小于 1% 的阈值；
但**对拍必须让两边抽到同一个 t**，所以 `check_ab c2/c3` 里两边都用
`epsilon=1e-3` 并重新播种。

**回头计划**：正式训练时要不要统一成 dllm 的写法？目前 A 版保持
「手写版 = notes/03 原式」+「对拍时对齐」，两种口径都在代码注释里写明。

---

## OPEN-3 · decode 的 `steps` 在两套实现里含义不同

**现象**：dllm 的 `steps` 是**全序列**预算，会按 block 数摊薄
（`samplers/mdlm.py:131`：`steps = ceil(steps / num_blocks)`）；
`notes/03` §7 的语义是**每块**步数。同一个 `--dlm-steps 8`：
dllm 跑 8 次 forward，A 版跑 32 次。

**影响**：任何跨实现的解码速度对比，除非换算到同一工作量，否则测的是
「谁配置的步数少」。`check_ab decode-bench` 已经做了换算。

**回头计划**：B 版要不要统一成一个语义？建议保留 A 版口径（每块），
在 CLI 上写明，并在对拍脚本里换算。

---

## OPEN-4 · packing 没有 attention 隔离

**现象**：块由多篇文档首尾相接拼成，没有 document mask，模型能看见上一篇的尾巴。
与 MiniMind 的做法一致，**是刻意的**（test/02 和 A 版模块注释都写明了）。

**影响**：AR 无所谓；DLM 尤其可疑 —— 被遮的位置会从上一篇的尾巴取信息。
行3 的 Qwen track 同样如此。

**回头计划**：正式规模实验前做一次消融 —— 带 document mask vs 不带，
在固定 seed 下比 eval loss。先记着，别在 smoke 阶段改。

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

## DEFERRED-2 · BD3LM 没有 A 版

`dllm/core/trainers/bd3lm.py` 和 `samplers/bd3lm.py` 都在，本项目矩阵里没有这一列。
不实现。