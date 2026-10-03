# Step B1 — SFT 数据：从对话 JSONL 到「哪些 token 该被学」

**你要写的**：`data/sft/`（展平器）+ 落盘格式
**实验版参考**：`example/common/sft.py::build_sft_blocks` / `write_prepared`，`example/base/hf/make_sft_data.py`
**验收**：你落盘的 `assistant_spans.npy` 与实验版**逐个 span 相同**。

这是整条 SFT 路径上**最容易写错、而且错了 loss 曲线照样好看**的一步。
`ar-3.2m-hf-sft` 和 `ar-3.2m-native-sft` 的 loss 逐位相同（C7），
说明这一层只要对了，后面两个 base 就不可能不一致。

---

## B1.1 输入长什么样

`data/raw/minimind_dataset/sft_t2t_mini.jsonl`，每行一个对话：

```json
{"conversations": [
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "<think>...</think>答案..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

`user` = **prompt**（不被学），`assistant` = **completion**（被学）。
只有这两两个 role。别的 role 一律 `ValueError`——不要静默跳过，
静默跳过会让你以为格式没问题。

## B1.2 三种落盘产物

```
tokens.bin           uint16  [n_blocks, block_size]   展平后的 token
assistant_spans.npy  int64   [n_spans, 3]            (block, start, end)
meta.json            {n_blocks, supervised_tokens, ...}
```

**span 为什么是 3 元组而不是 2 元组。** 早期版本存 `(start, end)`，
loader 靠 `start // block_size` 反推块号。`start` 恰好是块边界时，
反推出的块号会错一位，于是**所有 span 都落在 block 0，padding 全被当成监督信号**
（loss 当然还在降）。直接带上块号，这种错就写不出来了。

`start` 含、`end` 不含 —— `[start, end)`。

## B1.3 展平：一个块里塞多条对话

```
[BOS] user\n{q1}[EOS] assistant\n{a1}[EOS] user\n{q2}[EOS] assistant\n{a2}[EOS] ...
       ^^^^^^^^^^^^^^^^^^^^ prompt      ^^^^^^^^^^^^^^^^^^ 监督区间（含 EOS）
```

三条容易搞反的：

1. **一个块只有一个 BOS，在最前面。** 每条 turn 之间**只有 EOS，没有 BOS**。
2. **turn 之间用 EOS 分隔，不是换行。** 所以采样时的 prompt 也必须用 EOS
   （见 B2.7），写成 `user\n{q}\nassistant\n` 就不对。
3. **assistant 后的 EOS 在监督区间内。** 它是唯一教模型「说完了」的那一票。
   把它排除在 span 外，模型就永远学不会停——实测生成 64 token、零个 EOS。

一个块装多条 → 每块都有足够监督 token。单条对话独占一块时，块尾的 PAD 占比很高。

## B1.4 装不下就整条丢弃

对话总长 > `block_size` 时**整条丢**，不做截断。

为什么不截断：截断会把 assistant 的后半句切掉，而 assistant span 的 `end`
正好在切点附近。**一个被切断的 span 会教模型在半句话中间停。**

代价是会丢数据。实验版 4,096 块切片里丢了 351 条对话 / 3,420 个 turn。
`meta.json` 记了 `n_dropped_dialogues` 和 `n_long_turns`，不要静默丢。

## B1.5 think 块的处理

MiniMind 的 assistant 内容里混着 `<think>...</think>`。实验版选了
**方案 A**：把它折进 assistant 文本，一起监督。

`reasoning_content` 字段（有的数据带）是另一种表示，本项目不用。

**注意 ctx 128 下的实际效果**：think 消息中位 276 token > 128，
所以绝大多数 think 被 B1.4 整条丢掉了，实测存活率 **5.5%**
（写在 `meta.json` 的 `reasoning_survival_rate` 里）。
你如果给 SFT 用更长的 context，存活率会大幅上升——
见 `example/ISSUES.md` DONE-7 的实测表。

## B1.6 块内 padding 用 IGNORE

padding 位置的 label 填 `IGNORE = -100`，不是 0。

- `0` 是一个真实 token（MiniMind 的词表里有），填 0 等于教模型预测 token id 0。
- `-100` 是 `F.cross_entropy(ignore_index=...)` 的默认忽略值，
  这一个数同时让 loss 忽略它、梯度也不回传。

## B1.7 坑

| 坑 | 症状 |
|---|---|
| span 存 2 元组靠反推块号 | 块边界处错位，所有 span 落进 block 0，padding 被当监督 |
| **span 停在 body 末、不含 EOS** | 模型不会停。loss 曲线完全正常，采样每次跑满 max_new_tokens |
| **采样 prompt 用 `\n` 而非 EOS** | 前缀与训练分布不符，样本文本和 rep-4 全部没有意义 |
| 每条 turn 之间插 BOS | 语料与文档描述不符，且与 pretrain 的 packing 不一致 |
| 判定「装不下」时多算一个换行 | `n_long_turns` 虚高，恰好卡在边界上的 turn 被误报 |
| padding 的 label 填 0 而不是 -100 | loss 里混进「预测 token 0」的任务，曲线照样下降 |
| span 的 `end` 取闭区间 | 最后一个 token 不被监督，少一票 EOS 的 loss |
| 装不下时截断而不是丢弃 | assistant span 被切断，模型学会在半句话中间停 |
| 对话之间不插 BOS | 上一条的 EOS 和下一条的首 token 相邻，模型学会跨文档续写 |
| 静默跳过未知 role | 格式坏了但流程跑通，最后只表现为「数据变少」 |
| 用 `view` 改 `spans` 的形状 | 非连续张量会抛 `RuntimeError: view size is not compatible` |

## B1.8 验收命令

```sh
# 实验版跑一遍拿基准
.venv/bin/python example/base/hf/make_sft_data.py --max-blocks 4096

# 4,096 块切片上应该看到：
#   n_blocks 4096 · mean_block_fill 0.72 · supervised_tokens 260,590
#   n_spans 5139  (= supervised_tokens - mean_supervised_per_block * 4096,
#                  每个 span 末尾那个 EOS 是其中一票)
#   reasoning_survival_rate 0.055

# 对拍：把你落盘的 spans 和实验版逐个比
.venv/bin/python example/tests/test_sft_flatten.py
```

对应实践版目录：`data/sft/`。**SFT 的 AR objective 和 DLM objective 共用同一份数据**——
见 `AGENTS.md` 的「SFT 阶段两个 base 共用同一份代码」。