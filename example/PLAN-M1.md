# M1 · SFT 展平器　`example/common/sft.py`　status: DONE

## 1. correct

先写 `example/tests/test_sft_flatten.py`（8 节，40+ 断言），跑出来是
`ModuleNotFoundError`，然后才写实现。断言全部对着**解码后的文本**写，不对着索引写——
span 偏一个 token，块照样解出通顺文本、loss 照样是个 loss，没有一个症状像 bug。

测试抓到的三个真 bug（都在实现里，都是「看起来对」的那类）：

| # | bug | 症状 | 为什么测试能抓到 |
|---|---|---|---|
| 1 | span 偏移用 `len(encode(整块)) - len(encode(body))` | BPE 在 header/body 接缝处改变切分，差值不等于 header 长度，**span 偏 1-2 token** | 「每个 assistant 答案都在某个 span 里」失败 |
| 2 | 拟合检查 `len(cur) - 1 + need > block_size` | 块长到 **129 = block_size + 1**；`np.asarray` 在真实语料上炸 inhomogeneous | 只有真实语料才凑满块；补的断言是「块长 ≤ block_size」 |
| 3 | span 存成块内 `(a, b)`，`load_sft` 却按全局偏移 `divmod(a, block_size)` 解 | **所有 span 落进第 0 块**，整个块连 PAD 一起被监督 | 落盘往返测试（写之前没有这一节）|

第 3 个是最阴的：内存路径全对，落盘路径全错。已把 span 格式改成
`(block, start, end)` 三列，块号在文件里，歧义从根上不存在。

## 2. quality

初版写了三套重叠的编码逻辑（`spans_from_turns` / `flatten_dialogue` /
`_dialogue_blocks`），其中一行 `ids[-len(ids)+start:] = ids[start:]` 是彻底的死代码。
收成一条：`_turn_ids` 编码 header 和 body 各一次、分别追加，`_pack` 只管分块。
`non-atomic` 分支的条件原本写成 `room > body_len`（用 body 长度判断截断），
对 user 轮次永远不成立，整条分支是死的——改成按 `room` 截断并记录裁剪后的 span。

`mean_block_fill` 一开始算的是「每块 span 个数 / block_size」= 0.0098，
一个荒谬的数字。修正为实际填充长度 / block_size = **0.72**，并把 `mean_used_tokens`
一起写进 meta。

## 3. iterate

| 轮 | 改了什么 |
|---|---|
| pass 1 | 三个实现 bug（见上表） |
| pass 2 | 三套逻辑合一，删死代码，修 non-atomic 分支，修 fill 指标 |
| pass 3 | 测试自身也有 6 处错断言：把 14 轮对话拿去核对第一个块、`labels[1] == BOS`、用 token id 当行号索引、`batch` 形状按 `[B,2,L]` 取、padding 检查看错了张量。**测试写错和实现写错一样要改**——否则 PASS 是假的 |

## 4. record

`example/ISSUES.md` **OPEN-7**：`reasoning_content` 与 ctx 128 冲突。
`example/common/sft.py` 的模块 docstring 记录了 option A 的代价。

## measured

`data/prepared/sft-mini-6400/`（4096 块 · 2166 条对话 · 351 条整条丢弃）：

| 指标 | 值 |
|---|---:|
| 块填充率 | 0.72（均 92.2 / 128 token） |
| 监督 token | 255,451（每块均 62.4 = **48.7%**） |
| span 数 | 5,139 |
| **带 think 但装不进 128 的轮次** | **3,164 / 3,420 = 92.5%** |

## OPEN-7 · 方案 A 在 ctx 128 下收益为零

用户 2026-10-02 定：Q4 选**方案 A**（拼进 assistant 文本并监督 think）。
实测这个组合自相矛盾：

| 采样 5,000 条对话 | 值 |
|---|---:|
| 消息总数 | 36,977 |
| 带 `reasoning_content` 的消息 | 7,100（**19.2%**） |
| 消息 token 长度中位 | 31 |
| **带 think 的消息长度中位** | **276** |
| ctx 128 装不下的消息 | 7,734（20.9%） |
| 其中带 think 的 | 6,890 = **全部消息的 18.6%** |

**带 think 的消息中位 276 token，一个整块只有 128。** 按整轮丢弃（GOAL.md §8
的决定）之后，几乎每一条 think 消息都被丢掉——**方案 A 的收益是零，代价是
丢掉 19% 的 assistant 消息**。

`mean_block_fill 0.72` 也说明这点：整轮丢弃让 28% 的块是 padding。

**三个选项**：

| | 做法 | 后果 |
|---|---|---|
| **A1** | SFT 格用 `context_length: 256` 或 384 | think 大部分能装下。**代价：Tier 0 的 ctx 是 10 格已完成的配置，改它会让那 10 格的数字全部作废**（它们是 ctx 128） |
| **A2** | 保持 ctx 128，接受丢弃 | 数字干净，SFT loss 只反映「无 think 的对话」。GOAL.md §4.3 的 A 变成空头支票 |
| **A3** | 装不下时只丢 think、保留答案 | 语义变成「部分消息按 A、装不下的按 C」，两种口径混在一份语料里，最难解释 |

代码现在跑在 A2 上（丢弃量是 meta.json 里可见的数字），等决定。

## commits

见 `git log --oneline -- example/common/sft.py`
