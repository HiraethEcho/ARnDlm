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
| 新路线 A2D → SFT | 6.3872 | 6.3845 | 6.4133 | 6.4125 |
| 原路线 SFT → A2D → SFT | **5.9773** | **5.9776** | **5.9713** | **5.9709** |

**原路线优 0.4251 nat，四对全部同向。** 结论写进 `SPEC.md` G3。

⚠️ 这组数字是 **2026-10-03 重跑后的**：修完 DONE-13（8 格训成因果模型）之前
这里是 6.4880–6.5279 / 6.0485–6.1029，差 0.40 nat。结论方向没变，
但那批数字里 dllm 那一臂量的是因果模型。

## DEFERRED-10 · DLM 行不做 DPO（用户 2026-10-02 决定放弃）

**为什么 DLM 行做不了**：DPO 的目标是 `log pi(y|x)`（自回归因子的乘积）的差。
掩码扩散模型**同时对所有位置去噪**，不定义这个量。SPEC 自己的矩阵就写了
「DLM 无序列 logprob」。要做得先有一个 ELBO 形式化的 `log pi(y|x)`，
**那是研究问题，不是配置开关**。

**用户决定：放弃**。所以 `dlm-3.2m-{hf,dllm}-sft-dpo` 与 `qwen3-0.6b-*-sft-dpo`
**不做**，不是待办。AR 行两格已跑通（偏好准确率见 RESULTS.md）。

**历史记录**：

## DONE-11 · 续训的 RNG 快照取错了位置，C4 又恰好测不到

**这个是本轮 review 抓到的最严重的一个。**

`common/loop.py::fit` 把「训练结束时的 RNG 状态」的快照写在了 **while 循环之前**：

```python
self._rng_after_train = rng_state()     # 循环之前 —— 取到的是训练【开始】时的状态
while self.step < limit: ...            # 训练在这里消耗 RNG
```

`save()` 于是把训练**开始前**的 RNG 存进 checkpoint。

**后果**：任何用 `--resume` 的 **DLM** 格，续训时会从头重放同一串掩码，
而不是接着上次那串。`ar-3.2m-*-sft-*a2d` 这一类格的续训**静默地训在错误的噪声序列上**。
实测 `worst_step_delta = 5.2e-01`（对照 1e-6 的容差）。

**为什么没被发现**：C4 就是为「续训 == 一次跑完」设的验收，但
**C4 用的是 `ar_loss`，而 `ar_loss` 根本不消耗 RNG**。
代码注释还写着「for a DLM cell the loss draws from it too — which is exactly
the equivalence C4 checks」——**这句话描述的检查并不存在**。

**这就是「错的测试比错的代码更危险」的实例**：注释给了假的安心感，
而不是代码缺一个特判。

**已修**：
1. 快照移到 while 循环**之后**、eval 与采样之前（两者都消耗 RNG）。
2. C4 拆成两个臂：`ar_loss` 臂（测 batch 流 + optimizer）和
   新增的 `_rng_loss` 臂（随机掩码，测 RNG 快照）。
   **新臂确实会失败**——把修复回退掉，它报 `rng_worst_step_delta=2.356e-01` FAIL；
   恢复后 `0.000e+00` PASS。同一时刻 `ar_loss` 臂在两种情况下都是 `0.000e+00`，
   证明它从来没覆盖过这一项。

**教训**：验收项的注释不能声称覆盖它实际没覆盖的东西。
要问「这个检查真的会在 bug 存在时失败吗」，而不只是「它现在是 PASS 吗」。

## DONE-12 · SFT 三处静默偏差：span 不含 EOS、采样 prompt 差一个 token、docstring 与代码不符

一轮 review 抓到三条，**全部不报错、全部让 loss 曲线看起来正常**。

### 12a · assistant 后的 EOS 不在监督 span 内（最严重）

`_pack` 先 `cur_spans.append((start, len(cur)))` 再 `cur.append(eos)`，
所以 EOS 落在半开区间**之外**。EOS 是唯一教模型「说完了」的那一票。

实测：SFT 后的模型生成 64 token，**零个 EOS**，每次跑满。
pretrain 的 `ar_loss` 注释明说 EOS 位置的 loss 教模型停止，SFT 却没给这个信号。

**已修**：span 结束位置移到 `cur.append(eos)` 之后。
监督 token 255,451 → **260,590**（+5,139 = span 数，每个一个 EOS）。
修完生成 4 / 19 token 就在 EOS 停下。

### 12b · 采样 prompt 与训练语料差一个 token

语料里 turn 用 **EOS** 分隔，所以 prompt 前缀是
`[user\n]{question}[EOS][assistant\n]`。
而采样写的是 f-string `f"user\n{p}\nassistant\n"`——**换行**占了 EOS 的位置。

实测 `"1+1等于几"`：两种写法**都是 13 个 token，内容不同**。
不报错，但每个样本文本和每个 rep-4 数字都在描述一个模型没训过的前缀。

`sample_sft` 的 docstring 还写着「with the chat shape the SFT rows actually
trained on」——**这句话是错的**。

**已修**：`common/sft.py::encode_prompt` 在 token 级拼接，5 个调用点统一走它；
两个 sampler 加 `prompt_ids` 通道（f-string 表达不了中间的 EOS）。
已对真实语料验证逐位一致：`训练前缀 == encode_prompt(q)` → True。
Qwen track 另有 `lora_common.sft_prompts_for`（不同 tokenizer track）。

**附带**：`sample_ar` 返回的 `"prompts"` 标签用的是默认 AR prompts，
所以 repetition 表把 SFT 样本标成了 `'Once upon a time'`。一并修。

### 12c · 模块 docstring 的块结构图与代码不符

图里写 `[BOS] user\n{q}\n[EOS] assistant\n{a}\n[EOS]`，
实际是 `[BOS] user\n{q}[EOS] assistant\n{a}[EOS] user\n{q2}[EOS]`——
**每条 turn 前没有 BOS**，EOS 前也没有换行。
图是下游写 prompt 的人的唯一步规格，12b 就是从它来的。

顺带：`Turn.text()` 带尾部换行而 `_pack` 不发这个换行，
`n_long_turns` 用 `len(enc(t.text())) + 2` 统计，与 `_pack` 的判据差一个 token。
改为直接用 `len(enc(header)) + len(enc(body)) + 1`。
`n_long_turns` 3,420 → **3,398**（22 个恰好卡边界的 turn 之前被误报）。

## DONE-13 · `*-dllm_a2d` **八格**训的是因果模型，不是 A2D

`train_a2d_from_sft.py::build_parts` 和 `train_dlm_sft.py::_hf_parts/_native_parts`
对**两个 objective 都**返回 `build_model`（HF）/ `build_native_model`（native）。
两个脚本是同一个 bug 的两份拷贝——先修第一份时，第二份还在。

但 dllm 的 objective 走 `MDLMTrainer.compute_loss`，那个函数
**forward 时不传 `attention_mask`** —— transformers 于是按因果建 mask。
实测：改末位 token，前面所有 logits 位位不变（delta `0.0e+00`）= 因果；
换成 `build_a2d_model` 则是 `8.3e-01` = 双向。

**后果：8 个格训的是因果模型，却叫 A2D**：

| 格 | 脚本 |
|---|---|
| `ar-3.2m-{hf,native}-sft-dllm_a2d` | `train_a2d_from_sft.py` |
| `ar-3.2m-{hf,native}-sft-dllm_a2d-sft` | `train_dlm_sft.py` |
| `dlm-3.2m-dllm-sft` | `train_dlm_sft.py` |

于是 G3 的那个结论在 dllm 那一臂上比较的是
**双向模型 vs 因果模型**，不是两个实现；G5 的对拍也失去意义。

`minimind` 那一臂没事——它在 forward 时显式传 4D 全零掩码，不依赖 builder。
四种组合现在的实测：

| row | objective | 掩码来源 |
|---|---|---|
| hf | dllm | 模型自带双向（`A2DQwen3LMHeadModel`） |
| hf | minimind | 显式 4D 全零掩码 |
| native | dllm | `set_causal(False)` |
| native | minimind | forward 的 `is_causal=False` 参数 |

**已修**：两个脚本都按 objective 选 builder（HF 用 `build_a2d_model`，
native 用 `set_causal(False)` 包装）。A2D 模型与普通模型的 state_dict 键完全一致，
所以 `load_into` 不用改。

**另外加了一道会响的**：`native/a2d_dllm.py::make_dllm_loss` 现在检查
`model._last_is_causal` 并直接抛错。静默接受一个因果模型，就是这 8 格发生的方式。

**修完暴露出第三份拷贝：采样器。** `train_dlm_sft.py` 对两个 objective 都用
AR 采样器，而 AR 采样器走 `generate`，那假设因果解码。以前能用**只是因为模型
本来就是因果的**；换成真的 A2D 模型立刻炸：

```
RuntimeError: The expanded size of the tensor (1) must match the existing
size (22) at non-singleton dimension 2.  Target sizes: [1, 2, 1, 22]
```

`train_a2d_from_sft.py` 一直按 objective 选了 sampler，所以只有 `train_dlm_sft.py`
漏了。HF 臂改用 `sample_dllm`（dllm 的 MDLMSampler，额外加了 `prompt_labels`
——SFT 的 prompt 是 id 列表，不加的话 repetition 表的「prompt」列会打印裸 id）；
native 臂改用 `sample_native_dlm`，它通过 `native_forward` 取 logits，
所以与实现无关（dllm 在这一格提供的是 objective，不是 sampler）。

**这个 bug 只有在修好上一个之后才存在。** 因果模型让错误的采样器看起来是对的，
两个错误互相掩护。

**教训**：`train_a2d_dllm.py`（G2 的格）一直用的是 `build_a2d_model`。
三个脚本长得几乎一样，只有这一行不同——而**修复也必须做三遍**，
因为修一处不会让另两处自动正确。下一步应该是把 `(row, objective) -> builder`
收进一个共享函数。

## DONE-14 · 静态检查：引用漂移、死参数、重复 import、一个 shell label

**这轮只做静态分析，没有跑任何测试或训练。**

### 14a · 五处 ISSUES 引用指向已改名的标题

代码里 `ISSUES.md OPEN-7` / `OPEN-8` / `OPEN-10` 共 5 处，但那些标题已经改成了
`DONE-7` / `DONE-8` / `DEFERRED-10`。照着注释去找会找不到条目。

**已修**：全部改指现在的标题。核对脚本（逐个引用对照 ISSUES 的标题集合）：
剩余漂移 0。

### 14b · 死参数：接受、忽略、没人检查

| | 问题 | 改法 |
|---|---|---|
| `ckpt.load_into` 的 `atol` | 声明了、没用。**调用方调它以为在改什么** | 删。docstring 写明原因 |
| `dpo.sequence_logprob` 的 `ids` | 只用 logits 和 labels | 删。3 个调用点同步 |
| `check_ab.c5` 的 `args` | 唯一未用 args 的 check | 保留（`CHECKS[name](args)` 统一签名），docstring 说明 |
| `hf/direct_dlm.loss_fn` 的 `model` | Loop 的 `loss_fn(model, batch)` 契约 | 保留，不是 bug |
| `train_dpo.make_dpo_loss` 的 `policy` | Loop 已把 model 当 policy 传入，重复 | 删，docstring 说明 |

**这一类最值得记**：一个接受但忽略、没人检查的参数，等于**邀请调用方传错的
tensor 并保持沉默**。它比抛错更危险，因为它让错误调用看起来正常。

### 14c · 未使用的 import（3 处真冗余）

`train_dlm_sft.py` 的 `sample_dlm_row`（改用 dllm 采样器后遗留）、
`train_dpo.py` 的 `copy`、`common/dpo.py` 的 `encode`。

### 14d · 函数内重复 import（4 处）

`train_dlm_sft.py` 在模块级和函数体内各 `from common.sft import encode_prompt`
一次（×3 处，模块级已有）；`train_dpo.py` / `check_ab.py`（×4 处）/ 
`test_sft_flatten.py` 里同样有模块级已有的延迟 import。
已删函数内冗余那份，**延迟 import 本身保留**（那是避免循环依赖的正统写法）。

### 14e · `BASE_OF`：三个脚本写同一字符串两次

`"Qwen-mini-HF" if row == "hf" else "Qwen-mini-MiniMind"` 在 3 个脚本里出现，
`config.py` 里已有 `BASE_OF`。收进 `common/config.py::BASE_OF`，3 处改用它。
一处曾漂移（DONE-13 之前 dllm 臂标签错），统一后不会再漂。

### 14f · `smoke_all.sh` 的 cell 9/10 label 重复了 `qwen3-0.6b-`

`"qwen3-0.6b-qwen3-0.6b-dllm_a2d"` → `"qwen3-0.6b-dllm_a2d"`。
只影响日志显示，不影响行为。

### 14g · `check_ab.py` 里 `6.5408 vs 6.5252` 的实测注释已过时

换成 `6.4502 vs 6.4344`（本轮重跑后的数字）。

### 14h · `sample_dllm` 里局部变量 `prompt_ids` 类型漂移

行 136 赋值为 `list[Tensor]`，行 147 同名赋值 `list[int] | str`。
**没坏，但同一个名字两种类型**，读的人得回头确认哪次是哪次。
改名后者为 `prefix`。

### 14i · Qwen track 的 EOS 注释写错

`common_qwen.py` 写「its EOS is `endoftext>` = 151643」，实际 `eos_token_id`
是 **151645**（`<|im_end|>`）。151643 也在词表里，所以不报错。
数据与采样都用的是 151645（对），只是注释写错。

### 一个反面教训

本轮差点把 `loss_windows` 的 `k = min(k, len//2 or 1)` 标成「头尾窗口重叠」的
bug——因为我凭脑内推演 `losses[:k]` 和 `losses[-k:]` 在 n=2k 时重叠。
**实测**（严格递增序列）显示它们各占一半，不重叠，注释是对的。
静态分析最容易犯的错：不跑数就下「正确/错误」的结论。

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

**代码跑在 A2**（保持 ctx 128，接受丢弃）。

**用户 2026-10-02 拍板**：实验版按 A2 接受这个代价——原话是
「实验版用最小代价，能跑就行，不关心效果和训练量」。

**已落实**：`meta.json` 不再写 `reasoning_in_loss: true`（那是假的——95% 的 think
根本不在语料里），改成两个从**落盘的块里数出来**的字段：

```
"reasoning_option": "A (fold <thinking> into the assistant text)",
"reasoning_survival_rate": 0.0551
```

配套计数（4,096 块切片）：`n_dropped_turns_with_thinking 3,158` / `n_long_turns 3,398`。

**给实践版的建议（未执行，`configs/practical/tier1.yaml` 仍是空的，现在改零成本）**：
`context_length` 与参数量**无关**，放大参数解决不了这个问题——它是独立旋钮，
而且参数量越大越慢。正确做法是给 SFT 单独一个更长的 context。

**没做 A1 的代价要说清楚**：Tier 0 的 ctx 128 是已完成的 20+ 格的配置，
改它会让那些格子的数字全部作废。实验版不值得。

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