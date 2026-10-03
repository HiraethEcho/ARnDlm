# ARnDlm SPEC

## 1. 目标与两个版本

在一致的 tokenizer、数据集和测量口径下，比较 Qwen-style Transformer 的 **AR** 与 **DLM**，
并把两者都训到**能对话**的程度。项目用于学习、复现和可测量对比，不追求 SOTA。

本节只定义两件事：**实验版 / 实践版怎么分**，和**模型怎么命名**。
具体要对比什么，在 §2。

### 1.1 实验版 / 实践版（同一件事写两遍）

| | **实验版** | **实践版** |
|---|---|---|
| 谁写 | AI | 你 |
| 位置 | `example/` | 根目录 `models/` `training/` `diffusion/` `evaluation/` `utils/` `data/` |
| 模型规模 | **Tier 0**：3.21M · ctx 128 · 2 层 | **Tier 1（待测）**：~50M · ctx 512 · 8 层 |
| 语料 | tinystories-6400 切片（~20 万 token） | minimind3-6400（332M token） |
| 步数 | 200 | 按吞吐阶梯定 |
| 目的 | **pipeline 能不能跑通 + 代码对不对** | 正式实验，拿可引用的数字 |
| 能力 | **不看**（loss 与 rep-4 只记录） | 看 |
| 关系 | **实验版是实践版的正确性基准** | 实践版输出必须与实验版对齐 |

**实验版跑通 ≠ 实践版可以写。** 写每一步之前，先用 `example/teach/<step>.md` 里的
「验收命令」跑一遍实验版，拿到基准数。

### 1.1.1 两个版本在哪里区分

| 维度 | 实验版 | 实践版 | 规则 |
|---|---|---|---|
| **代码** | `example/` | 根目录 `models/` `training/` `diffusion/` `evaluation/` `utils/` `data/` | `AGENTS.md` 契约：B 版目录只有注释 |
| **checkpoint** | `checkpoints/<name>-example/` | `checkpoints/<name>/` | 后缀由加载器统一加，不在各调用点拼 |
| **run log** | `logs/run/<name>-example/` | `logs/run/<name>/` | 同上 |
| **配置** | `configs/example/tier0.yaml` | `configs/practical/tier1.yaml` | 同一个加载器，两套文件 |
| **结果** | `example/RESULTS.md` | `RESULTS.md` | 各自就近，文件名对称 |
| **配置加载** | `load_config("example", "tier0")` | `load_config("practical", "tier1")` | 同一个 `example/common/version.py` |

**为什么后缀由加载器加而不是写在调用点**：拼错的后果不是报错，是**静默覆盖另一个
版本的权重**。两版算的是同一件事、跑同一个名字，读写错了看不出任何异常。

**为什么配置也要分开**：如果实践版沿用 `example/common/config.py` 里的硬编码常量，
那么「实验版能跑通」和「实践版配置是对的」就是两条从未互相验证过的代码路径。
两套配置走**同一个加载器**，才能保证只有数值不同、路径相同。

**为什么结果分两份文件**：实验版的数字来自 3.2M 参数 / 20 万 token，
实践版来自 ~50M / 332M token，**两者不可比**。放进同一张表，最可能的结果是有人
拿实验版的 eval loss 去引用。

### 1.2 模型命名规则

**名字就是 pipeline**。从左到右读，就是这个模型的完整来路。

```
<family>-<size>[-<impl>][-<stage>]...
```

| 段 | 取值 | 含义 |
|---|---|---|
| `family` | `ar` / `dlm` / `qwen3` | AR / Direct-DLM / 现成 Qwen3 |
| `size` | `3.2m` / `50m` / `0.6b` | 参数量，小写 m / b |
| `impl` | 随家族，见下 | **这份权重是怎么跑出来的** |
| `stage` | `sft` / `mini_a2d` / `dllm_a2d` / `dpo` | 训练阶段，按发生顺序追加 |

**`impl` 的取值随家族变化**（同一件事的两种写法）：

| 家族 | `impl` 取值 | 含义 |
|---|---|---|
| `ar-` | `hf` / `native` | `transformers.Qwen3ForCausalLM` / 纯 `torch.nn` 手写 |
| `dlm-` | `hf` / `dllm` | `transformers` + 本项目手写的 masked CE / 调 `dllm` 包的 `MDLMTrainer` |
| `qwen3-` | — | 固定 `transformers`，不写 |

**`dlm-` 没有 `native` 版本**：Direct-DLM 的意义是「从随机初始化直接预训练一个
diffusion LM」，要对比的是**两种主流实现**（自己写 vs 调包），而不是再引入第三个
建模实现 —— 那属于 G1 已经回答过的问题。

**规则**

1. **`-` 分段，`_` 只出现在「实现 × 阶段」内部**：`mini_a2d` 是一个整体，不再拆。
2. **每个阶段加一段**，顺序就是执行顺序，不可重排。
3. **pretrain 不占段**：`ar-3.2m-hf` 就是 pretrain 完成后的权重。
4. **`sft` 的目标函数由前一段决定**，不另写：前段是 `ar-*` → next-token CE；
   前段是 `*_a2d` / `dlm-*` → masked CE。所以 `ar-3.2m-hf-sft` 和
   `ar-3.2m-hf-mini_a2d-sft` 都是 SFT，objective 不同而名字不必区分。
5. **`impl` 只对自训家族（`ar-` / `dlm-`）要求**；`qwen3-` 固定用
   `transformers`，不写。
6. **家族名隐含 tokenizer track**：`ar-` / `dlm-` ⇒ 6400-vocab track；
   `qwen3-` ⇒ Qwen3 track。
7. **实验版 checkpoint 加 `-example` 后缀**：`checkpoints/ar-3.2m-hf-mini_a2d-example/`。
   名字其余部分与实践版完全相同，唯一区别就是有这个后缀。

**命名里有两条互不相关的「包」轴，段的位置不同**

| 段 | 管什么 | 不用包 | 用包 |
|---|---|---|---|
| `impl`（`hf` / `native` / `dllm`） | **这份权重怎么跑出来的** | `native`、`mini_a2d` | `hf`、`dllm_a2d` |
| `stage`（`mini_a2d` / `dllm_a2d`） | **A2D 目标函数怎么算** | `mini_a2d` | `dllm_a2d` |

自由组合，两个轴互不依赖：

```
ar-3.2m-native-sft-mini_a2d    纯 torch 建模 + 手写 A2D
ar-3.2m-hf-sft-dllm_a2d        transformers 建模 + dllm 包 A2D
```

**四条基准例子**

| 名字 | pipeline | 含义 |
|---|---|---|
| `ar-50m-native-sft-mini_a2d` | pretrain(AR,native) → SFT → A2D(minimind) | **原路线** |
| `ar-50m-native-mini_a2d-sft` | pretrain(AR,native) → A2D(minimind) → SFT | **新路线** |
| `dlm-50m-sft` | Direct-DLM pretrain → SFT | G2 对照 |
| `qwen3-0.6b-mini_a2d` | Qwen3-0.6B → A2D(minimind) | G4 |

**实验版（Tier 0）完整清单**（实践版把 `3.2m` 换成实测后的 size）

| 名字 | 组 | 状态 |
|---|---|---|
| `ar-3.2m-hf` | G1 | ✅ eval 4.6356 |
| `ar-3.2m-native` | G1 | ✅ eval 4.6283 |
| `ar-3.2m-hf-sft` | G3 对照 | ✅ eval 6.2659 |
| `ar-3.2m-native-sft` | G3 对照 | ✅ eval 6.2475 |
| `ar-3.2m-hf-mini_a2d` | G2/G3 新 | ✅ eval 4.9723 |
| `ar-3.2m-native-mini_a2d` | G2/G3 新 | ✅ eval 4.9692 |
| `ar-3.2m-hf-dllm_a2d` | G5 | ✅ eval 4.9765 |
| `ar-3.2m-native-dllm_a2d` | G5 | ✅ eval 4.9728 |
| `ar-3.2m-hf-mini_a2d-sft` | G3 新路线 | ✅ eval 6.3872 |
| `ar-3.2m-native-mini_a2d-sft` | G3 新路线 | ✅ eval 6.4133 |
| `ar-3.2m-hf-dllm_a2d-sft` | G3/G5 | ✅ eval 6.3845 |
| `ar-3.2m-native-dllm_a2d-sft` | G3/G5 | ✅ eval 6.4125 |
| `ar-3.2m-hf-sft-mini_a2d` | G3 原路线 | ✅ eval 5.0315 |
| `ar-3.2m-native-sft-mini_a2d` | G3 原路线 | ✅ eval 5.0335 |
| `ar-3.2m-hf-sft-dllm_a2d` | G3/G5 原路线 | ✅ eval 5.0378 |
| `ar-3.2m-native-sft-dllm_a2d` | G3/G5 原路线 | ✅ eval 5.0376 |
| `ar-3.2m-hf-sft-mini_a2d-sft` | G3 原路线（尾部 DLM-SFT） | ✅ eval 5.9773 |
| `ar-3.2m-native-sft-mini_a2d-sft` | G3 原路线（尾部 DLM-SFT） | ✅ eval 5.9713 |
| `ar-3.2m-hf-sft-dllm_a2d-sft` | G3 原路线（尾部 DLM-SFT） | ✅ eval 5.9776 |
| `ar-3.2m-native-sft-dllm_a2d-sft` | G3 原路线（尾部 DLM-SFT） | ✅ eval 5.9709 |
| `ar-3.2m-native-sft-dpo` | G3 DPO | ✅ loss 0.8853 -> 0.7623 |
| `dlm-3.2m-hf` | G2 | ✅ eval 5.6427 |
| `dlm-3.2m-dllm` | G2 | ✅ eval 5.6460|

| `ar-3.2m-hf-sft-dpo` | G3 DPO 对照 | ✅ loss 0.8704 -> 0.7744 · acc 0.5 |

| `qwen3-0.6b-mini_a2d` | G4 | ✅ eval 4.8397 |
| `qwen3-0.6b-dllm_a2d` | G4 | ✅ eval 4.8238 |
| `qwen3-0.6b-mini_a2d-sft` | G4 | ✅ eval 3.7559 |

**两处偏离你的示例，需要确认**

| 你的示例 | 我写的 | 原因 |
|---|---|---|
| `dlm-size-sft` | `dlm-3.2m-hf-sft` | Direct-DLM 也有两种写法（transformers / dllm 包），不带 `impl` 分不开 |
| `dlm-3.2m-dllm-sft` | G2 | ✅ eval 6.6878 |
| — | 加 `dpo` 段 | RL 阶段目前只排在 P2，先把位置定下来 |

---

## 2. 五组对比

每组只动**一个变量**。格子里的模型对象与阶段见 §8 网格。

### G1 · 两种 AR 实现

| | |
|---|---|
| **问题** | `transformers` 写法和纯 torch 原生写法是不是同一个模型？ |
| **变量** | 模型实现（用包 / 不用包） |
| **格子** | `ar-hf` pretrain · `ar-native` pretrain |
| **验收** | 同权重同 batch 下 logits `max\|Δ\| ≤ 1e-5`（**C1**，实验版实测 6.91e-06） |
| **产出** | 两个 AR pretrain 权重 —— 后续所有 SFT / A2D 的起点 |

### G2 · Direct-DLM vs A2D

| | |
|---|---|
| **问题** | AR 权重对 DLM 值多少？ |
| **变量** | 起点权重（随机初始化 vs 从 AR 转换） |
| **备注** | `dlm-*` 只有两种写法：`hf`（transformers + 本项目手写的 masked CE）和 `dllm`（调包）。**没有 native 版** |
| **格子** | `dlm` pretrain + SFT · `a2d` pretrain + SFT（**都是 3.2M 极小模型，不含 0.6B**） |
| **验收** | 两格都要能 chat；assistant 段 eval loss 之差 = AR 权重的价值（实验版 pretrain 阶段实测 0.67 nat） |

### G3 · 顺序：原路线 vs 新路线

```
原路线：pretrain(AR) → SFT → A2D          ← 先 SFT 后 A2D
新路线：pretrain(AR) → A2D → SFT          ← 先 A2D 后 SFT
```

**原路线在前，因为它是被验证过的做法**：Qwen3-0.6B 和 `dllm` 的 A2D，
作用对象本来就是**已经能 chat 的模型**（`notes/03` §12 提到作者的 `dllm_768.pth`
是从 SFT 后的权重转出来的）。换句话说，业界已有的 A2D 全部落在原路线上。

**新路线是本项目要验的变体**：A2D 在裸文本上学会双向去噪，再用 SFT 教对话。
两条路线各有对方无法推断的失败模式，所以都做。

| | |
|---|---|
| **问题** | 先 SFT 还是先 A2D？ |
| **变量** | 顺序 |
| **格子** | 原路线：`ar-sft` → A2D → DLM-SFT · 新路线：`ar-a2d` → DLM-SFT |
| **验收** | 两条线的「能 chat」四条门槛（§6.3）都能报；差值就是顺序的代价 |
| **注意** | **AR-SFT 是必需的对照组**。没有它，无法判断 DLM 的输出是「变差了」还是「本来就差」 |

**2026-10-02 实测结论（Tier 0）**。两条路线都终止于「assistant 段上的 A2D loss」，
同一份 SFT 语料、同一个目标，因此四对实现可直接比较：

| 路线 | hf/mini | hf/dllm | native/mini | native/dllm |
|---|---:|---:|---:|---:|
| 新路线 A2D → SFT | 6.4958 | 6.4880 | 6.5279 | 6.5088 |
| 原路线 SFT → A2D → SFT | **6.1029** | **6.0851** | **6.0851** | **6.0485** |
| 差（原 − 新） | −0.393 | −0.403 | −0.443 | −0.460 |

**原路线稳定优约 0.40 nat，四个实现对全部同向。** 与 SPEC 把原路线排在前面
（业界已有的 A2D 全部落在原路线上）一致。

⚠️ **AR-SFT 对照组（`ar-3.2m-{hf,native}-sft`，eval 6.3639 / 6.3462）不能和上表直接比**——
它是 AR 的 next-token CE，上表是 A2D 的加权 masked CE，两者不是同一个量。
跨模型族的比较要靠 §6.3 的「能 chat」四条门槛或 DPO 的偏好准确率，不能靠 loss 数值。

### G4 · 不同基模做 A2D

| | |
|---|---|
| **问题** | A2D 换个基模还成立吗？ |
| **变量** | 基模的规模与来源 |
| **格子** | `ar-hf-sft`（3.2M）· `ar-native-sft`（3.2M）· `qwen3-0.6b`（600M + LoRA） |
| **验收** | 三者都要能 chat；报 eval loss / 步数 / 可训参数占比 |
| **注意** | 两条 tokenizer track 的 loss **不可直接比较**（词表 6400 vs 151,669）；跨 track 按字符/字节归一化 |

### G5 · `dllm` 包 vs minimind-dlm 手写

| | |
|---|---|
| **问题** | 上游包和手写实现是不是同一个东西？ |
| **变量** | 实现 |
| **格子** | 每一个 A2D 格都跑两条路径 |
| **验收** | 同 checkpoint 同 seed **同 corruption** 下 loss 相对差 `< 1%`（**C2/C3**，实测 7.6e-08 / 1.5e-07）；转换 **0 丢失 0 漂移**（**C5**） |

### 2.6 对照组汇总

| 组 | 变量 | 依赖 |
|---|---|---|
| G1 | 模型实现 | — |
| G2 | 起点权重 | G1 |
| G3 | 顺序 | G1（AR-SFT 是 G3 的对照） |
| G4 | 基模 | G1 · G3 |
| G5 | 实现 | G1 |

---

## 3. 模型、数据与 tokenizer

### 3.1 统一架构

Qwen-style decoder-only：RoPE · RMSNorm · SwiGLU · GQA · QK-Norm · tied embedding。
AR 用因果 mask，DLM 去掉因果 mask（**权重一个都不动**）。

### 3.2 两条 tokenizer track

| | 6400-vocab track | Qwen3 track |
|---|---|---|
| 用在哪 | 实验版 + 实践版的全部自训模型 | 只有 `qwen3-0.6b` |
| 词表 | 6400 | 151,669 |
| MASK | **id 27**（`<\|buffer1\|>`，预留，**不扩词表**） | **id 151662**（`<\|fim_pad\|>`，**复用，不新增 token、不 resize embedding**） |
| EOS / BOS | 2 / 1 | 151645 / **无 BOS**（`bos_token_id is None`） |

两条 track 的 loss、perplexity、token/s **只能在各自 track 内比较**。

> `SPEC` 最初写「Qwen3 的 MASK 用 151669」，实测**越界**（合法 id 只到 151668），
> 改为复用 151662：不动 embedding，参数量保持精确值。详见 `example/ISSUES.md` DONE-1。

### 3.3 语料

| 阶段 | 源 | 规模 |
|---|---|---|
| pretrain | `data/prepared/minimind3-6400/` | 332M token |
| pretrain（实验版） | `data/prepared/tinystories-6400/` | 3.1M token，取前 ~20 万 |
| sft | `data/raw/minimind_dataset/sft_t2t.jsonl` | 14 GiB，509 万条对话 |
| dpo | `data/raw/minimind_dataset/dpo.jsonl` | 53 MB |

**对话数据必须先展平。** `scripts/prepare_data.py` 的 `read_text()` 只取 `text` 键，
对话数据没有 `text`，会被**静默跳过**（0 token，不报错）。

展平产物 `data/prepared/sft-6400/`：

| 文件 | 内容 | 用途 |
|---|---|---|
| `tokens.bin` | 展平后的 token（含 BOS/EOS） | 输入 |
| `doc_lengths.npy` | 每条对话的 token 数 | packing / 边界 |
| **`assistant_spans.npy`** | `[n_docs, 2]`，assistant 段的 token 起止 | **加噪 / loss 区间** |
| `meta.json` | vocab / n_docs / total_tokens / sha256 | 自检 |

展平后的文本（纯文本标记，不新增 special token）：

```
<user>\n{提问}\n<assistant>\n{回答}\n
                   ^^^^^^^^^^^^^  assistant 段：含末尾 \n，不含 <assistant> 标记
```

**`assistant_spans.npy` 是 SFT 的核心。** 没有它：

- AR 只能算全序列 CE（等于没做 SFT）；
- DLM 会把 prompt 也遮掉（等于没在看问题）——**而 loss 照样下降**。

---

## 4. 配置

### 4.1 Tier 0（实验版，唯一的冒烟配置）

```yaml
hidden_size: 256
num_layers: 2
num_attention_heads: 2
num_key_value_heads: 1
head_dim: 128
intermediate_size: 768
context_length: 128
vocab_size: 6400
tie_word_embeddings: true
```

参数量解析值 **3,213,056**（tied 只算一次）。**只用于 pipeline validation，不代表正式规模。**

### 4.2 Tier 1（实践版，待测后定）

```yaml
hidden_size: 512        # 候选，冒烟+吞吐测量后再拍
num_layers: 8
num_attention_heads: 8
num_key_value_heads: 4
intermediate_size: 1408
context_length: 512
```

**先测吞吐阶梯，再定参数量。** 参数量、序列长度、数据量是耦合的，先拍数字再测是本末倒置。

### 4.3 已知硬件约束

`max_seq_len ≤ 512`（attention O(T²)）· batch 8 就够 · 不开 gradient checkpointing ·
packing 优于 padding · 瓶颈在 attention backward kernel · run-to-run 方差可达 ±20%。

---

## 5. 环境

Python 3.14 + ROCm torch 2.12 + **transformers 钉 4.57.0**（dllm 的 pin）。
不装 deepspeed / wandb / torchvision / torchaudio。`dllm` 从钉死的 commit clone 后打
本仓库的 patch 再 editable 安装。HF 走 `hf-mirror` 镜像。

**不要写 5.x 的代码**：5.x 的 `create_causal_mask` 签名变了，dllm 的 A2D 模型接不上。
transformers 4.57 里「关掉因果 mask」的正确做法是**传一个 4D 全零 attention mask**
（`create_causal_mask` 会原样返回已准备好的 4D mask），不是改 `self_attn.is_causal`
（那是 5.x 才有的字段）。

---

## 6. 验收

### 6.1 跨实现对拍（G5 的硬指标）

| # | 比什么 | 阈值 | 实验版实测 |
|---|---|---|---|
| **C1** | 原生 torch vs `transformers` logits（同权重同 batch） | `≤ 1e-5` | 6.91e-06 |
| **C2** | minimind-dlm 手写 vs `dllm` loss，**同 corruption** | 相对差 `< 1%` | 7.56e-08 |
| **C3** | 同上，在原生 base 上 | `< 1%` | 1.54e-07 |
| **C4** | 连续跑 N 步 vs M + resume + (N−M) | 逐步差 `≤ 1e-6` | 0.0 |
| **C5** | A2D 转换 | 0 丢失 / 0 新增 / 0 形状变化 / 0 漂移 | 全 0 |

**对拍必须在同一输入上比。** 两套实现各自抽随机数 → loss 不同 → 无法区分
「实现错了」和「随机数不一样」。做法是两边用同一个全局 seed 且 RNG 消耗顺序一致，
并**验证两次 corruption 逐位相同**，而不是只看两个相近的 loss 数字。

### 6.2 SFT 专项

- **T4a 展平正确**：抽查 3 条，`decode(tokens)` 能还原带 `<user>` / `<assistant>` 标记的原文；
  区间长度与回答长度一致（误差 ≤ 1 token）。
- **T4b 区间正确**：label 在 user 段全 `-100`、assistant 段全不是。**区间错位是最阴的
  bug** —— loss 照样下降，模型学会了「不看问题直接答」。
- **T4f 区间外永不加噪**：抽 100 个 batch，断言 `MASK` 出现的位置 ⊆ assistant 区间。
- **T4g 两种 objective 在 SFT 下仍一致**：区间版 C2。
- **T4h AR-SFT 与 DLM-SFT 的 assistant 段 eval loss 可比**，差值即「双向去噪 vs 自回归」
  在同预算下的实际代价。

### 6.3 「能 chat」的门槛（SFT 的真正验收）

不追求指标，只要求能像个对话模型回话。四条同时满足：

| # | 判据 | 阈值 |
|---|---|---|
| 1 | assistant 段 eval loss | `≤ 1.5 ×` 同条件 AR-SFT |
| 2 | rep-4（≥64 token 生成） | `< 0.5` |
| 3 | 三个 prompt 都能在 `<assistant>` 后输出非空、格式正确 | — |
| 4 | 不回声提问 | 抽查 2 条，回答不含 prompt 原文 |

**步数**：能 chat 即可，不要求收敛。先 200 步看 rep-4 和格式，不够追到 400 / 800，
每加一档都重报 rep-4。

### 6.4 测量规则

- **必须同时报 train-only 和 wall-clock 两个吞吐**（验证和采样不污染 train-only 分母）。
- benchmark ≥ 400 步，丢弃预热；`Predicted` 与 `Measured` 分列。
- **所有 DLM 结论必须报 repetition rate，不能只报 loss** —— DLM 失败最典型的表现
  就是复读而 loss 照样下降（`notes/03` §11）。
- **rep-4 不能给实现排名**：同一方法换一个 base，A2D 在 0.40 ~ 0.81 之间跳
  （32 token 样本的噪声）。正式评测要几百 token × ≥5 个 prompt。

---

## 7. 目录

```text
configs/example/         # 实验版配置 tier0.yaml
configs/practical/       # 实践版配置 tier1.yaml（待测吞吐后填）
RESULTS.md               # 实践版实测
example/                 # 实验版参考实现 + RESULTS.md + ISSUES.md + teach/
scripts/                 # 环境、数据、训练、可视化
data/                    # dataset、packing、tokenizer 适配
tokenizer/               # 6400 BPE
models/hf_qwen/          # Transformers Qwen 适配
models/minimind_qwen/    # 原生 PyTorch Qwen-style model
training/ar_hf/          # G1 行1
training/ar_native/      # G1 行2
training/a2d_minimind/   # G5 不用包
training/a2d_dllm/       # G5 用包
training/direct_dlm/     # G2
training/sft_ar_hf/      # G2/G3 的 AR-SFT 对照
training/sft_ar_native/
training/sft_a2d_hf/     # G3 新路线 / G4
training/sft_a2d_native/
training/dpo_ar_native/  # RL（只在 AR 上做）
diffusion/               # corruption、scheduler、loss、sampler
evaluation/              # generation、benchmark、repetition rate
utils/ data/             # logger、device、seed、checkpoint、metrics
example/                 # 实验版参考实现 + teach/ + RESULTS.md + ISSUES.md
notes/ logs/ scripts/patches/
```

依赖单向：公共数据和工具被模型、训练、评测使用；A2D 可以读取 AR 权重，但不导入 AR 训练循环。

---

## 8. 任务网格

**行 = 模型对象**，**列 = 阶段**。✅ = 实验版已完成（在 main）· ⬜ = 未写 · P0/P1/P2 = 优先级。

### 8.1 实验版（AI 写，`example/`，Tier 0 3.2M）

模型名见 §1.3；下表是「哪一格产出哪个名字」。

| base | pretrain | a2d | sft | rl |
|---|---|---|---|---|
| **ar**（hf / native） | `ar-3.2m-{hf,native}` ✅ | `ar-3.2m-{hf,native}-{mini,dllm}_a2d` ✅ | `…-sft` ✅ · `{mini,dllm}_a2d-sft` ✅ · `sft-{mini,dllm}_a2d` ✅ · `sft-{mini,dllm}_a2d-sft` ✅ | `…-sft-dpo` ✅ `ar-3.2m-hf-sft-dpo`（native 版与 dlm 版未做，见 OPEN-10） |
| **qwen3-0.6b** | — 用现成 checkpoint | `qwen3-0.6b-{mini,dllm}_a2d` ✅ | `…-sft` ✅ | ⬜ **放弃**（OPEN-10 同理） |
| **dlm**（直接预训练 DLM） | `dlm-3.2m-hf` ✅<br>`dlm-3.2m-dllm` ✅ | — 本来就是 DLM | `dlm-3.2m-{hf,dllm}-sft` ✅ | **放弃**（DPO）—— DLM 无序列 logprob |

### 8.2 实践版（你写，根目录，Tier 1 ~50M / ctx 512）

模型名把 `3.2m` 换成实测后的 size（如 `50m`）。

| base | pretrain | a2d | sft | rl |
|---|---|---|---|---|
| **ar-hf** | `models/hf_qwen/`+`training/ar_hf/` | `training/a2d_minimind/`（不用包）<br>`training/a2d_dllm/`（用包） | `training/sft_ar_hf/` | `training/dpo_ar_hf/` |
| **ar-native** | `models/minimind_qwen/`+`training/ar_native/` | ↑ **同一对目录**，`--base` 切 | `training/sft_ar_native/` | `training/dpo_ar_native/` |
| **qwen3-0.6b** | — | ↑ 同一对目录，行3 分支（LoRA） | 同 sft 目录，LoRA | **放弃**（DPO） |
| **dlm** | `training/direct_dlm/`（手写 masked CE）<br>同一目录 + `dllm` 包（调 `MDLMTrainer`） | — | 同 sft 目录 | — |

**a2d / sft 三个 base 共用同一份代码**：`training/a2d_*/` 吃任意 AR checkpoint，
`--base` 切换。A/B 的轴是**用不用包**，不是加载哪个 base。

### 8.3 依赖顺序

```
G1 (ar-hf, ar-native pretrain)
  ├─▶ G2 (direct-dlm vs a2d)
  ├─▶ G5 (dllm vs minimind)
  └─▶ SFT (AR-SFT 对照组) ──▶ G3 原路线 ──▶ G4
        └─▶ A2D ──▶ SFT ──────────────▶ 新路线
```

### 8.4 对拍命令

| 任务 | 命令 | 判定 |
|---|---|---|
| 实验版全跑 | `bash example/smoke_all.sh --quick` | 任一格 FAILED 会打出来 |
| G1 | `python example/check_ab.py c1 --layer-report` | `max\|Δ\| ≤ 1e-5`，失败时逐层定位 |
| G5 | `python example/check_ab.py c2` · `c5` | `< 1%` · 全 0 |
| 逐行对拍 | `python example/tests/test_*.py` | 失败非零退出 |
