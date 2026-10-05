# ARnDlm SPEC

## 1. 目标

在一致的 tokenizer、数据集和测量口径下，比较 Qwen-style Transformer 的 **AR** 与 **DLM**，
并把两者都训到**能对话**的程度。项目用于学习、复现和可测量对比，不追求 SOTA。

### 1.2 模型命名规则

**名字就是 pipeline**。从左到右读，就是这个模型的完整来路。

```
<family>-<size>[-<impl>][-<stage>]...
```

| 段       | 取值                                    | 含义                         |
| -------- | --------------------------------------- | ---------------------------- |
| `family` | `ar` / `dlm` / `qwen3`                  | AR / Direct-DLM / 现成 Qwen3 |
| `size`   | `3.2m` / `50m` / `0.6b`                 | 参数量，小写 m / b           |
| `impl`   | 随家族，见下                            | **这份权重是怎么跑出来的**   |
| `stage`  | `sft` / `mini_a2d` / `dllm_a2d` / `dpo` | 训练阶段，按发生顺序追加     |

**`impl` 的取值随家族变化**（同一件事的两种写法）：

| 家族     | `impl` 取值        | 含义                                                     |
| -------- | ------------------ | -------------------------------------------------------- |
| `ar-`    | `trans` / `native` | 充分使用`transformers`包 / 纯 `torch.nn` 手写            |
| `dlm-`   | `trans` / `dllm`   | 充分使用 `transformers` 包 / 调 `dllm` 包的 `MDLMTrainer` |
| `qwen3-` | —                  | 没有训练过程，不写                                       |

**规则**

1. **`-` 分段，`_` 只出现在「实现 × 阶段」内部**：`mini_a2d` 是一个整体，不再拆。
2. **每个阶段加一段**，顺序就是执行顺序，不可重排。
3. **pretrain 不占段**：`ar-3.2m-trans` 就是 pretrain 完成后的权重。
4. **smoke test checkpoint 加 `-smoke` 后缀**：`checkpoints/ar-3.2m-trans-mini_a2d-smoke/`。
   名字其余部分与正式训练完全相同，唯一区别就是有这个后缀。

例如

```
ar-3.2m-native-sft-mini_a2d    纯 torch 建模 + 手写 A2D
ar-3.2m-trans-sft-dllm_a2d        transformers 建模 + dllm 包 A2D
```

**四条基准例子**

| 名字                         | pipeline                                  |
| ---------------------------- | ----------------------------------------- |
| `ar-50m-native-sft-mini_a2d` | pretrain(AR,native) → SFT → A2D(minimind) |
| `ar-50m-native-mini_a2d-sft` | pretrain(AR,native) → A2D(minimind) → SFT |
| `dlm-50m-sft`                | Direct-DLM pretrain → SFT                 |
| `qwen3-0.6b-mini_a2d`        | Qwen3-0.6B → A2D(minimind)                |

## 五组对比

每组只动**一个变量**。格子里的模型对象与阶段见 §8 网格。

### G1 · 两种 AR 实现

|          |                                                                           |
| -------- | ------------------------------------------------------------------------- |
| **问题** | `transformers` 写法和纯 torch 原生写法是不是同一个模型？                  |
| **变量** | 模型实现（用包 / 不用包）                                                 |
| **格子** | `ar-trans` pretrain · `ar-native` pretrain                                |
| **验收** | 同权重同 batch 下 logits `max\|Δ\| ≤ 1e-5`（**C1**，实测锚点 6.91e-06） |
| **产出** | 两个 AR pretrain 权重 —— 后续所有 SFT / A2D 的起点                        |

### G2 · Direct-DLM vs A2D

|          |                                                                                                             |
| -------- | ----------------------------------------------------------------------------------------------------------- |
| **问题** | AR 权重对 DLM 值多少？                                                                                      |
| **变量** | 起点权重（随机初始化 vs 从 AR 转换）                                                                        |
| **备注** | `dlm-*` 只有两种写法：`trans`（transformers + 本项目手写的 masked CE）和 `dllm`（调包）。**没有 native 版** |
| **格子** | `dlm` pretrain + SFT · `a2d` pretrain + SFT（**都是 3.2M 极小模型，不含 0.6B**）                            |
| **验收** | 两格都要能 chat；assistant 段 eval loss 之差 = AR 权重的价值（pretrain 阶段实测锚点 0.67 nat）           |

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

|          |                                                                                    |
| -------- | ---------------------------------------------------------------------------------- |
| **问题** | 先 SFT 还是先 A2D？                                                                |
| **变量** | 顺序                                                                               |
| **格子** | 原路线：`ar-sft` → A2D → DLM-SFT · 新路线：`ar-a2d` → DLM-SFT                      |
| **验收** | 两条线的「能 chat」四条门槛（§6.3）都能报；差值就是顺序的代价                      |
| **注意** | **AR-SFT 是必需的对照组**。没有它，无法判断 DLM 的输出是「变差了」还是「本来就差」 |

### G4 · 不同基模做 A2D

|          |                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------- |
| **问题** | A2D 换个基模还成立吗？                                                                            |
| **变量** | 基模的规模与来源                                                                                  |
| **格子** | `ar-trans-sft`（3.2M）· `ar-native-sft`（3.2M）· `qwen3-0.6b`（600M + LoRA）                      |
| **验收** | 三者都要能 chat；报 eval loss / 步数 / 可训参数占比                                               |
| **注意** | 两条 tokenizer track 的 loss **不可直接比较**（词表 6400 vs 151,669）；跨 track 按字符/字节归一化 |

### G5 · `dllm` 包 vs minimind-dlm 风格手写

|          |                                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **问题** | 上游包和仿照 minimind-dlm 实现是不是同一个东西？                                                                                     |
| **变量** | 实现                                                                                                                                 |
| **格子** | 每一个 A2D 格都跑两条路径                                                                                                            |
| **验收** | 同 checkpoint 同 seed **同 corruption** 下 loss 相对差 `< 1%`（**C2/C3**，实测 7.6e-08 / 1.5e-07）；转换 **0 丢失 0 漂移**（**C5**） |

### 2.6 对照组汇总

| 组  | 变量     | 依赖                      |
| --- | -------- | ------------------------- |
| G1  | 模型实现 | —                         |
| G2  | 起点权重 | G1                        |
| G3  | 顺序     | G1（AR-SFT 是 G3 的对照） |
| G4  | 基模     | G1 · G3                   |
| G5  | 实现     | G1                        |

## 3. 模型、数据与 tokenizer

### 3.1 统一架构

Qwen-style decoder-only：RoPE · RMSNorm · SwiGLU · GQA · QK-Norm · tied embedding。
AR 用因果 mask，DLM 去掉因果 mask（**权重一个都不动**）。

### 3.2 两条 tokenizer track

|           | 6400-vocab track                                 | Qwen3 track                                                                   |
| --------- | ------------------------------------------------ | ----------------------------------------------------------------------------- |
| 用在哪    | 自训模型                                         | 只有 `qwen3-0.6b`                                                             |
| 词表      | 6400                                             | 151,669                                                                       |
| MASK      | **id 27**（`<\|buffer1\|>`，预留，**不扩词表**） | **id 151662**（`<\|fim_pad\|>`，**复用，不新增 token、不 resize embedding**） |
| EOS / BOS | 2 / 1                                            | 151645 / **无 BOS**（`bos_token_id is None`）                                 |

两条 track 的 loss、perplexity、token/s **只能在各自 track 内比较**。

### 3.3 语料

两条 track 共用一套数据管道：源语料（minimind 数据集，中文对话）由 `common/prepare.py`
下载并 tokenize，每篇文档 `[BOS] … [EOS]` 包裹拼接，按 context_length 切定长块，
落盘 `data/prepared/minimind3-6400/`（tokens.bin / doc_lengths.npy / meta.json）。
自洽约束：三件文件互相对账；MASK id 不允许出现在正文；vocab 与 tokenizer 断言一致。

## 4. 配置

### 4.1 smoke

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

### 4.2 Tier 1（正式对照，已定档）

```yaml
hidden_size: 512 # 已实测定档（数字见 logs/dev/02-hardware-perf.md）
num_layers: 8
num_attention_heads: 8
num_key_value_heads: 4
intermediate_size: 1408
context_length: 512
```

**先测吞吐阶梯，再定参数量。** 参数量、序列长度、数据量是耦合的，先拍数字再测是本末倒置。

实测已回收：参数量 26,879,488（→ 命名 `ar-26.9m-*`），B8 / ctx512 下
train-only 7,995 tok/s、332M token ≈ 11.6 h。完整阶梯见 `logs/dev/02-hardware-perf.md`。

## 5. 环境

Python 3.14 + ROCm torch 2.12 + **transformers 钉 4.57.0**（dllm 的 pin）。
不装 deepspeed / wandb / torchvision / torchaudio。`dllm` 从钉死的 commit clone 后打
本仓库的 patch 再 editable 安装。HF 走 `hf-mirror` 镜像。

**不要写 5.x 的代码**：5.x 的 `create_causal_mask` 签名变了，dllm 的 A2D 模型接不上。
transformers 4.57 里「关掉因果 mask」的正确做法是**传一个 4D 全零 attention mask**
（`create_causal_mask` 会原样返回已准备好的 4D mask），不是改 `self_attn.is_causal`
（那是 5.x 才有的字段）。

## 6. 验收

### 6.1 跨实现对拍（G5 的硬指标）

| #      | 比什么                                                | 阈值                                  | 实测锚点                  |
| ------ | ----------------------------------------------------- | ------------------------------------- | ---------- |
| **C1** | 原生 torch vs `transformers` logits（同权重同 batch） | `≤ 1e-5`                              | 6.91e-06   |
| **C2** | minimind-dlm 手写 vs `dllm` loss，**同 corruption**   | 相对差 `< 1%`                         | 7.56e-08   |
| **C3** | 同上，在原生 base 上                                  | `< 1%`                                | 1.54e-07   |
| **C4** | 连续跑 N 步 vs M + resume + (N−M)                     | 逐步差 `≤ 1e-6`                       | 0.0        |
| **C5** | A2D 转换                                              | 0 丢失 / 0 新增 / 0 形状变化 / 0 漂移 | 全 0       |

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

| #   | 判据                                                  | 阈值                            |
| --- | ----------------------------------------------------- | ------------------------------- |
| 1   | assistant 段 eval loss                                | `≤ 1.5 ×` 同条件 AR-SFT         |
| 2   | rep-4（≥64 token 生成）                               | `< 0.5`                         |
| 3   | 三个 prompt 都能在 `<assistant>` 后输出非空、格式正确 | —                               |
| 4   | 不回声提问                                            | 抽查 2 条，回答不含 prompt 原文 |

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

## 7. layout

```text
AGENTS.md  SPEC.md  PLAN.md      # 契约与计划（不装代码）
notes/     logs/dev/             # 知识与手写记录（logs/run/ 不进 git）
common/                    # 底座：每条实现 x 每个阶段都用
  config.py  seed.py  device.py  tokenizer.py(+tokenizer.json)
  data.py    prepare.py  metrics.py
configs/                   # smoketest.yaml / tier1.yaml
scripts/                   # 环境（setup_env / install_dllm / setup_symlinks + patches）
src/                       # 格子专属代码
  models/  trans.py + native/（包，接口对齐 trans）   # G1 的两条实现
  training/ main.py ar_trans.py ar_native.py state.py
            sft.py dlm.py a2d_dllm.py a2d_mini.py dpo.py
  eval/    chat.py（能 chat 四门槛 + rep-4）
data -> /mnt/extra  checkpoints -> /mnt/extra   # 大文件软链（gitignore）
.venv -> 主仓 .venv              # uv 环境
```

代码成对：`x.py` 骨架（只有注释，待填）+ `x-example.py` 参考实现（AI 写，验收获收时跑它出正确行为）。


## 8. 任务网格

行 = 模型对象，列 = 阶段。同一入口（`src/training/main.py`）用 `--impl` / `--base` 切；
a2d / sft 各 base 共用同一对层，A/B 的轴是**用不用包**，不是加载哪个 base。
依赖单向：common 被 src 用；a2d 可以读 AR 权重，但不 import AR 训练循环。

| base          | pretrain                                                  | a2d                                        | sft         | rl (dpo)                          |
| ------------- | --------------------------------------------------------- | ------------------------------------------ | ----------- | --------------------------------- |
| **ar-trans**  | `src/training/ar_trans.py`（Trainer 用包）                 | `a2d_dllm.py`（调包）/ `a2d_mini.py`（手写） | `src/training/sft.py` | `src/training/dpo.py`             |
| **ar-native** | `src/training/ar_native.py`（native 包）                   | ↑ 同一对层，`--impl` 切                     | 同 sft 层   | 同 dpo 层                          |
| **qwen3-0.6b** | —（现成权重）                                             | ↑ 同层 + LoRA 分支（peft）                  | 同层，LoRA  | **放弃**（DPO）                    |
| **dlm**       | `src/training/dlm.py`（trans=手写 masked CE / dllm=调包）   | —                                          | 同 sft 层   | —（DLM 无序列 logprob，需 ELBO）   |

### 8.1 依赖顺序

pretrain（G1 两格）→ SFT / A2D（可并行）→ DLM / DPO → tier1 复跑全部列。
smoke 档（3.2M）先跑通全链，再上 tier1（26.9M）。

### 8.2 对拍

C1-C5 / SFT T4 系列的阈值与实测锚点在 §6；实现顺序与验收归属见 `PLAN.md` 的阶段表。

