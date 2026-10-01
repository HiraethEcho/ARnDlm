# Vocab / Tokenizer / Mask Token

## tokenizer

把文本切成整数编号。干三件事:

| 职责 | 例子 |
|---|---|
| 编码 | `'人工智能'` → `[2225]` |
| 解码 | `[2225]` → `'人工智能'` (byte-exact) |
| 持词表 | 记住全部 token 和合并规则 |

**命名陷阱: 它不按「词」切。** 它按**字节**切, 再看哪些字节常一起出现
(byte-level BPE)。真正的按词切是 jieba, 完全另一回事。

```
'人工智能'      → 1 token   [2225]
'Transformer'  → 3 tokens  ['Trans', 'form', 'er']
'🧠'            → 3 tokens  [4544, 136, 290]
```

**模型从头到尾没见过一个汉字, 它只见过 `[2225, ...]`。**

---

## vocab 是什么

**vocab (词表) = tokenizer 的编号表。** `vocab_size` 是它的行数。

模型里有一块 `embedding` 矩阵, 形状 `(vocab_size, hidden_size)`, 第 `i` 行
是 token `i` 的向量。`lm_head` 通常也是 `(vocab_size, hidden_size)`;
当 `tie_word_embeddings=True` 时两者**共享同一份权重**。

### 为什么 vocab 大小决定模型能不能训起来

| | Qwen3-0.6B | minimind-3 |
|---|---|---|
| `vocab_size` | 151936 | 6400 |
| `hidden_size` | 1024 | 768 |
| embedding 参数量 | 151936 × 1024 = **155M** | 6400 × 768 = 4.9M |
| 占整个模型 | **23%** | 7% |
| 标称 params | 680M | 68.8M |
| 真正干活的 transformer | 525M | 63.9M |

Qwen3-0.6B 宣称 680M 参数, 其中 **155M 纯粹是查表用的 embedding table**。

**64M 的模型配 151936 的 vocab 是荒谬的** —— 大半参数是查表, 而 332M token
根本训不完 1.5 亿个 embedding。

> **换 vocab = 换 tokenizer = 权重全部作废。**
> vocab 是架构的一部分, 不是可以随便换的超参。

---

## 怎么查任何模型的 config

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
cfg.num_attention_heads    # Q heads
cfg.num_key_value_heads    # K/V heads (GQA)
cfg.head_dim               # 可能不存在, 要自己算
```

不用 transformers 也可以 —— 直接读模型仓库里的 `config.json`。
想知道 tokenizer 到底多少条, 直接读 `tokenizer.json` 的 `model.vocab` 长度,
或用独立的 `tokenizers` 库 (`Tokenizer.from_file(...)` → `.get_vocab()`)。

---

## mask token (DLM 需要)

DLM 要在序列里插入 `[MASK]` 占位符, 需要一个 `mask_token_id`。

### 方案 A: 复用空闲的 special token

很多 tokenizer 继承了一堆用不上的占位符 (Qwen 系列从 Llama 继承的
`<|object_ref_start|>`、`<|vision_pad|>`、`<|buffer1|>` 等)。

**minimind 的 tokenizer 里 `id 27` 就是 `<|buffer1|>` —— minimind 讨论 618
的作者用的正是 `mask_token_id: 27`。**

### 方案 B: 用 `vocab_size` 里的空闲 slot (推荐)

`config.vocab_size` 常常**故意大于** tokenizer 的实际 token 数, 留出空位
给 pad/special token。Qwen3-0.6B:

```
config vocab_size  : 151936
tokenizer 实际最大 : 151669  (151668 是最高的 added_token)
空闲 slot          : 267 个 → 151669 .. 151935
```

**直接用 151669。不需要 `resize_token_embeddings`, 所有已有 token id 不变,
预训练权重 100% 兼容。** LLaMA 系列的经典做法。

验证方法:

```python
cfg = AutoConfig.from_pretrained(model)
vocab = Tokenizer.from_file("tokenizer.json").get_vocab()
real = max(vocab.values()) + 1
free = cfg.vocab_size - real          # 267
assert max(added_token_ids) < real     # 确认 special token 没占这些 slot
```

### 方案 C: `add_special_tokens` + resize

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<|mask|>"]})
model.resize_token_embeddings(len(tokenizer))
```

会改变 embedding 形状, 已有权重需要重新加载。**能用, 但比 A/B 麻烦。**
