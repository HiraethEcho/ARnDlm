# Step 12 — Qwen3-0.6B LoRA 两格（行3）

**你要写的**：`training/a2d_dllm/` + `training/a2d_minimind/` 的行3 分支
**A 版参考**：`example/base/qwen3_0p6b/`
**验收**：两格都跑通，重复率**必报**（`SPEC` §6：T2c 必须报 repetition rate）。

---

## 12.1 行3 和前面九格的三个根本差异

| | 6400 track（格1-8） | Qwen3-0.6B track（格9-10） |
|---|---|---|
| 词表 | 6400，MASK = 27（预留） | 151,670，MASK = **151669（需新增）** |
| 数据 | `data/prepared/*` 直接读 | 要用 Qwen tokenizer **现场 tokenize** |
| backbone | 全参训练 | **冻结**，只训 LoRA r=8 |

**三条 tokenizer track 的 loss 不能互相比**（`SPEC` 3.3）。
跨 track 要按相同原文和字符数归一化，A 版没做这件事，所以
格9/10 的 eval 10.1 不能和格3 的 4.97 放一张表里比。

## 12.2 MASK token 的三种处理

`SPEC` 3.2 写「MASK 使用 id 151669」，但 Qwen3-0.6B 的 tokenizer
最大 id 是 **151668**。151669 恰好是**新加 token** 拿到的 id。

| 方案 | 做法 | 代价 |
|---|---|---|
| A（当前） | 加 `<|diff-mask|>`，`resize_token_embeddings(151670)` | embedding 从 151936 行缩到 151670，**掉 272,384 参数** |
| B | 复用未使用的 special token（如 `<|fim_pad|>` 151662） | 不改 embedding，但要确认它在训练数据里从未出现 |
| C | 改 SPEC，写「复用 151662」 | 最干净，但要你拍板 |

A 版选了 A 并在 `ISSUES.md` OPEN-1 记着。**参数量对不上时不要查 bug**，
是 resizing 掉的。

## 12.3 LoRA 挂哪、冻结哪

```
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]   # 只挂 attention
r = 8, alpha = 16, dropout = 0
```

实测：**2,293,760 / 598,071,296 = 0.384%** 可训。

理由是 `notes/03` §10 的原话：FFN 存知识，AR→DLM 的核心变化是
注意力从因果变双向。**这一条是可证伪的**，`build_a2d_lora(trainable=...)`
留了 `freeze_ffn` 和 `qk` 两个档，正好用来做那个消融。

A 版测试已经验了默认档：`trainable_names` 里**没有一个含 `.mlp.`**。

`dropout = 0` 不是随便填的：LoRA 有 dropout 就没法做 bit 级对拍。

## 12.4 数据：不能读 `data/prepared/`

那三个文件是 **6400-vocab** tokenize 的。行3 用 Qwen tokenizer，
所以现场从 `data/raw/tinystories.jsonl` 重新 encode：

```python
ids.extend(tokenizer.encode(text, add_special_tokens=False))
ids.append(tokenizer.eos_token_id)          # Qwen3 没有 BOS！
```

**`bos_token_id` 是 `None`。** 直接 `ids.append(tokenizer.bos_token_id)`
会让 `torch.tensor` 报 `TypeError: 'NoneType' object cannot be interpreted
as an integer` —— 报错信息里完全看不出是 tokenizer 的问题。

编码速度 ~1.5M tok/s，所以现场 tokenize 2000 篇的开销可以忽略。

## 12.5 A 版实测（诚实记录，不解释）

| 格 | loss 首 → 末 | eval | tok/s | rep-4 |
|---|---|---|---|---|
| `qwen3-0.6b-dllm-a2d-lora` | 11.9521 → 10.0386 | 10.0984 | 476 | **0.9828** |
| `qwen3-0.6b-minimind-a2d-lora` | 12.2210 → 10.0533 | 10.1203 | 687 | **0.9828** |

**两格的输出都是一串 `.`，rep-4 0.98。**

这不是 bug，是 200 步 / 51,200 token 对一个 600M 模型什么都没教会。
`notes/03` §11 早就说过「别期待 A2D 之后模型变强」。

**这一格真正验证的是**：LoRA 注入、冻结、mask token、双向转换、
两种 loss 在真权重上都跑得通。**不是**「LoRA A2D 有用」。

## 12.6 吞吐差距 476 vs 687

又是 `compute_loss` 的 python 开销（`check_ab c2 --bench`：
14.31 ms vs 22.95 ms）。在 0.6B 上这个固定开销占比更大，
所以差距从 1.6 倍拉到了 1.44 倍 —— 比例反而小了，
因为每步的主体（模型前反向）变重了。

## 12.7 坑

| 坑 | 症状 |
|---|---|
| 直接用 id 151669 | 下标越界，或悄悄用了别的 token |
| 读 `data/prepared/*` | 6400-vocab 的 token，行3 索引全错 |
| `bos_token_id` 为 None 直接 append | `TypeError` 指向 torch，不指向 tokenizer |
| LoRA 带 dropout | 对拍永远对不齐 |
| 把 10.1 和 4.97 放一起比 | 跨 track，比的是词表大小 |
| rep-4 不报 | 这一格的验收就不算过（`SPEC` §6 明写） |

## 12.8 验收命令

```sh
.venv/bin/python example/tests/test_lora_0p6b.py
.venv/bin/python example/base/qwen3_0p6b/train_lora_dllm.py
.venv/bin/python example/base/qwen3_0p6b/train_lora_minimind.py
```