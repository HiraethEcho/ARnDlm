# src/ — 七条路线

依赖方向**单向**, 永远不要反过来 import:

```
base ──> utils ──> transformer ──┐
  │                  │           ├──> dllm-a2d ──> dllm
  └──> minimind ─────┴──> a2d ────┘
```

`a2d` 在 AR 权重上做, 所以它 import `transformer` 或 `minimind` 拿权重,
但**不 import 它们的训练循环** —— 否则 A2D 的对比就被污染了。

| 目录 | 是什么 | ROADMAP |
|---|---|---|
| `base/` | 训练/后训练的公用工具: tokenizer、dataset、packing、masking、objective | 所有路线的底座 |
| `utils/` | 功能性代码: logger、web 展示、device、seed、checkpoint、metrics | 所有路线共用 |
| `transformer/` | 调用 `transformers` 包训练 + LoRA 后训练 | T1a |
| `minimind/` | 原生 pytorch 训练 + 后训练 | T1b |
| `a2d/` | 用 **minimind-dlm** 的思路手动做 A2D | T2a |
| `dllm-a2d/` | 用 **dllm** 包的工具做 A2D | T2b |
| `dllm/` | 直接用 dllm 包训 DLM | T3 |

T2c（对 Qwen3-0.6B 做 A2D）不占新目录 —— 它复用 `a2d/` 或 `dllm-a2d/`，
只是权重来源不同。

## 两条 A2D 路线的验收标准

`a2d/` 和 `dllm-a2d/` 做的是同一件事。**在同一 checkpoint 上必须跑出同一个
loss**（差 < 1%）。不一致就是其中一边错了 —— 而 A2D 的 bug 往往不会让 loss
变差, 只让模型变差。

## 开工顺序

模块桩**不要现在写**。做到 ROADMAP.md 的哪一步, 就先给那一步要用的模块写
docstring（作用 / 要写什么 / 注意 / 依赖 / 验收）, 再写实现。

```
base/dataset.py + base/packing.py
   └─> minimind/{config,attention,block,mlp,model,init}.py
          └─> minimind/pretrain.py                      ← T1b 最小闭环
                 └─> minimind/sample.py

transformer/pretrain.py  ← T1a, 与 T1b 并行, 用来对拍
   └─> transformer/lora.py                             ← T1a 后训练

a2d/convert.py + a2d/corruption.py + a2d/loss.py
   └─> a2d/train.py ──> a2d/sample.py                  ← T2a
dllm-a2d/  (同上, 调包)                                  ← T2b
dllm/                                                       ← T3
```

`base/objective.py` 里 AR 和 DLM 的 loss **必须分开写**, 不要合并。
两条路径要能在同一个 checkpoint 上对拍, 混在一起就没法判断是谁错了。

## 三个必踩的坑

抄一遍 `notes/04-qwen-architecture.md`, 这里只列最容易忘的:

1. **`head_dim` 从 `hidden_size // n_heads` 算, 不要从 config 读死。**
   Qwen3-0.6B 的 `head_dim=128` 而 `hidden_size/n_heads=64`, 两者不等。
2. **GQA 的 repeat 方向是 `n_heads // n_kv_heads`** —— 把 K/V（少）复制成 Q（多）。
   写反得到 0, GQA 静默失效, 然后在 SDPA 报形状不匹配。
3. **tied embedding 要用 `id()` 去重再传给 optimizer。**
   `tie_word_embeddings=True` 时 `embed_tokens.weight` 和 `lm_head.weight`
   是同一个对象, 不去重等于 lr 翻倍 —— 静默训错, 不报错。

## tokenizer

`src/base/tokenizer.json` — minimind BPE, vocab 6400, 6108 merges。

```
PAD 0 <|endoftext|>   BOS 1 <|im_start|>   EOS 2 <|im_end|>   MASK 27 <|buffer1|>
```
