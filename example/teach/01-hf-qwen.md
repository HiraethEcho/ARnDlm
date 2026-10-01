# Step 1 — `models/hf_qwen/`：用包建模型

**你要写的**：`models/hf_qwen/`（Qwen3 配置 + 加载 + 形状信息）
**A 版参考**：`example/common/model.py` + `example/base/hf/model.py::build_model`
**验收**：同 Tier 0 配置下，你的模型和 `transformers.Qwen3ForCausalLM` 的
logits **逐元素相等**（fp32 CPU，`max|Δ| ≤ 1e-5`），参数量解析式对得上。

---

## 1.1 这一步在矩阵里的位置

行1 的 base。行2 的 base（step 3）是同一份形状的纯 torch 版 —— 两者的
`ModelCfg` 应该是同一个数字来源，不要各写一份常量。

## 1.2 必须显式写的字段

`transformers` 的默认值会随版本变，静默默认的字段会变成一个查不出来的 A/B 差值：

| 字段 | Tier 0 取值 | 不写会怎样 |
|---|---|---|
| `tie_word_embeddings` | `True` | False → 参数量多 1.6M，且 A2D 转换时词表扩展的形状对不上 |
| `use_sliding_window` | `False` | 开了滑动窗口后 causal mask 变了，C1 永远对不齐 |
| `sliding_window` / `max_window_layers` | `None` / `0` | 同上 |
| `head_dim` | 128（= hidden / heads） | 4.57 会自己算，但换版本就不一定 |
| `rope_theta` | 10000.0 | 同上 |
| `attention_dropout` | 0.0 | 有 dropout 就没法做 bit 级对拍 |

**版本锁 4.57.0**（`AGENTS.md`）。5.x 的 `create_causal_mask` 签名变了，
`dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` 接不上。

## 1.3 参数量解析式

先手算，再和模型对（`example/tests/test_hf_ar.py::analytic_params`）：

```
head_dim = hidden / num_heads

attn = h·h (q) + 2·(h·kv_h·head_dim) (k,v) + h·h (o) + 2·head_dim (QK-Norm)
mlp  = 3·h·inter            (SwiGLU: gate, up, down)
norm = 2·h                  (每层两个 RMSNorm)

params = vocab·h  +  layers·(attn + mlp + norm)  +  h
```

Tier 0 实测 **3,213,056**（embedding tied 只算一次）。
`state_dict()` 有 25 项但只存 24 个张量 —— `lm_head.weight` 和
`embed_tokens.weight` 共享存储。**存盘时必须去重**，safetensors 遇到共享存储直接报错。

## 1.4 seed 要在构造之前打

`Loop.__init__` 里的 `seed_all` 太晚了 —— 那时参数已经初始化完。
两次要对比的运行，必须在 `build_model()` **之前**设种子。

```python
seed_all(seed)
model = Qwen3ForCausalLM(build_qwen3_config(cfg))
```

A 版第一版就是漏了这一步，「同 seed 同权重」的检查立刻失败。

## 1.5 坑

| 坑 | 症状 |
|---|---|
| 忘了 `.to(device)` | 在 CPU 上跑，`torch.cuda.is_available()` 照样 True，看不出异常 |
| `Qwen3ForCausalLM(config)` 没 tie | 参数量对不上，C1 的 1e-5 变成 10.0 |
| 滑动窗口默认开 | mask 形状不同，logits 差 1e-2 起步 |
| 加载 `strict=True` 自己的 checkpoint | tied 模型必然报 `missing=['lm_head.weight']`，用 `ckpt.load_into()` |

## 1.6 验收命令

```sh
# A 版基准数
.venv/bin/python example/tests/test_hf_ar.py

# 你的实现写完后，加一条对拍（写进 test/ 或自己的脚本）
#   同 cfg -> 同权重 -> 同 batch -> logits max|Δ|
```

对拍脚本可以先抄 `example/common/verify.py::logits_delta` 的思路：
**两边都 `.eval()`、都 fp32、都跑同一批固定 token**。