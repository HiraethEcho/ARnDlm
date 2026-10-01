# example/ — 参考实现 (A 版, AI 写)

**这里的代码是 AI 写的。根目录下 B 版目录里的代码是你写的。**

## 为什么有两版

| | 位置 | 谁写 | 用途 |
|---|---|---|---|
| **A 版** | `example/` | AI | 用 `transformers` / `dllm` 包, 当**正确性基准** |
| **B 版** | `models/`、`training/`、`diffusion/` 等 | **你** | 同一功能, 自己实现 |

B 版的每个模块都应该在这里找得到对应物, **输出应该一致**。
不一致的地方就是你 B 版写错了。

## 状态

PENDING **还没有代码。** A 版分别在 T1a / T2a / T2b / T3 开工时写进来
（见 `PLAN.md`）。本文件先定契约。

## 为什么不把 A 版放在 B 版目录里

根目录已经有七条路线（`base` `utils` `transformer` `minimind` `a2d`
`dllm-a2d` `dllm`），但它们全部是 B 版。A 版只存在于这里。

`training/ar_hf/` 和 `training/a2d_dllm/` 看起来像「AI 写的参考实现」——
不是。B 版目录里的一切都是你写的，只是**用不用某个包**的区别：

| | 不用包 | 用包 |
|---|---|---|
| AR 训练 | `models/minimind_qwen/` + `training/ar_native/` | `models/hf_qwen/` + `training/ar_hf/` |
| A2D | `training/a2d_minimind/` | `training/a2d_dllm/` |

A/B 对照是**同一路线内部**的对照：用 `transformers` 写的和你手写的
`models/minimind_qwen/`，在同一 batch 同一权重下 logits 要对齐到 1e-5。

## 约束: transformers 4.57.0

环境钉 `4.57.0`（dllm 的 pin）。**不要写 5.x 的代码。**

`~/Projects/dlm/example/hf_dllm.py` 那份 A2D 参照是照 **5.17** 写的，
靠一个 5.x 才有的后门绕过因果 mask：

```python
for layer in model.model.layers:
    layer.self_attn.is_causal = False
BIDIR_MASK = {"full_attention": None}     # ← 5.x 的 Qwen3Model.forward 才认这个
```

4.57 里 `create_causal_mask` 的签名不同，这套不成立。移植到本项目时
按 4.57 的源码重写。参考：`dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py`
—— 它就是 4.57 版的 `modeling_qwen3.py` 改出来的。

## 对应表

PENDING 待 T1b / T2a 开工时填。列名固定为 `B 版模块` / `A 版参考` / `notes/`。

预期形状：

| B 版 (你写) | A 版 (参考) | notes/ |
|---|---|---|
| `minimind/model.py` | `transformers.Qwen3ForCausalLM` + 本项目的 config | `04-qwen-architecture.md` |
| `minimind/attention.py` | 不重写，直接用 `Qwen3Attention` | `04` |
| `a2d/corruption.py` | `dllm/core/trainers/mdlm.py` 的加噪段 | `03-minimind-dlm.md` §4 |
| `a2d/loss.py` | `dllm/core/trainers/mdlm.py:189-213` | `03` §4 |
| `a2d/sample.py` | `dllm/core/samplers/` | `03` §5 |
| `utils/metrics.py` | 重复率计算 | `03` §11 |
| `dllm/` 全套 | `dllm/` 包本身 | — |

## 注释约定

`example/` 的注释解释**设计意图和 transformers / dllm 源码机制**
（引源码行号那种），**但要精简** —— 只写「这行为什么这么写」和
「不这么写会怎样」，不写教科书段落。
