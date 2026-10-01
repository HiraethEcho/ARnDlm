# example/ — A 版参考实现

**这里的代码是 AI 写的。根目录下 B 版目录里的代码是你写的。**
执行计划见 [`PLAN.md`](PLAN.md)，实测数字见 [`RESULTS.md`](RESULTS.md)。

---

## A/B 的轴是「用包 vs 不用包」，不是「AI vs 人」

同一个功能（AR 训练 / A2D / 从零训 DLM）有**两版实现**：

| SPEC 2.3 行 | 用不用包 | A 版（AI 写） | B 版（你写） |
|---|---|---|---|
| Qwen-mini-HF | **用包** —— `transformers.Qwen3ForCausalLM` | `base/hf/` | `models/hf_qwen/` + `training/ar_hf/` |
| Qwen-mini-MiniMind | **不用包** —— 纯 `torch.nn` 手写 | `base/native/` | `models/minimind_qwen/` + `training/ar_native/` |
| Qwen3-0.6B | 现成 checkpoint + LoRA | `base/qwen3_0p6b/` | `training/a2d_dllm/` + `training/a2d_minimind/` |

三条规则：

1. **每一行、每一格都有 A 版和 B 版。** 不是「AI 写包版、你写手写版」——
   你手写的那份同样有 `transformers` 版和原生 torch 版。
2. **`example/` 覆盖矩阵全部 10 格**，包括行2 的纯 torch 实现。
   A 版不是「包的使用示例」，它本身是一套完整的、能跑出权重的实现。
3. **A/B 对照发生在行内**：行1 对 `models/hf_qwen/`，行2 对 `models/minimind_qwen/`。

行1 和行2 之间是对照**不是** A/B 对照——那是两条路线的横向比较：
它验的是「用包这条路的实现对不对」（`check_ab.py` 的 C1，logits `1e-5`）。
等你写完行2 之后，C1 同时也是行2 的 A/B 对拍。

---

## 目录结构

```
example/
  common/        A 版公用底座 (data · ckpt · metrics · seeds · config) —— 不碰根目录 utils/
  base/hf/       行1  train_ar.py · a2d_minimind.py · a2d_dllm.py · direct_dlm.py
  base/native/   行2  model.py (RoPE·RMSNorm·SwiGLU·GQA·QK-Norm·tied) + 同样四个
  base/qwen3_0p6b/  行3  lora_dllm_a2d.py · lora_minimind_a2d.py
  check_ab.py    跨版本对拍 (C1..C5)
  smoke_all.sh   按顺序跑完 10 格
```

三个 DLM 方法跨 base 复用：AR 训练产出权重，`a2d_dllm` / `a2d_minimind` /
`direct_dlm` 吃**任意** AR 权重（hf、native、0.6B 都行）。矩阵的两个轴是独立的。

## 状态

| | |
|---|---|
| 代码 | **已完成** —— `common/` + `base/{hf,native,qwen3_0p6b}/` + 8 个测试 + `check_ab.py` + `smoke_all.sh` |
| 实测 | **已完成** —— 10 格全部跑通，数字见 [`RESULTS.md`](RESULTS.md) |
| 对拍 | C1-C5 全绿（C1 6.91e-06 · C2 7.56e-08 · C3 1.54e-07 · C4 0.0 · C5 0/0/0） |
| 未决问题 | [`ISSUES.md`](ISSUES.md) 5 条 |
| 下一步 | 按 `teach/` 逐步写 B 版，从 step 0 开始 |

## 对照表（模块级）

| B 版模块（你写） | A 版参考（这里） | `notes/` |
|---|---|---|
| `models/hf_qwen/` | `base/hf/model.py`（`Qwen3Config` + `Qwen3ForCausalLM`） | `04-qwen-architecture.md` |
| `models/minimind_qwen/` | `base/native/model.py` | `04` |
| `training/ar_hf/` | `base/hf/train_ar.py` | — |
| `training/ar_native/` | `base/native/train_ar.py` | — |
| `training/a2d_minimind/` | `base/*/a2d_minimind.py` | `03-minimind-dlm.md` §4 §5 |
| `training/a2d_dllm/` | `base/*/a2d_dllm.py` | `03` §4 §5 |
| `training/direct_dlm/` | `base/*/direct_dlm.py` | `03` §4 |
| `diffusion/sampler.py` | `a2d_minimind.py` 里的采样段 / `dllm.core.samplers` | `03` §5 |
| `utils/metrics.py` | `common/metrics.py` | `03` §11 |
| `utils/ckpt.py` | `common/ckpt.py` | — |

验收阈值见 `SPEC.md` §6：T1b logits `1e-5`，T2a/T2b loss 差 `<1%`，
所有 DLM 路线必须报 repetition rate（`eval.repetition_ngram`，默认 4-gram）。

---

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

---

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

## 注释约定

`example/` 的注释解释**设计意图和 transformers / dllm 源码机制**
（引源码行号那种），**但要精简** —— 只写「这行为什么这么写」和
「不这么写会怎样」，不写教科书段落。