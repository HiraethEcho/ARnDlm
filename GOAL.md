# GOAL — 教我写实践版

**创建**：2026-10-03　**状态**：进行中
**前置**：旧 GOAL（实验版全流程）已完成——SPEC 2.3 矩阵 29 格全绿，
G3 结论「原路线优 0.4251 nat、四对全同向」，见 `example/RESULTS.md`。

---

## 1. 目标

我（用户）在 `learn` 工作区**亲手写实践版**，你（AI）当导师：

| 你的角色 | 做什么 |
|---|---|
| **教** | 每一步先讲「要写什么、为什么、坑在哪」，再让我动手 |
| **改我代码里的 bug** | 直接改，改完**解释为什么** |
| **查实验版源码的 bug** | 静态检查为主；发现 bug 直接修并记录 |
| **不跑训练** | **不主动跑任何训练/测试**。除非 (a) 我主动要求，或 (b) 要做正式训练 |

**不改**：`example/`（那是参考答案）。你只改**我的**代码和文档。
不 push。

---

## 2. 工作区

```
~/Projects/ARnDlm         main     ← 参考答案，example/ 全部跑通
~/Projects/ARnDlm-learn   learn    ← 我写代码的地方
```

learn 里软链了 `.venv` / `data` / `checkpoints`（与 main 共用）。
**注意**：两边共用同一份语料和 checkpoint 目录——我重建 SFT 语料会覆盖
main 的那份。

---

## 3. 我的路径（已定，不重开）

```
ar-hf  →  ar-hf-sft  →  ar-hf-sft-dllm_a2d
  A0 → A1 → A2        B1 → B2        C1 → C2 → C3 → C4
```

| 步骤 | 我写 | 参考实现 | 对应格 |
|---:|---|---|---|
| A0 | `utils/`（训练循环/计时/存盘/指标） | `example/common/{loop,data,ckpt,metrics,seeds}.py` | 底座 |
| A1 | `models/hf_qwen/` | `example/base/hf/model.py` | `ar-hf` |
| A2 | `training/ar_hf/` | `example/base/hf/train_ar.py` | `ar-hf` |
| B1 | `data/sft/`（对话→块+assistant spans） | `example/common/sft.py` | SFT 数据 |
| B2 | `training/sft_ar_hf/` + `training/sft_a2d_hf/` | `example/base/hf/{train_sft,train_dlm_sft}.py` | `ar-hf-sft` |
| C1 | `diffusion/corruption.py` | `example/common/a2d.py::corrupt` | 方法 |
| C2 | `diffusion/loss.py` | `example/common/a2d.py::a2d_loss` | 方法 |
| C3 | `diffusion/sampler.py` | `example/common/a2d.py::sample_dlm` | 方法 |
| C4 | `training/a2d_dllm/` | `example/base/hf/train_a2d_from_sft.py` | `ar-hf-sft-dllm_a2d` |

每一步的详细说明在 `example/teach/README.md` → 对应文件。

---

## 4. 每一步怎么走

固定流程，不加东西：

1. **讲**：我把该写的接口/坑先讲一遍（基于 `example/teach/` 对应文件，
   不复制代码给你）。
2. **我写**：你在 learn 里写实现。**只能写 Python 代码，不复制 example**。
3. **你查**：我**静态检查**你的代码（读 + 对比参考实现），列出 bug 和偏差。
   改我的 bug 时**解释为什么**。
4. **查实验版**：同一轮我顺带静态检查参考实现里相关的部分，有 bug 一起修。
5. **commit**：每完成一步 commit 一次（在 learn 分支）。不 push。
6. **对拍**：只有你**主动要求**时才跑训练对拍；否则我把「验收命令写在这里」，
   由你决定何时跑。

---

## 5. 静态检查清单（第 3/4 步每次都过一遍）

| # | 检查什么 | 为什么 |
|---|---|---|
| 1 | `batch.dim() == 3` 的分支是否都覆盖 `[2,B,L]` | 这是 SFT 的 batch 形状，漏一个分支就是静默错 |
| 2 | span 的 `end` 是否含 EOS | assistant 的 EOS 是教模型停止的唯一一票 |
| 3 | 采样 prompt 是否与训练语料**逐 token** 一致 | turn 用 EOS 分隔，不用换行 |
| 4 | resume 是否恢复 optimizer / RNG / stream position | 否则续训不是同一条曲线 |
| 5 | eval 是否在 fp32、是否用冻结 batch | bf16 舍入差比要测的差异大 |
| 6 | tied embedding 是否按 `id()` 去重 | 否则等效 lr 翻倍 |
| 7 | 死代码 / 未用 import / 未用参数 | 一个没人检查的参数是邀请静默错 |
| 8 | 注释与代码是否矛盾 | 注释比代码先失效，而它看起来还对 |

**已知不用再报的**（实验版已接受）：OPEN-4 packing 无 attention 隔离、
OPEN-6 rep-4 不能排名、ctx 128 下 think 存活率 5.5%。

---

## 6. 「跑测试」什么时候可以发生

| 情况 | 做不做 |
|---|---|
| 我（用户）没说 | **不跑** |
| 我说「跑一下」 | 跑 |
| 完成某一步、要正式训练验证 | 跑，但先告诉我预计时长 |
| 你想自己确认代码对 | **用静态检查，不跑** |

正式训练 = 完整跑 29 格那一级别的运行。20 步冒烟也算训练，同样需要我点头。

---

## 7. 规模

实践版和实验版**同规格起步**：Tier 0（3.21M 参数 · ctx 128 · 200 步）。
等 A2 跑通、对拍 C1/C7 都过了，再讨论上不放大的问题
（`configs/practical/tier1.yaml` 仍是空的）。

---

## 8. 记录

- 每一步的结论写进 `logs/dev/01-local-state.md` 的「项目当前进度」
- 你修的 bug 记进 `example/ISSUES.md`（实验版）或本文件（我的代码）
- **不在对话里留结论，留在文件里**
