# PLAN

计划可变，契约不变（SPEC.md）：验收阈值/C1-C5 在 SPEC §6，测量口径在 §6.4 + notes/05。

## 双文件规矩

| 文件 | 谁写 | 内容 |
|---|---|---|
| `x.py` | 你 | 骨架：只有「这个文件是干什么的」注释，逐条填代码 |
| `x-example.py` | AI | 正确的参考实现，每个代码块一行功能注释；验收时跑它出基准行为 |

骨架填完对拍同目录 `-example`（同 seed 同输入逐位一致），对拍通过才算完。

## 目录

见 SPEC §7。一句版：`common/` 底座（模型无关），`src/` 格子（models / training / eval），
`scripts/` 环境，`configs/` 两份 yaml，`/mnt/extra` 软链放大数据与 checkpoints。

## 主干接口（换实现不改主代码）

`src/training/main.py` 只认一个 `build(cfg) -> (model, loss, sample)` 三元组：

- `trans`：transformers `Qwen3ForCausalLM` + Trainer 路线（模型层 + 训练层都尽量用包）
- `native`：`src/models/native/` 纯 torch 包（接口对齐 trans）
- G1 = 同一 main 跑两个 build；SFT/A2D 同理只换 build

## 命名（SPEC §1.2）

- tier1 权重名用实测参数量：`ar-26.9m-trans` / `ar-26.9m-native`（26,879,488）
- smoke checkpoint 带 `-smoke` 后缀：`checkpoints/ar-3.2m-trans-smoke/`
- `impl` 段：`trans` / `native`；`-` 分段，`_` 只在「实现×阶段」内部（`mini_a2d`）

## 阶段

| Phase | 内容 | 验收（SPEC 条款） |
|---|---|---|
| **P0 文档与骨架** | SPEC + PLAN + 目录树 + 骨架注释 | 结构 vs §7 一致 |
| **P1 common/** | config（pyyaml）/ seed / device / tokenizer / data / prepare / metrics | yaml 浮点为 float；同 seed 三路一致；`_stream` 手工例 |
| **P2 models** | `trans.py` + `native/` 包 | **C1**：logits max\|Δ\| ≤ 1e-5（锚 6.91e-06） |
| **P3 smoke 训练** | `main.py` + `ar_trans.py` + `ar_native.py` + `state.py` | 20 步跑通；双口径 tok/s；checkpoint 名合 §1.2 |
| **P4 数据** | prepare.py 跑通（minimind3-6400 已在 /mnt/extra） | vocab 6400；MASK=27 不在语料 |
| **P5 SFT** | `sft.py`（assistant 区间 loss） | T4a/T4b 对拍；§6.3 四门槛全报 |
| **P6 A2D + DLM** | `a2d_dllm.py`（dllm 包）+ `a2d_mini.py`（手写）+ `dlm.py` | **C2/C3** < 1% 同 corruption；**C5** 转换全 0；rep-4 必报 |
| **P7 tier1** | `ar-26.9m-*` 正式对照 + G4 qwen3-0.6b LoRA | G2/G3 逐格结论进 logs/dev |

依赖顺序：P1 → P2 → P3/P4 可并行 → P5 → P6 → P7。
每个 Phase 结束更新 `logs/dev/01-local-state.md` 的进度段。

## 尺寸基准（实测，勿重测）

smoke 3.2M（解析值 3,213,056）；tier1 26.9M = 7,995 train-only tok/s @ B8/512，
332M token ≈ 11.6 h，峰值 1.41 GB。细节在 `logs/dev/02-hardware-perf.md`。
