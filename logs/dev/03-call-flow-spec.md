# 代码调用流程 —— 按 SPEC 现版文件设计（P0 视角的设计记录）

骨架尚未填实现（P1–P3 未开工），本文描述**设计意图**：python 启动一个训练格子时，
按 SPEC §7 的文件依次调用什么、干什么。实现落成后如与本文不符，改实现，改前先改这
份（它比实现先动）。

> smoke 级模型配置：3.2M（h256 L2 ctx 128 B8，即 `configs/smoketest.yaml`）。
> smoke 只验证代码正确（对拍 + loss 能降），不复用 legacy 的吞吐数字 —— 那是
> 实验版在旧机器口径下的实测，与性能无关，smoke 本来也不看性能。

## 启动：`python -m src.training.main --tier smoketest --impl trans --steps 20`

```
解释器
  └ import src.training.main
      ├ import common.config    # pyyaml 读 configs/<tier>.yaml → frozen dataclass(MODEL/TRAIN) + 常量
      ├ import common.seed      # 不执行，等调用
      ├ import common.device    # 定义 _DTYPES / pick_device，不执行
      ├ import src.models.trans # transformers 只在此 import（Qwen3ForCausalLM）
      └ import src.training.ar_trans  # Trainer / TrainingArguments 在此 import
```

环境前提：`HF_ENDPOINT=https://hf-mirror.com`（transformers 首次拉 token 时才碰网）。

## main.py 依次做的七件事

| # | 调用 | 干什么 | 资源 |
|---|---|---|---|
| 1 | `common.device.pick_device` | 选 cuda 并打印设备名 | 显存仍 0 |
| 2 | `common.config`（已在 import 时读完） | 取 MODEL / TRAIN / BLOCK_SIZE / DTYPE | CPU |
| 3 | `common.data.build_packed(C.DATA_DIR，BLOCK_SIZE)` | memmap tokens.bin → [BOS]doc[EOS] 流 → int32 定长块，四道自检 | 磁盘→RAM 一次性 |
| 4 | `seed_all` → `src.models.trans.build(cfg)` **或** `src.models.native.build(cfg)` | Config 显式字段 → 建模；权重先在 CPU | CPU |
| 5 | impl 分支开始：见下两条路线 | 权重 `.to(cuda)` | H2D 一次 |
| 6 | 训练循环（每步） | 见「每步流水」 | GPU 全部 |
| 7 | 收尾：eval_loss(fp32) → sample（common.tokenizer decode）→ 存盘 | 验证与采样不进 train-only 计时 | GPU→CPU→磁盘 |

## 两条 impl 路线的差别（第 5-6 步）

**trans**（`ar_trans.py`）：
- 打包块包成 Trainer 可吃的 dataset（无 padding，块即样本）
- `TrainingArguments`：max_steps / warmup / cosine / bf16 全部来自 config，不出现第二处数字
- `Trainer.train()` 内部自管 DataLoader 与调度；日志与存盘点被 Trainer 钩住
- 吞吐用 TrainerCallback 自记（Trainer 不暴露单步秒数，双口径只能近似）

**native**（`ar_native.py`，native 包自带的最小循环）：
- 自管逐步：`stream.next() → data.batch(idx).to(cuda) → autocast(bf16) → loss.backward() → clip → step → zero_grad`
- `torch.cuda.synchronize()` 两端夹住计时 → `common.metrics.TpsMeter`（train-only 精确口径）
- 逐步 loss 记进 metrics.csv；scheduler 用 common 层的 warmup+cosine（同一公式）

两条路线**同一 main、同一 config、同一 metrics、同一存盘布局**，逐位可比就是
trans/native 各自不掺私货换来的。

## 每步流水（native 视角，smoke 尺寸 ~19 ms/step 稳态）

```
① stream.next()              CPU  取一批块索引（epoch 与 permutation 由 seed+step 推回）
② data.batch(idx).to(cuda)   H2D  8×128×8B ≈ 8KB 非阻塞
③ autocast(bf16): forward    GPU  matmul 走 bf16，master 权重保持 fp32
④ loss.backward()            GPU  瓶颈在 attention backward kernel（legacy/02）
⑤ clip_grad_norm_ + step()   GPU  AdamW state 首步时才分配（2×4B/参）
⑥ zero_grad(set_to_none)     GPU
⑦ synchronize + meter.add    GPU  这段秒数只进 train-only 分母
```

显存账（tier1 实测 1.41 GB 峰值；smoke 尺寸远小于此）：权重 + 梯度 + AdamW
state ≈ 4× 参数字节数，加激活与碎片。

## 收尾

```
tokenizer.decode(sample_ids)   → summary["sample"]["texts"]
state.py 存 train_state.pt     → step / optimizer / 批次位置 / 训练结束时的 RNG 快照
common.metrics.as_dict()       → logs/run/<格子名>/summary.json（唯一真相）+ metrics.csv
```

## 本文档留下的待验证点（实现时逐条消掉）

1. trans 路线的逐位 resume（Trainer 的 checkpoint 与我们的 train_state.pt 关系）
2. Trainer 的 bf16 开关与我们 device.autocast 的语义是否一致（Trainer 用 accelerate 管 AMP）
3. TrainerCallback 计时的近似口径要到哪一步（初步：只报 wall + 回调近似的 train-only）
