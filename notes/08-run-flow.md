# 训练脚本运行流程：从 `python training/ar_hf/train.py` 到一次权重更新

适用：实践版 ar-hf 格子（`smoketest` 档，hidden 512 / 2 层 / ctx 128 / vocab 6400）。
代码位置全部是实践版路径；对照实现在括号里标注。

引用格式 `文件::符号`。

---

## 一、启动（Python 解释器加载，还没碰 GPU）

```
.venv/bin/python training/ar_hf/train.py
```

| 步骤 | 发生什么 | 资源 |
|---|---|---|
| 1 | `train.py` 把**仓库根**插进 `sys.path`，然后 import 各模块 | 纯 CPU |
| 2 | import `utils.config` → `utils/version.py::parse_yaml` 读 `configs/practical/smoketest.yaml` → 构造 `MODEL` (ModelCfg) / `AR_TRAIN` (TrainCfg) / 常量 | 纯 CPU，读 ~1 KB yaml |
| 3 | import `models/hf_qwen/model.py` → `utils/tokenizer.py` 读 `tokenizer/tokenizer.json`（6400 词条 BPE，一次性解析后缓存） | CPU，内存 ~几 MB |

**此刻显存占用 0** —— GPU 还没被碰过。

---

## 二、main() 里的四件事

```
main()
 ├─ 1  data      = build_packed(...)        # 语料 → 定长块
 ├─ 2  model     = build_model(...)         # Qwen3ForCausalLM → GPU
 ├─ 3  loop      = Loop(model, cfg, ...)    # 优化器、RNG、搬权重
 ├─ 4  summary   = loop.fit(...)            # 训练 + 验证 + 采样
 └─ 5  loop.save(...)                       # checkpoint + 日志
```

### 1. 读语料 `utils/data.py::build_packed`

| 动作 | 在哪 | 内存 |
|---|---|---|
| `tokens.bin` 用 `np.memmap(dtype="<u2")` 映射 | **磁盘**，按页懒读 | 虚拟地址空间，不占物理 RAM |
| `_stream()` 拼成 `[BOS] doc [EOS] ...` 一维 int32 流 | CPU/RAM | ≈ 2× 语料字节数（minimind3-6400 全量 ~0.7 GB int32；`max_blocks` 截断后只留前段） |
| 切成 `[n_blocks, 128]` 的 int32 张量、拷贝 | RAM | 4096 块 × 512 B ≈ 2 MB（smoketest 的 max_blocks） |
| 四道自洽检查（对账 / vocab / 越界 / MASK） | CPU | 可忽略 |
| `fixed_batches()` 给验证用的一组冻结索引 | RAM | 可忽略 |

### 2. 建模型 `models/hf_qwen/model.py::build_model`

| 动作 | 在哪 |
|---|---|
| `seed_all(seed)`（torch/numpy/random 三处） | CPU |
| `Qwen3ForCausalLM(config)` 初始化全部权重 | 先 CPU 生成，随 Loop 上 GPU |
| 权重 fp32，≈ **6.1M 参数 ≈ 24 MB** | — |

**种子在这时打** —— 在 Loop 里打就晚了，权重已经初始化完。

### 3. 构造 Loop `utils/loop.py::Loop.__init__`

| 动作 | 在哪 | 内存 |
|---|---|---|
| `model.to(cuda)` | CPU → GPU 一次拷贝 (24 MB) | 显存 24 MB |
| `_optimizer(model, cfg)` 建 AdamW 参数组（按 `id()` 去重 tied） | GPU 上暂无 state | 0 |
| `seed_all(cfg.seed)` 再打一次（保险） | CPU | — |

AdamW 的 state (`exp_avg` + `exp_avg_sq`) 在**第一次 `opt.step()` 时才分配**：2 × 参数量 × 4 B ≈ 46 MB。

### 4. 训练 `utils/loop.py::Loop.fit` —— 每一步的完整流水

```
while step < limit:
  ① stream.next()            CPU    取一批块索引 (RAM 里的 permutation)
  ② data.batch(idx).to(cuda) CPU→GPU  H2D 拷贝 8×128×8B = 8 KB, 非阻塞
  ③ _train_step(batch)
     ┌ synchronize()         GPU    排空队列, 计时开始
     ├ autocast(bf16):
     │   loss_fn(model, batch)  GPU    forward: 全部 matmul 走 bf16
     ├ loss.backward()         GPU    反向, 注意力 backward 是瓶颈 kernel
     ├ clip_grad_norm_         GPU
     ├ opt.step()              GPU    AdamW 更新; 此刻分配 exp_avg/exp_avg_sq
     ├ opt.zero_grad()         GPU    梯度置 none (引用释放)
     └ synchronize(); 计时结束  GPU    这段秒数 → meter.add (train-only 分母)
  ④ 记 loss / lr / 双口径 tps 进 self.rows
```

**每步显存账（稳态）**：

| 项 | 大小 | 说明 |
|---|---|---|
| 权重 fp32 | 24 MB | master 一直是 fp32 |
| 梯度 fp32 | 24 MB | `backward` 时分配 |
| AdamW state | 48 MB | `exp_avg` + `exp_avg_sq`，各 1×24 MB |
| autocast 的 bf16 副本 | 峰值时临时 | matmul 输入/输出的 bf16 副本，用完即释放 |
| 激活 (B=8, L=128) | <10 MB | 小模型小上下文，反传需要的中间量按层释放 |
| **合计** | **< 1 GB** | 实测基准 (Tier 0 同形状) 0.214 GB |

**CPU 在每步做的事**：取索引、`.to(device)` 发起 H2D、Python 循环本身。GPU 做全部 matmul/attention/AdamW。瓶颈在 attention 的 backward kernel（本机实测，见 `logs/dev/02-hardware-perf.md`）。

**步间磁盘活动：0。** 语料已整个在 RAM；checkpoint 只在 `fit()` 结束后写。

### 5. 训练结束后（仍在 `fit()` 内）

| 动作 | 计入哪个口径 | 显存 |
|---|---|---|
| RNG 快照 (`rng_state`) | 都不计 | CPU |
| `eval_loss`：冻结批次、**fp32**、no_grad 前向 | wall ✔ / train-only ✘ | 与训练同量级 |
| `sample_ar`：`model.generate()`，自回归 32 token × 2 prompt | wall ✔ / train-only ✘ | KV cache 少量 |
| peak GB 读取 | 不计 | — |

**这就是两个口径的差值的物理含义**：wall_tps 比 train_only_tps 低的那部分 = 验证 + 采样 + 日志的时间。

### 6. `Loop.save`：checkpoint 落盘

```
checkpoints/ar-512-smoke/
  model.safetensors    权重 (tied 去重后 24 张量), GPU→CPU 拷贝后写盘
  config.json          ModelCfg/TrainCfg 快照
  meta.json            参数普查 / loss / 吞吐
  train_state.pt       step + optimizer state + 流位置 (epoch,pos) + RNG 快照
```

同时 `logs/run/ar-512-smoke/` 写 `summary.json` + `metrics.csv`。

---

## 三、SM 的对应关系速查（写循环时会来回查）

| 你在 train.py 看到 | 实际函数 | 文件 |
|---|---|---|
| 「读语料」 | `build_packed` / `_stream` / `_load` | utils/data.py |
| 「取一批」 | `BlockStream.next` → `PackedData.batch` | utils/data.py |
| 「建模型」 | `build_model` (seed→config→Qwen3ForCausalLM) | models/hf_qwen/model.py |
| 「算 loss」 | `ar_loss`（移位 + CE） | training/ar_hf/loss.py |
| 「更新一步」 | `Loop._train_step`（计时区间夹 forward/backward/step） | utils/loop.py |
| 「LR 调度」 | `_lr_at`（warmup 线性 → cosine） | utils/loop.py |
| 「验证」 | `Loop.eval_loss`（fp32、冻结批次） | utils/loop.py |
| 「存盘」 | `Loop.save` → checkpoint.py 三件套 + train_state | utils/checkpoint.py |
| 「吞吐/复读率」 | `TpsMeter` / `repetition_rates` | utils/metrics.py |
| 「采样」 | `sample_ar`（generate） | models/hf_qwen/model.py |

---

## 四、还没写/延后的部分

| 文件 | 状态 | 不在时的行为 |
|---|---|---|
| `utils/cli.py` | 延后 | smoke 参数钉在 `train.py` 的 argparse 里 |
| `utils/logger.py` | 延后 | `summary.json`/`metrics.csv` 由 `loop.py::_write_logs` 直接写 |
| `test/06` 实现 | 待写 | 两版解析器的一致性还没被断言（现在靠手测） |
| `training/ar_hf/sample.py`、`sft.py`、`lora.py` | B2/C 阶段 | 不影响本格 |
