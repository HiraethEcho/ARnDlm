# example/ — A 版执行计划

**范围**: `example/` 覆盖 SPEC 2.3 实验矩阵**全部 10 格**, 在 Tier 0 极小配置下
把每条 pipeline 跑通。**不关心模型能力**, 只关心 pipeline 成立 + 代码正确 + 代码简洁。

跑完之后进入 B 版教学阶段 (见 §10)。

---

## 0. 这 10 格是哪 10 格

SPEC 2.3 的矩阵是 **3 个 base model × 3 种 DLM 方法**（Direct-DLM 在 0.6B 那格是
「—」，所以 3×3−1 = **8 格**），另外 **2 个 AR baseline** 是所有 DLM 格的前置。
合计 **10 格**，本次全部要跑通。

| # | 格子 | base 权重从哪来 | A 版文件（我写） | B 版目录（你写） | smoke 产物 |
|---|---|---|---|---|---|
| 1 | **AR baseline · 行1**（矩阵前置） | 随机初始化 | `base/hf/train_ar.py` | `models/hf_qwen/` + `training/ar_hf/` | `qwen-mini/hf-ar-example/` |
| 2 | **AR baseline · 行2**（矩阵前置） | 随机初始化 | `base/native/train_ar.py` | `models/minimind_qwen/` + `training/ar_native/` | `qwen-mini/minimind-ar-example/` |
| 3 | Qwen-mini-HF · **dLLM-A2D** | 格 1 的权重 | `base/hf/a2d_dllm.py` | `training/a2d_dllm/` | `qwen-mini/hf-dllm-a2d-example/` |
| 4 | Qwen-mini-HF · **MiniMind-A2D** | 格 1 的权重 | `base/hf/a2d_minimind.py` | `training/a2d_minimind/` | `qwen-mini/hf-minimind-a2d-example/` |
| 5 | Qwen-mini-HF · **Direct-DLM** | 随机初始化（**不吃 AR 权重**） | `base/hf/direct_dlm.py` | `training/direct_dlm/` | `qwen-mini/hf-direct-dlm-example/` |
| 6 | Qwen-mini-MiniMind · **dLLM-A2D** | 格 2 的权重 | `base/native/a2d_dllm.py` | `models/minimind_qwen/` + `training/a2d_dllm/` | `qwen-mini/native-dllm-a2d-example/` |
| 7 | Qwen-mini-MiniMind · **MiniMind-A2D** | 格 2 的权重 | `base/native/a2d_minimind.py` | `models/minimind_qwen/` + `training/a2d_minimind/` | `qwen-mini/native-minimind-a2d-example/` |
| 8 | Qwen-mini-MiniMind · **Direct-DLM** | 随机初始化（**不吃 AR 权重**） | `base/native/direct_dlm.py` | `models/minimind_qwen/` + `training/direct_dlm/` | `qwen-mini/native-direct-dlm-example/` |
| 9 | Qwen3-0.6B · **dLLM-A2D**（LoRA） | 已有 checkpoint + LoRA | `base/qwen3_0p6b/lora_dllm_a2d.py` | `training/a2d_dllm/` | `qwen3-0.6b/dllm-a2d-lora-example/` |
| 10 | Qwen3-0.6B · **MiniMind-A2D**（LoRA） | 已有 checkpoint + LoRA | `base/qwen3_0p6b/lora_minimind_a2d.py` | `training/a2d_minimind/` | `qwen3-0.6b/minimind-a2d-lora-example/` |

两个轴是**独立**的：base 决定权重从哪来，method 决定怎么转成 DLM。
所以三个 DLM 训练目录吃**任意** AR 权重，格 3-8 只是「换了个 base 再跑一遍」。

行1 的 A/B 对照对象是 `models/hf_qwen/`，行2 是 `models/minimind_qwen/` ——
**用不用包是这个项目里真正的 A/B 轴**，不是「谁写的」。

---

## 1. 非目标

- 不追求 loss 好看到能读懂
- 不做正式规模 (Tier 1/2) 训练, 不定参数量 —— 等 B 版跑完再定
- 不碰 `data/`、`utils/`、`models/`、`training/`、`diffusion/`、`evaluation/`
  (除注释)、`test/`、`configs/`
- 不动 `.venv`, 不装包, 不跑 `scripts/setup_env.sh`

## 2. 前置检查

| 项 | 期望 | 失败怎么办 |
|---|---|---|
| `.venv/bin/python -c "import torch, transformers, dllm"` | 4.57.0 / dllm import 通过 | 停下报告, 不自行重装 |
| `tokenizer/tokenizer.json` | vocab 6400 · MASK 27 · BOS 1 · EOS 2 · PAD 0 | `test/01-tokenizer-mask.py` |
| `data/prepared/tinystories-6400/` | tokens.bin · doc_lengths.npy · meta.json | `test/02-data-pipeline.py` |
| Qwen3-0.6B | `/mnt/extra/hf/hub/models--Qwen--Qwen3-0.6B` | HF_ENDPOINT=hf-mirror |
| GPU 空闲 | `rocm-smi --showuse` → use 接近 0% | 有别的负载就等, 不在共享卡上测吞吐 |
| device | 每个入口 `assert x.device.type == 'cuda'` | torch 默认在 CPU 上, 见 `logs/dev/01` |

## 3. 目录与文件清单

```
example/
  README.md            A/B 契约 · 完整对应表 · 状态 (开工时更新)
  PLAN.md              本文件
  RESULTS.md           smoke 实测数字 (10 格表)
  common/
    __init__.py
    config.py          Tier0 常量 + 路径常量 (tokenizer/data/ckpt/log)
    data.py            tokens.bin 内存映射 · doc_lengths · packing dataloader · 固定 seed
    ckpt.py            safetensors save/load · 训练状态 (step/optimizer/rng) · resume
    metrics.py         train-only + wall 双吞吐 · repetition rate (4-gram)
    seeds.py           seed / device / dtype 统一入口
  base/hf/             行1 base = transformers Qwen3
    model.py           Tier0 → Qwen3Config
    train_ar.py        AR 训练 (手写循环, 不用 Trainer)
    a2d_minimind.py    关因果 + 词表加 MASK 27 · 加噪 · 加权 CE · 分块采样器
    a2d_dllm.py        dllm.pipelines.a2d.convert + MDLMTrainer + MDLMSampler
    direct_dlm.py      随机初始化 + MDLMTrainer
  base/native/         行2 base = 纯 torch Qwen (不用包)
    model.py           RoPE · RMSNorm · SwiGLU · GQA · QK-Norm · tied embedding
    train_ar.py        AR 训练
    a2d_minimind.py    同 hf 版 (逻辑共用, 只换 base)
    a2d_dllm.py        同上
    direct_dlm.py      同上
  base/qwen3_0p6b/     行3 base = 已有 checkpoint + LoRA
    lora_dllm_a2d.py       r=8 LoRA 挂 attention 四投影, 冻结 backbone
    lora_minimind_a2d.py   同权重 + 手写 minimind A2D
  check_ab.py          跨版本对拍 (C1..C5)
  smoke_all.sh         按顺序跑完 10 格 + 汇总
```

预估 12 个 py 文件 + 1 个 sh, 合计 1500-2000 行。

## 4. smoke 配置

| 项 | 值 |
|---|---|
| 架构 | SPEC Tier 0: hidden 256 · 2 层 · 2 Q heads · 1 KV head · inter 768 · ctx 128 |
| head_dim | 128 (= 256 / 2) |
| vocab | 6400 (项目 tokenizer, **不加新 token**, MASK 复用 27) |
| 参数量 | 解析估算 ≈ 3.2M (embedding tied 1.64M + 2 层 × 0.79M)。以实测为准 |
| dtype / attn | bf16 / sdpa |
| optimizer | AdamW · lr 3e-4 (AR) / 1e-4 (A2D/LoRA) · cosine · warmup 20 |
| batch / grad_accum | 8 / 1 |
| steps | **200** = 204,800 tok |
| 数据 | `tinystories-6400` 全量 3,110,358 tok, packing 定长块零 padding, 只吃前 ~205k |
| 采样 prompt | `Once upon a time` / `The future of AI is` |
| 采样参数 | AR: top_k 50 · temp 0.3。DLM: block_size 8 · steps 8 · temp 0.3 · top_k 50 · cfg 0.0 |
| T2c 额外 | seq 128 · batch 2 · **20 步** · LoRA r=8 · alpha 16 · attention 四投影 |

**步数封顶不用 epoch** —— smoke 的目的是跑通, 不是训到收敛。

## 5. 执行顺序

| # | 做什么 | 产出 checkpoint | 验收 |
|---|---|---|---|
| 1 | `common/` 六个模块 | — | dataloader 出 batch, 数值合理 |
| 2 | `base/hf/model.py` + `train_ar.py` | `qwen-mini/hf-ar-example` | 6 项 + C3 基线 |
| 3 | `base/native/model.py` + `train_ar.py` | `qwen-mini/minimind-ar-example` | 6 项 + **C1** |
| 4 | `base/hf/a2d_minimind.py` | `qwen-mini/hf-minimind-a2d-example` | 6 项 |
| 5 | `base/hf/a2d_dllm.py` | `qwen-mini/hf-dllm-a2d-example` | 6 项 + **C2** |
| 6 | `base/hf/direct_dlm.py` | `qwen-mini/hf-direct-dlm-example` | 6 项 |
| 7 | `base/native/a2d_minimind.py` | `qwen-mini/native-minimind-a2d-example` | 6 项 + **C3** |
| 8 | `base/native/a2d_dllm.py` | `qwen-mini/native-dllm-a2d-example` | 6 项 |
| 9 | `base/native/direct_dlm.py` | `qwen-mini/native-direct-dlm-example` | 6 项 |
| 10 | `base/qwen3_0p6b/lora_dllm_a2d.py` | `qwen3-0.6b/dllm-a2d-lora-example` | 6 项 |
| 11 | `base/qwen3_0p6b/lora_minimind_a2d.py` | `qwen3-0.6b/minimind-a2d-lora-example` | 6 项 |
| 12 | `smoke_all.sh` + `RESULTS.md` | — | 表填满 |

## 6. 每格验收 6 项

1. backward 不出 NaN/Inf, loss 全程有限
2. 前 20 步 loss 均值 vs 后 20 步 loss 均值 —— **只记录, 不设阈值** (smoke 不看能力)
3. ckpt 保存 → **新进程**加载 → 同 batch logits 逐元素一致 (fp32 CPU, `max|Δ| ≤ 1e-6`)
4. 能采样: AR 出 32 token; DLM 出 block_size×steps 的 token
5. **repetition rate (4-gram)** —— 核心指标, 不许只报 loss
6. **train-only + wall 两个 tok/s** —— 验证只包住 fwd+bwd+step, 采样/验证不污染分母
   (方法论 `notes/05-measurement.md` Bug 1/2)

## 7. 跨格对拍

| # | 对什么 | 阈值 | 抓什么 bug |
|---|---|---|---|
| C1 | 行1-hf vs 行2-native, 同权重同 batch logits | `max|Δ| ≤ 1e-5` | RoPE 位置 / GQA 复制 / QK-Norm / tied embedding |
| C2 | 行1 `a2d_minimind` vs `a2d_dllm`, 同 ckpt 同 seed 同 batch loss | 相对差 `< 1%` | 加权 CE 分母 · p_mask 形状 · 采样 t 分布 |
| C3 | 行2 同 C2 | `< 1%` | 同上 (base 换了) |
| C4 | ckpt resume: 一次跑 220 步 vs 200+20 续训 | 末 loss `1e-6` | optimizer state / RNG / step 计数漏存 |
| C5 | dllm convert 前后 | `missing == unexpected == []` | A2D config 与 4.57 `create_causal_mask` 对不上 |

**等你写完 B 版**, 同一个 `check_ab.py` 增加两列指向 `models/hf_qwen/` 与
`models/minimind_qwen/`。B 版模块不存在时打印 `SKIP`, 不静默通过。

## 8. checkpoint 路径

```
checkpoints/example/
  qwen-mini/
    hf-ar-example/                     minimind-ar-example/
    hf-dllm-a2d-example/               native-dllm-a2d-example/
    hf-minimind-a2d-example/           native-minimind-a2d-example/
    hf-direct-dlm-example/             native-direct-dlm-example/
  qwen3-0.6b/
    dllm-a2d-lora-example/             minimind-a2d-lora-example/
```

每个目录: `config.json` + `model.safetensors` (+ `adapter_model.safetensors` for LoRA)
+ `train_state.pt` (step/optimizer/rng) + `meta.json` (参数量/步数/loss 首末/吞吐)。

## 9. 耗时预算

| 阶段 | 估计 |
|---|---|
| 写 `common/` + 行1 四个 | 50 min |
| 写 `native/model.py` (最容易错的部分) + 行2 四个 | 60 min |
| 写 T2c 两格 (含 dllm 4.57 差异排查) | 40 min |
| 10 格 smoke 跑 (AR 4 格 ~30 s/格, DLM 6 格 ~2 min/格, 0.6B 两格 ~15 min) | 40 min |
| `check_ab.py` + `RESULTS.md` | 20 min |
| **合计** | **约 3.5 h** |

单步 ms 极小 (3.2M 参数 · seq 128), 跑测不构成超时风险; 真正的瓶颈是写代码。

**实际用时**：2026-10-01 20:30 → 23:05, 共 **2h35m** (含全部 commit),
比预估的 3.5 h 短。写代码比预估快, 测试比预估慢 —— 每个格子都抓到了真 bug
(BOS/EOS 偏移、resume 丢批次位置、reveal 索引错行、decode 计时无预热、
LoRA 行的 `bos_token_id is None`)。

## 10. 教学路线图 (A 版跑通后开始)

按依赖顺序, 每步你写、我审、我跑对拍:

| Step | 你写什么 | 对拍对象 | 通过标准 |
|---|---|---|---|
| 0 | `utils/` (logger · device · seed · ckpt · metrics) | `example/common/` | 行为一致, 双吞吐都报 |
| 1 | `models/hf_qwen/` | `example/base/hf/model.py` | 同 config 同权重同 batch |
| 2 | `training/ar_hf/` | `example/base/hf/train_ar.py` | 同 seed 同 batch 逐 token 一致 |
| 3 | `models/minimind_qwen/` | `example/base/native/model.py` | **C1 `1e-5`** |
| 4 | `training/ar_native/` | `example/base/native/train_ar.py` | loss 一致 |
| 5 | `diffusion/corruption.py` + `scheduler.py` | `example/base/*/a2d_minimind.py` | 同 t 同 seed mask 位置一致 |
| 6 | `diffusion/loss.py` | `a2d_minimind` + `test/04` | **C2 `< 1%`** |
| 7 | `diffusion/sampler.py` | 同上 | 同 seed 同参数输出 token 逐个相同 |
| 8 | `training/a2d_minimind/` | 同上 | 端到端 loss 一致 |
| 9 | `training/a2d_dllm/` | `example/base/*/a2d_dllm.py` | dllm 路径接得上 |
| 10 | `training/direct_dlm/` | 同上 | — |
| 11 | `evaluation/` (generation + repetition rate) | `example/common/metrics.py` | 重复率数值一致 |
| 12 | Qwen3-0.6B LoRA 两格 | `example/base/qwen3_0p6b/` | 可训参数占比一致 |
| 13 | Tier 1 正式规模定档 | `logs/dev/02` 吞吐基线 | 先测吞吐再定参数量 |

**Step 3 是整个项目的核心** —— `1e-5` 对齐会把 RoPE 位置、GQA 头复制、
QK-Norm 位置、tied embedding 这四类 bug 全部逼出来。

## 11. 风险

| 风险 | 应对 |
|---|---|
| 0.6B LoRA 20 步超时 | 降到 5 步, 或先 tiny 代理验代码路径 |
| dllm A2D 与 4.57 签名不一致 | patch 已打 (`scripts/patches/`), `C5` 抓回归 |
| native 实现细节错 | `C1` 抓; 抓不到就逐层对拍 (embedding → norm → attn → mlp) |
| GPU 被抢 | 测吞吐前 `rocm-smi --showuse`; 抢卡时只跑功能不报吞吐 |
| packing 跨文档污染 | 本轮不管 (与 MiniMind 做法一致), 但 RESULTS.md 要写明未验 |
| `nohup` 长任务失联 | 按 AGENTS.md: 非阻断 subagent 监督 python 进程 |
