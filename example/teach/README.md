# example/teach/ — 实践版教学计划

**这里是你（实践版）要写的东西的施工图。**

## 你的学习路径

```
ar-hf  →  ar-hf-sft  →  ar-hf-sft-dllm_a2d
  A0 → A1 → A2        B1 → B2        C1 → C2 → C3 → C4
```

| 顺序 | 文件 | 你写什么 | 对应矩阵格 |
|---:|---|---|---|
| 1 | [`A0` = `00-common-base.md`](00-common-base.md) | `utils/`：训练循环、计时、存盘、指标 | 底座 |
| 2 | [`A1` = `01-hf-qwen.md`](01-hf-qwen.md) | `models/hf_qwen/`：加载并冻结 Qwen3 | 行1 base |
| 3 | [`A2` = `02-ar-hf.md`](02-ar-hf.md) | `training/ar_hf/`：自回归训练 | **`ar-hf`** ✅ |
| 4 | [`B1` = `14-sft-data.md`](14-sft-data.md) | `data/sft/`：对话 → 块 + assistant spans（含 EOS） | SFT 数据 |
| 5 | [`B2` = `15-sft-train.md`](15-sft-train.md) | `training/sft_ar_hf/`：assistant-only loss | **`ar-hf-sft`** ✅ |
| 6 | [`C1` = `05-corruption-scheduler.md`](05-corruption-scheduler.md) | `diffusion/corruption.py`：掩码 + 时间步 | 三格方法 |
| 7 | [`C2` = `06-diffusion-loss.md`](06-diffusion-loss.md) | `diffusion/loss.py`：A2D 目标 | 三格方法 |
| 8 | [`C3` = `07-diffusion-sampler.md`](07-diffusion-sampler.md) | `diffusion/sampler.py`：迭代去噪 | 三格方法 |
| 9 | [`C4` = `09-a2d-dllm.md`](09-a2d-dllm.md) | `training/a2d_dllm/`：调 dllm 包 | **`ar-hf-sft-dllm_a2d`** ✅ |

走完 9 个文件，第 3 步那三格矩阵就齐了。每格在 `RESULTS.md` 里都有实测数字
可对。

## 后面再看（不在你的路径上）

| 文件 | 你写什么 | 什么时候需要 |
|---|---|---|
| [`03-minimind-qwen.md`](03-minimind-qwen.md) | `models/minimind_qwen/` | 想跑 G1 的第 2 行 |
| [`04-ar-native.md`](04-ar-native.md) | `training/ar_native/` | 同上 |
| [`08-a2d-minimind.md`](08-a2d-minimind.md) | `training/a2d_minimind/` | 想手写 A2D 而不是调 dllm 包 |
| [`10-direct-dlm.md`](10-direct-dlm.md) | `training/direct_dlm/` | 想看 G2 |
| [`11-evaluation.md`](11-evaluation.md) | `evaluation/` | 任何时候 |
| [`12-qwen3-lora.md`](12-qwen3-lora.md) | 行3 LoRA 两格 | 想跑 0.6B 真模型 |
| [`13-tier1.md`](13-tier1.md) | Tier 1 定档 | 准备上正式规模 |

## 怎么用

1. 按上面的顺序读。**每一步都建立在前面所有步之上**，跳步会缺东西。
2. 自己写 实践版，**不要复制 实验版代码** —— 复制过来就没有对照意义了。
   实践版目录里**只能写注释/docstring**，不能写实现。
3. 跑该节的验收命令。**两个实现不一致的地方就是你写错了的地方**——
   这正是每个验收命令都是「跑 实验版 + 跑你的 + 比」的原因。

## 通则（每一步都适用）

| 规则 | 出处 |
|---|---|
| 同时报 train-only 和 wall-clock 两个吞吐 | `notes/05-measurement.md` Bug 1 |
| 分子分母范围一致（都增量或都累计） | 同上 Bug 2 |
| DLM 必须报 repetition rate，不能只报 loss | `notes/03-minimind-dlm.md` §11 |
| 用 4.57.0 的 API，不要写 5.x | `AGENTS.md` |
| 数值先在 CPU 上对拍，再上 GPU | `logs/dev/01-local-state.md` |
| 注释只写「这行为什么这么写」，不写教科书段落 | `AGENTS.md` |

## 这一轮 review 抓到的三个「不报错但错」的例子

都在 SFT 这一段，都让 loss 曲线看起来正常：

1. **assistant 后的 EOS 不在监督 span 内** → 模型不会停，生成永远跑满。
   更糟的是**测试在断言这个错误行为**，所以它躲过了一轮 review。
2. **采样 prompt 写成 `f"user\n{p}\nassistant\n"`** → 换行占了 EOS 的位置。
   长度相同、token 不同、不报错，但每个样本文本和 rep-4 都在描述一个没训过的前缀。
3. **`*-sft-dllm_a2d` 四格训的是因果模型** → dllm 的 loss 不传 mask，
   双向性必须由模型自己带，而那个脚本用了普通 builder。
   格子的名字写着 A2D，训出来的是因果模型。

详见 `example/ISSUES.md` 的 DONE-11 / DONE-12 / DONE-13。

## 三条反复踩到的经验

这三条在多个文件里都出现过，所以先说：

1. **错的测试比错的代码更危险。** 一个「不可能失败」的断言会给你假的安心感。
   实验版有两个断言曾经永远通过，因为它们检查的东西根本不参与计算。
2. **loss 下降不代表模型学对了。** 复读（repetition）是 DLM 失败最典型的表现，
   而 loss 照样下降。所有 DLM 评测都要报 repetition rate。
3. **静默错误比崩溃危险。** span 存 2 元组靠反推块号、padding 的 label 填 0、
   全忽略 batch 不特判——这三个都不报错，loss 曲线都好看。