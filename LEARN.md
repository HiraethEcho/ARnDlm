# learn 工作区

你在 `ARnDlm-learn/` 里**动手写实践版**，`ARnDlm/`（main 分支）是参考答案。

## 规则（和 main 一致，见 `../AGENTS.md`）

| 目录 | 你写什么 | 实验版参考 |
|---|---|---|
| `models/hf_qwen/` | 你的实现 | `../example/base/hf/model.py` |
| `training/ar_hf/` | 你的实现 | `../example/base/hf/train_ar.py` |
| `data/sft/` | 你的实现 | `../example/common/sft.py` |
| `training/sft_ar_hf/` | 你的实现 | `../example/base/hf/train_sft.py` |
| `training/sft_a2d_hf/` | 你的实现 | `../example/base/hf/train_dlm_sft.py` |
| `diffusion/` | 你的实现 | `../example/base/hf/a2d_{minimind,dllm}.py` |
| `training/a2d_dllm/` | 你的实现 | `../example/base/hf/train_a2d_from_sft.py` |

**不要复制 `../example/` 的代码。** 复制过来就没有对照意义了——
你要的是自己算一遍每个结论。

## 学习顺序

```
ar-hf  →  ar-hf-sft  →  ar-hf-sft-dllm_a2d
  A0 → A1 → A2        B1 → B2        C1 → C2 → C3 → C4
```

完整索引在 **`../example/teach/README.md`**，那一页按顺序列出了 9 个文件。
前 9 步走完，上面那三格矩阵就齐了。

## 每一步怎么验收

**跑参考实现拿基准数，再跑你的，比。** 不一致的地方就是你写错了的地方。

```sh
# 0. 先确认环境
../.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# 1. 数据（SFT 之前的步骤不需要）
../.venv/bin/python ../example/base/hf/make_sft_data.py --max-blocks 4096

# 2. 单格
../.venv/bin/python ../example/base/hf/train_ar.py       # 拿基准
../.venv/bin/python train_ar.py                          # 你的

# 3. 跨实现对拍
../.venv/bin/python ../example/check_ab.py c7
```

## 三个「不报错但错」的例子（都在 SFT 段）

写之前先知道，因为它们都让 loss 曲线看起来正常：

1. **assistant 后的 EOS 不在监督 span 内** → 模型不会停。
   而且**曾经有测试在断言这个错误行为**，所以它躲过了一轮 review。
2. **采样 prompt 用 `f"user\n{p}\nassistant\n"`** → 换行占了 EOS 的位置。
   长度相同、token 不同、不报错。
3. **docstring 里的块结构图和代码不一致** → 下游照着图写就错。

详见 `../example/ISSUES.md` 的 DONE-11 / DONE-12 / DONE-13。

## 容易记错的三条约定

| 约定 | 原因 |
|---|---|
| turn 之间用 **EOS** 分隔，不是换行 | 采样 prompt 必须同样用 EOS |
| **一个块只有一个 BOS**（在最前面），turn 之间不再插 BOS | pretrain 的 packing 也这样 |
| span 的 `end` **含** assistant 的 EOS | EOS 是教模型停止的唯一一票 |

## 报告数字

- **同时报 train-only 和 wall-clock 两个吞吐**（`../notes/05-measurement.md` Bug 1）
- **DLM 必须报 repetition rate**，loss 降不代表没复读
- 跨实现比 loss 时**用 fp32**，bf16 的舍入差比你要测的差异还大