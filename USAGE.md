# USAGE

项目完成后，拿到这个仓库的人从零开始用的指南。
设计契约在 SPEC.md，施工计划在 PLAN.md；本文只讲「怎么用」。

## 1. 环境准备

**硬件要求**：单卡 amd RDNA3+（ROCm 10.0 wheel），显存 ≥ 12 GB；纯 CPU 也能跑 smoke（慢）。

```sh
# 1. uv venv（Python 3.14 + ROCm torch wheel，几个 GB，装在项目 .venv/）
./scripts/setup_env.sh

# 2. dllm 包（钉版本 clone + 打补丁 + editable 安装；A2D 的 dllm 路线用）
./scripts/install_dllm.sh

# 3. 大文件软链（disk 空间分配，按需改目标路径）
./scripts/setup_symlinks.sh     # data -> /mnt/extra/data , checkpoints -> /mnt/extra/ckpt
```

直连 huggingface.com 会超时的话：`export HF_ENDPOINT=https://hf-mirror.com`。

## 2. 数据准备

```sh
# minimind 数据集 -> 目标格式（tokens.bin / doc_lengths.npy / meta.json）
python -m common.prepare --dataset minimind --vocab 6400
```

语料要求：vocab 6400；MASK（id 27）不允许出现在正文；每篇文档用 BOS/EOS 包裹。
数据落在 `data/prepared/<name>/`，两条模型轨道（trans / native）共用同一份。

## 3. 训练

全部走同一个入口 `src/training/main.py`，换实现 / 换阶段不改命令形状：

```sh
# 冒烟：代码能跑通、loss 能下降（约 1 分钟，GPU；名字带 -smoke 后缀）
python -m src.training.main --tier smoketest --impl trans --steps 20

# 正式 pretrain（tier1，26.9M，约 11.6 h @332M token）
python -m src.training.main --tier tier1 --impl trans     # transformers + Trainer 路线
python -m src.training.main --tier tier1 --impl native    # 纯 torch 包路线

# SFT：接某个 pretrain 权重，只在 assistant 区间算 loss
python -m src.training.main --stage sft --base checkpoints/ar-26.9m-trans

# A2D：同 corruption 下两条实现互为对照（G5）
python -m src.training.main --stage a2d --impl dllm --base checkpoints/ar-26.9m-trans-sft
python -m src.training.main --stage a2d --impl mini --base checkpoints/ar-26.9m-trans-sft

# Direct-DLM：跳过 AR，从头训扩散语言模型（G2 的 dlm 侧）
python -m src.training.main --stage dlm --impl trans
```

checkpoint 名就是 pipeline：`ar-26.9m-trans-sft-dllm_a2d` 这类，规则见 SPEC §1.2。
冒烟档全部加 `-smoke` 后缀。

## 4. 评测

```sh
# 能 chat 四门槛（SPEC 6.3）：assistant eval loss / rep-4 / 三 prompt 格式 / 不回声
python -m src.eval.chat --base checkpoints/ar-26.9m-native-sft
```

DLM 系的结论必须报 repetition rate，不能只报 loss；rep-4 在短样本上噪声大，
正式对比用几百 token × ≥5 prompt。

## 5. 产物

| 路径 | 内容 |
|---|---|
| `checkpoints/<名字>/` | `model.safetensors` + `config.json` + `meta.json` + `train_state.pt` |
| `logs/run/<名字>/` | `summary.json`（唯一真相）+ `metrics.csv`（每步一行） |

## 6. 测量报告规则（SPEC §6.4，硬要求）

- 吞吐必须同时报 **train-only** 与 **wall-clock** 两个口径；
- benchmark ≥ 400 步，丢弃预热，`Predicted` / `Measured` 分列；
- 跨实现对拍用同 seed 同输入，逐位比较，不许看「两个相近的数字」。

## 7. 排障

坑与解法沉在两处：硬件与本机问题见 `logs/dev/`，与机器无关的机制问题见 `notes/`
（如 07 的 pyyaml 浮点与 RNG 命名、03 的 DLM 复读表现）。
