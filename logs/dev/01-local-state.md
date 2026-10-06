# 本机状态

## refactor 工作区现状（2026-10-05 night-run 后）

- 分支 `night/example-codes`（基于 refactor）。旧开发笔记归档在 `logs/dev-legacy/`，
  仅作背景查阅，**过时结论不要从这里引用**。
- **项目当前进度**：P0 完成。P1–P5 的 **example 侧**全部交付（10 个 example +
  10 个测试文件全过），x.py 骨架未动（待人填）。交付清单、盘上数据格式实测、
  踩坑与开放点见 `logs/dev/04-example-notes.md`。
  smoke 实测锚点：AR 20 步 loss 8.81→8.71→eval 8.70（trans 8.68）；SFT 200 步
  assistant eval loss 6.39、rep-4 0.33–0.43；**C1 2.15e-06 / C4 0.0**。
- 未开始：P6 A2D/DLM（C2/C3/C5）、dpo、tier1 正式训练。
- REPL 已交付（2026-10-05）：`src/eval/repl-example.py`，pretrain 续写 / chat 对话两模式，
  ckpt 目录自动识别 trans/native 存法。SFT 产物持久化在
  `checkpoints/ar-3.2m-{native,trans}-sft-smoke`（800 步, eval 4.61 / 4.78, 输出已是通顺中文）。
  保留的四个冒烟 ckpt：`ar-3.2m-{native,trans}-smoke`（AR 20 步）与上面两个。
- `logs/run/` 与 `checkpoints/*-example` 都是实验版遗留（gitignore，不算本分支工作）。
  新格子命名按 SPEC §1.2；smoke checkpoint 带 `-smoke` 后缀。

## 磁盘与数据

`.venv/` 指向主仓环境；`data/` `checkpoints/` 软链到 `/mnt/extra`。

| data/prepared/ | token 数 | 用途 |
|---|---:|---|
| `minimind3-6400` | 332,495,324 | 主实验语料（tier1 对照） |
| `wikitext-6400` | 194,509,960 | 英文维基百科书面语（2026-10-05 加，见 notes/06） |
| `sft-mini-6400` | 4,096 块 @ ctx 128 · 260,590 supervised | SFT 数据（assistant-only） |
| `tinystories-6400` | 3,110,358 | 小语料备份 |

smoke 级数据现状：`tinystories-6400`（310 万 token）与 `sft-mini-6400` 就位。
smoke 原则：**尽量轻和快**（SPEC §4.1 / PLAN「smoke 约定」）—— 最小模型 ×
最小数据 × 最少步数，只验代码正确，不评性能。想要更小的专用语料就单独 prepare。

## 环境（2026-10-01 实测，legacy 记录仍然成立）

```
.venv/        uv 建, Python 3.14.7, 6.2G, 108 包
torch         2.12.0+rocm10.0.0  | hip 7.15.26333
transformers  4.57.0   ← dllm 的 pin (PyPI 上 yank, wheel 可用, 已实测)。不要升 5.x
peft          0.17.1   | accelerate 1.15.0 (peft 硬依赖) | datasets 4.2.0
tokenizers    0.22.2
GPU           AMD Radeon 880M (gfx1150), cuda.is_available() = True
系统 python   torch 2.14.0+hip + transformers 5.17.0 —— 坏的, 不用
net           huggingface 直连超时, 用 HF_ENDPOINT=https://hf-mirror.com (dev.zsh 已设)
SwanLab       要 SWANLAB_API_KEY (dev.zsh)，没 export 就跳过不报错
```

ROCm 后端能算（512×512 matmul 与 CPU 参考误差 1.2e-4，fp32 正常范围）。
坑与细节都在 `logs/dev-legacy/01-local-state.md`：

1. torch 张量默认在 CPU。建完模型不 `.to('cuda')` 就在 CPU 上跑，
   且 `torch.cuda.is_available()` 照样 True，看不出异常。
2. `peft` 硬依赖 `accelerate>=0.21.0`，不显式装会让 uv 重解 torch。
3. `transformers==4.57.0` yank 是打包问题，wheel 本身可用；升 5.x 会与 dllm 断裂
   （`create_causal_mask` 签名改了）。
4. Python 3.14 无害告警：`multiprocess/connection.py` 两条 SyntaxWarning。

## 硬件性能（Radeon 880M / gfx1150，2026-10-01 实测）

完整表在 `logs/dev-legacy/02-hardware-perf.md`。本项目再写的要点：

- **瓶颈是 attention backward kernel**：forward 12.5 ms → backward 105 ms，
  0.25 TFLOPS；linear GEMM 同 step 里 5.0–7.8 TFLOPS，没问题。
- 硬件上限：GTT 带宽 75.6 GB/s，matmul 峰值 8.6 TFLOPS @2048。
- gradient checkpointing 本机纯亏，不开。
- 序列越长越慢：tier1 @512 比 @256 慢（tok/param 下降），`seq ≤ 512` 约束在本机复现。
- 小模型（smoke 尺寸）预热贵：前 50 步是编译相，单步比稳态慢 1.15–1.8 倍。

## 参考仓与外部资料

- `~/Projects/dlm`、`~/Projects/minimind`、`~/Public/dlm/`（dlm 包本体）、
  `~/Public/minimind/`、`~/Public/ChatLM-mini-Chinese/`
- minimind-dlm 原理页：https://jingyaogong.github.io/space/dllm_intro
- 旧版实现参考：`git show c014da7:<path>`（`learn` 分支，含全部旧实现）
