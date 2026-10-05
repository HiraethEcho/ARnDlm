# 本机实测性能（Radeon 880M / gfx1150）

> 详版（逐 kernel 效率表、SDPA backend 对比、带宽反证、gradient checkpointing、
> 5 轮取最快 + spread 判据的测量方法）在 `logs/dev-legacy/02-hardware-perf.md`。
> 本文只保留当前分支仍然要引用的结论。**数字都是本机实测**，重测前不要改。

## 硬件上限

```
GTT 带宽 (copy 1024MB)   75.6 GB/s   (read-only sum: 92.7 GB/s)
matmul 峰值              8.6 TFLOPS @2048, 7.6 @4096, 2.7 @8192(不可信, L2 装不下)
3090 对照                936 GB/s, 71 TFLOPS
```

## 瓶颈：attention 的 backward kernel

- SDPA forward 12.5 ms（2.65 TFLOPS）→ backward 105 ms（**0.28 TFLOPS**），8.4×。
  CUDA 上这个比值约 2×。ROCm gfx1150 没有 attention 的高效 backward kernel。
- 同 step 的 linear GEMM 跑 5.0–7.8 TFLOPS（峰值 6.8–8.6），不是瓶颈。
- 换 SDPA backend 无效：DEFAULT / FLASH / EFFICIENT 三个 backend backward 全慢
  8–10×；CUDNN 未编译，MATH 更慢。
- attention 只占 12.9% FLOPs，吃掉 36% step 时间 → **序列长度比 FLOPs 更贵**。
- gradient checkpointing 本机纯亏，不开。

## Tier 阶梯（2026-10-01 实测）

`bench_throughput.py`：warmup 50 + 计时 100 步，packing 零 padding，bf16 + AdamW +
sdpa，卡空闲（`GPU use 0%`）。模型 Qwen3 风格，vocab 6400。

| 配置 | params | train tok/s | wall tok/s | ms/step | peak GB | h@332M |
|---|---:|---:|---:|---:|---:|---:|
| tier0 h256 L2 @128 | 3,213,056 | 54,502 | 42,422 | 18.8 | 0.18 | 1.7 |
| tier0 h256 L2 @512 | 3,213,056 | 63,576 | 60,169 | 64.4 | 0.50 | 1.5 |
| tier1 h512 L8 @256 | 26,879,488 | 8,999 | — | 227.6 | 0.81 | 10.3 |
| tier1 h512 L8 @512 | 26,879,488 | 7,995 | 8,067 | 512.3 | 1.41 | **11.6** |
| tier1 h512 L8 @1024 | 26,879,488 | 6,731 | — | 1,217.0 | 2.68 | 13.7 |
| tier2 h768 L12 @512 | 80,433,408 | 3,540 | — | 1,157.1 | 2.76 | 26.1 |

- step 外开销（组批 / H2D / python）实测 < 1%。
- train-only ≈ wall（早期「固定开销占 36–48%」是 wall 口径 bug，已修，见 legacy/02）。
- seq 1024 比 512 慢 18% → AGENTS.md 的 `seq ≤ 512` 在本机复现，成立。

## smoke 尺寸的判读注意

- 前 50 步是编译相：单步比稳态慢 1.15–1.8×（h256 L2 实测）。冒烟曲线的
  **前 50 步 loss 不可信**，判读从预热之后开始。
- run-to-run 方差 ±20% 只在「卡真的空着」时成立；spread > 20% 的数字不要信
  （判据与全部口径规则在 `notes/05-measurement.md`）。
- 这些 tier 数字是旧布局下测的，**新代码冒烟不引用它们**——smoke 只验代码正确
  （对拍一致、loss 能降），不评性能。
