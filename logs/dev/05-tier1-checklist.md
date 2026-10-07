# 05 · tier1 启动前检查清单（2026-10-07，night-run「tier1 前的工作」收口）

全部条目已实测或由测试背书；打 ✅ 的不要再重查，打 ⚠️ 的启动当天要再看一眼。

## 契约与代码

| 项 | 状态 |
|---|---|
| tier1 形状与 26,879,488 锚点一致（head_dim 64 修正后） | ✅ test/06 [5b] 解析式逐位 |
| tier1 形状下的 C1 | ✅ 3.40e-06 ≤ 1e-5（test/06 [5b] 实测） |
| warmup 语义统一（HF: step0=0、峰在 warmup 整点） | ✅ lr_at 与 HF 同式；两路线 warmup 段同 lr |
| C1–C5 阈值硬断言 | ✅ test/06/13/14（C4 native+trans 双侧 0.0） |
| G1 同起点机制 | ✅ `--dump-init` + pretrain `--base`（step-0 loss |Δ|=0.0） |
| 门禁 | ✅ `collect_results.py --gate`（任一 ❌ -> exit 1） |
| SFT 数据生产者 | ✅ test/15 |
| rep-4 报告义务 | ✅ collect_results 3 prompt × 64 token；summary 里 3 prompt × 64 |
| 15/15 测试 | ✅ |

## ⚠️ 启动当天再看

1. **G1 顺序**：`--impl trans --dump-init checkpoints/init-26.9m` →
   `--impl trans --base checkpoints/init-26.9m --tier tier1` →
   `--impl native --base checkpoints/init-26.9m --tier tier1`。
   两终点的对拍：同 step 抽样 loss + 终点 C1 式 logits（load 谁进谁都是 state_dict 直拷）。
2. **数据**：`cfg.DATA_DIR = minimind3-6400`；Corpus 四道自检全量扫一遍（332M token MASK 扫描 ≈ 秒级）——
   smoke 时代只扫过 4,096 块切片。
3. **磁盘**：checkpoints（26.9M × 多份 + 训练态）先看 `df /mnt/extra`。
4. **wall-clock**：pretrain 81,174 步 ≈ 11.6 h（7,995 tok/s 是 2026-10-01 旧口径，
   bf16 + 新 warmup 语义后以首小时实测为准；基准 ≥400 步丢弃预热）。
5. **门禁阈值**：t2 < 0.5 与 t1 ≤1.5 是 SPEC 写死的；smoke 中 t2 对底座敏感，
   tier1 首次 gate 失败先看 rep-4 是不是短样本抖动（≥5 prompt × 几百 token）。
6. **native 逐位 init**：G1 走共享 init 权重（--base），不需要也没做复刻 HF fast-init。

## 已知开放（不阻塞 tier1）

- smoke 中 5 个 t2 失败格（阶梯 800 顶格后仍 >0.5）——tier1 复跑时按门禁裁决
- spec-conformance #6 中提到的「summary.json 只 3 prompt × 64 token」已合规（2026-10-07 修）

## 启动命令（备份）

```sh
.venv/bin/python src/training/main-example.py --tier tier1 --impl trans --dump-init checkpoints/init-26.9m
.venv/bin/python src/training/main-example.py --tier tier1 --impl trans  --base checkpoints/init-26.9m
.venv/bin/python src/training/main-example.py --tier tier1 --impl native --base checkpoints/init-26.9m
# G2/G3/G5 全链按 smoke 命名规律（SPEC 1.2），--steps 由 config（81,174）接管
```
