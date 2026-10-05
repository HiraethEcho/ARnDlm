# 04 · example 代码 night-run 记录（2026-10-05）

分支 `night/example-codes`。范围 P1–P5 的全部 `x-example.py` + `test/`。
x.py 骨架一律未动（留给本人填）。每个 milestone 一个 commit。

## 交付清单

| example | 测试 | 状态 |
|---|---|---|
| common/config-example.py | test/01 | 未知键/缺键/3e-4-as-str 一次报全; lr_at 共用公式 |
| common/seed-example.py, device-example.py | test/02 | 三源一致; snapshot/restore 逐位; 无静默 CPU 回退 |
| common/tokenizer-example.py | test/03 | vocab 6400; 中文 roundtrip; MASK 不产出 |
| common/data-example.py | test/04 | 四道自检; 纯函数批次; epoch 跨界 |
| common/metrics-example.py | test/05 | 双口径 TpsMeter; rep4; 时钟可注入 |
| src/models/trans-example.py + native/model-example.py | test/06 | **C1 因果 2.15e-06 / 双向 1.37e-06 (阈值 1e-5)**; 参数量 3,213,056 逐位 |
| src/training/main/ar_trans/ar_native/state-example.py | test/07 | 双 impl 20 步跑通; **C4 resume = 0.0** |
| common/prepare-example.py | test/08 | 本地 jsonl 路径全验; 下载路径未跑(开放点) |
| src/training/sft-example.py | test/09 | T4b 区间/ T4a decode / T4f 无 MASK 全过 |
| src/eval/chat-example.py | test/10 | 四门槛报告; 门槛1 pending 不装绿 |

跑法：`.venv/bin/python test/NN-*.py`（全部从仓库根目录跑）。

## 盘上数据格式（实测, 换机器不成立但换会话有用）

| prepared 目录 | 格式 |
|---|---|
| `tinystories-6400` / `minimind3-6400` | tokens.bin = **裸文档拼接 uint16**（无 BOS/EOS），doc_lengths 恰好划分它；包裹是 data 层虚拟做 |
| `sft-mini-6400` | tokens.bin = **预切块** [4096×128]；assistant_spans.npy = int64 [n_spans,3] (block, start, end-excl)；sum(end-start) == meta.supervised_tokens == 260,590 |

模板：`user\n{q}assistant\n{a}`，轮次分隔 = EOS(id 2)。采样前缀必须逐 token 拼出
`[BOS] user\n{q} [EOS] assistant\n`——f-string 会把 EOS 位置写成换行。

## 本次踩到的坑（写 x.py 时全部适用）

1. **连字符文件名不能被 import**。每个 example 自包含；互相调用走 `_load()`
   importlib 桥。x.py 填完后这些桥全部删除、换正常 import。
2. **`from __future__ import annotations` 会把 dataclass 的 `f.type` 变成 str**，
   `f.type is float` 静默失效 —— 类型校验变成 check-that-cannot-fail。别加。
3. **Trainer 对「有 `__len__` 的数据集」一律走 RandomSampler**（4.57）。既想定序
   又给 `__len__` = 批次顺序被静默洗掉（C4 推不回）。IterableDataset 不要定义
   `__len__`；裸类（不继承 IterableDataset）会被 SequentialSampler 要 len 而炸。
4. **searchsorted 区间映射要用区间起点**（`virt_start`），区间连续时
   `end[i]==start[i+1]`，用终点会让 doc 0 全部映射成 -1 —— 被 test/04 的朴素
   对拍抓住。
5. **rep-4 在有限长度上永远到不了 1.0**（全复读 = 1 - 1/(L-3)）；测试期望值要按
   定义算，不能拍脑袋写 1.0 / 0.5。
6. native（纯 torch）模型没有 `save_pretrained`；tied 权重共享存储会让
   safetensors 拒存 —— 只存一份、共享关系记进 metadata，与 transformers 同处理。
7. 两次 `rng.choice` 抽的组不一样 —— 「同一组两个口径」的检查必须抽一次存变量。

## 开放点（下个 night-run / 本人决策）

1. trans 侧 resume（HF Trainer checkpoint 与 train_state.pt 的关系）——
   03-call-flow 的遗留问题，未解。
2. prepare 的 hf 下载路径（22 GiB minimind）未跑。
3. C2/C3/C5（A2D 对拍）与 dlm/a2d_dllm/a2d_mini/dpo 的 example —— P6+，未开始。
4. 门槛 1 需要 DLM-SFT 参照；chat 报告现处 pending。
5. G1 的「同 seed 同 batch 逐步 loss 对齐」：trans 与 native 各自 init（构造序不同
   → 权重不同），现只能比曲线形状不比逐位。同权重起点 = SFT 侧已有
   `init_weights_path`，pretrain 侧未做。
