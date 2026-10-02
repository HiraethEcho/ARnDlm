# Step 0 — `utils/`：公用底座

**你要写的**：`utils/`（logger · device · seed · checkpoint · metrics）
**实验版参考**：`example/common/`（config · seeds · data · model · ckpt · metrics · loop · tokenizer · report）
**验收**：实践版的 `utils/` 与 实验版在同一输入下行为一致，且 `test/01`–`test/05` 仍全绿。

这一层没有矩阵格，但 10 格全部依赖它。**先写它，后面每一步都在它上面跑。**

---

## 0.1 模块划分

| 模块 | 实验版文件 | 职责 | 关键决策 |
|---|---|---|---|
| 配置常量 | `common/config.py` | Tier 0 形状、路径、special id | 一个 dataclass，不在脚本里重复写数字 |
| seed / device | `common/seeds.py` | 三个 RNG + 设备选择 | `require_cuda` 要能报错而不是静默用 CPU |
| 数据 | `common/data.py` | memmap → 定长块、批次顺序 | 块 int32、尾巴丢弃、顺序可恢复 |
| 建模 | `common/model.py` | `ModelCfg` → `Qwen3Config` / 原生 config | 字段全部显式写 |
| 存盘 | `common/ckpt.py` | safetensors + train state + 参数普查 | 跨行通用，不绑某一行的格式 |
| 指标 | `common/metrics.py` | 双吞吐 + repetition rate | 计时区间由调用方传入 |
| 训练循环 | `common/loop.py` | optimizer / 调度 / 计时 / resume | 一个循环，行间共用 |
| tokenizer | `common/tokenizer.py` | encode / decode | 默认不加 special token |
| 报表 | `common/report.py` | 追加一行到 RESULTS.md | rep-4 是必填参数 |

---

## 0.2 会踩的坑（全部在 实验版里真的踩过）

### 1. 打包时 BOS/EOS 写在错误的下标上

把 `[BOS] doc [EOS]` 拼成一条流时，需要**两张偏移表**：

```
starts      = cumsum(lengths + 2)   # 流里的位置（含 BOS/EOS 空位）
doc_starts  = cumsum(lengths)       # token 文件里的位置（不含）
```

用错哪张，正文 token 都会整体偏移几个位置。**症状不是报错，是数据看起来还挺通顺。**
实验版第一版就是这么错的，被 `example/tests/test_common.py` 的「手工构造三篇文档」检查抓住。

### 2. 尾巴用丢弃，不要用 padding

短掉的那段如果补 PAD(0)，PAD 就进了训练目标。除非你显式把 labels 置 -100，
否则模型会学着预测 PAD。直接 `n_blocks = len(stream) // block_size` 丢掉。

### 3. batch 顺序要能存进 checkpoint

`DataLoader` 的 sampler 状态存不下来，「跑 N 步 + 恢复」和「一次跑 N+K 步」就对不上。
实验版用 `BlockStream` 存 `(epoch, pos)`，permutation 由 `(seed, epoch)` 重新生成，
resume 只要存两个整数。

**验收就是 C4**：连续跑和分段跑，逐步 loss 必须逐位相同。
实践版的验收也要跑这一条 —— 见 `common/loop.py` 里 `max_steps` 的注释，
它存在就是为了让「跑一半」不改 LR 调度。

### 4. tied embedding 在 optimizer 里要按 `id()` 去重

`model.parameters()` 默认会去重，但 `state_dict()` 不会 ——
`model.embed_tokens.weight` 和 `lm_head.weight` 两份都留着。
`safetensors` 遇到共享存储会直接报错，所以存盘时也要去重。
`SPEC 3.1` 明确要求传 optimizer 前按 `id()` 去重：同一张量被 AdamW 更新两次，
等效学习率翻倍，200 步看不出来，10 万步就偏了。

### 5. wall-clock 别把验证和采样算进 train-only

实验版自己踩过一次：`summary` 在 eval / sample **之前**组装，于是 wall 和
train-only 三个数字位都一样 —— 正是 `notes/05-measurement.md` Bug 1 描述的形状。
现在 eval/sample 在计时区间内，summary 最后组装。

### 6. 跨实现比 loss 必须固定批次 + fp32

两套实现比 loss，要点：同一批 token（`fixed_batches()`，独立 seed）、
同一个 corruption seed、**fp32 跑**。bf16 下两套实现的舍入差比要测的差异还大，
比出来的是噪声。

### 7. 参数普查分两个数

`parameters()` 去重、`state_dict()` 不去重。两个数都记进 `meta.json`，
`describe()["tied"]` 标记它们是否一致。实验版第一版用「参数和 != 参数唯一和」判断 tying，
是错的。

---

## 0.3 验收命令

```sh
# 实验版自检（40+ 检查，失败非零退出）
.venv/bin/python example/tests/test_common.py

# 实践版写完后：同样的断言换成 utils/ 的实现，逐条对上
.venv/bin/python test/01-tokenizer-mask.py
.venv/bin/python test/02-data-pipeline.py
```

逐条要对齐的行为：

| 行为 | 实验版接口 |
|---|---|
| Tier 0 → `Qwen3Config` 字段全对 | `common/model.py::build_qwen3_config` |
| 手工语料的 BOS/EOS 拼接 | `common/data.py::_stream` |
| 同 seed 同批次顺序 | `common/data.py::BlockStream` |
| 存盘去重 + 位级一致 | `common/ckpt.py::_unique_state_dict` |
| train-only 不含 idle 时间 | `common/metrics.py::TpsMeter` |
| 复读文本 rep-4 > 0.8 | `common/metrics.py::repetition_rates` |
| 恢复后逐步 loss 相同 | `common/loop.py::Loop.save/load` |