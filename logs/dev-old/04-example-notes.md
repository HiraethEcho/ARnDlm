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
2. prepare 的 hf 下载路径（22 GiB minimind）未跑；wikitext-103 已走 parquet 直链
   入盘（2026-10-05，见 01/notes06）。
3. ~~C2/C3/C5~~ 已完成（见下）；**未做**：dlm.py（Direct-DLM）、dpo.py、REPL 的
   DLM 采样器（A2D ckpt 被 REPL 当 AR 加载会错采）。
4. 门槛 1 需要 DLM-SFT 参照；chat 报告现处 pending。
5. G1 的「同 seed 同 batch 逐步 loss 对齐」：trans 与 native 各自 init（构造序不同
   → 权重不同），现只能比曲线形状不比逐位。同权重起点 = SFT 侧已有
   `init_weights_path`，pretrain 侧未做。

## A2D 追加记录（2026-10-05 第二段 night-run，用户限定不做 direct-dlm）

| 交付 | 验收 |
|---|---|
| common/a2d-example.py（corrupt/loss/sample_dlm 共享层） | test/12：分母语义手算对照、per-sample t、valid 限制、Gumbel fp32 |
| a2d_mini-example.py（trans+native 双 base 闭包） | test/13 [1]：双向 logits 1.43e-06 |
| a2d_dllm-example.py（MDLMTrainer.__new__ + 假 tokenizer） | **C2/C3 相对差 = 0.0**（两个 base），好于 legacy 7.6e-08 |
| C5（test/14） | config 逐字段 + state_dict 直拵：0/0/0/0，转换后双向 logits 差 0.0 |
| main --stage a2d（mini\|dllm，200 步端到端） | loss 7.83→5.90 eval 6.02；mini 与 dllm 轨迹**逐位相同** |

**两个真实的坑（都写进了代码注释）**：
1. dllm 的 compute_loss 收不到 attention_mask 时，普通 Qwen3 **静默走因果 forward**
   —— DLLMLoss 必须注入 4D 全零 mask，test/13 [5] 专门钉住（去掉注入口，
   相对差变 9.75e-04，这个测试就是能失败的那个）。
2. BOS 前置被 `right_shift_logits=False` 整体关死，从不发生（源码验证）——
   我先入为主怀疑是它，被实验推翻后改掉了错误解释。

**rep-4 0.72/0.83（高复读）**：200 步 smoke 从 20 步 AR 底座训练，样本是点号串。
SPEC 6.4 的 DLM 失败模式（loss 降但复读）如实呈现，不是 bug；追步数/数据是
正式格子的工作。A2D ckpt：`checkpoints/ar-3.2m-{mini,dllm}_a2d-smoke`。

## 全链 night-run（2026-10-07 02:30–07:00，批准窗口）

范围：smoke 3.2M 全部格子 + direct-dlm/dpo 代码 + 四门槛全报。
截止规则：07:00 前干完则多模型 review+debug+re-smoke；未完成则完成当前格再停。

### 新增交付

| 项 | 内容 |
|---|---|
| `main --stage dlm` | Direct-DLM（随机初始化）。**2026-10-14 更正**：当时报的「双 impl 轨迹逐位相同」是假象——旧 `run_a2d` 的分支写作 `if impl == "mini" ... else dllm`，`--impl trans` 静默走了 dllm loss，两条从来不是两条。抽出 `dlm-example.py` 修复后：trans eval 5.0192 / dllm 5.0691（相对差 1.0%，同 C2 量级） |
| `main --stage a2d --sft-data` | DLM-SFT：a2d 循环吃 [B,2,L] SFT 批，span 限制加噪（T4f 语义复用） |
| `main --stage dpo` | DPO：冻结 ref 副本 + -logsigmoid(β·margin)，lr×0.1；summary 带 held-out margin acc |
| `dpo-example.py` | pair 编码（右 pad 定长，response 区间 logprob），PairBatcher 纯函数序 |
| REPL `--sampler dlm` | A2D/DLM ckpt 的双向块状揭示采样 |
| `scripts/collect_results.py` | 全格子四门槛报告 -> RESULTS-smoke.md（含 chat_report.json 逐格落盘） |
| `--name` 覆盖 | 复合格名按 SPEC 1.2 追加；强制 -smoke 后缀 |

### 格子矩阵（RESULTS-smoke.md 汇总，200→800 阶梯）

- G1 pretrain × 2 · SFT × 2（800）· G3 双顺序 × A2D 两 impl · G2 Direct-DLM × 2 · DPO × 2
- t2（rep-4<0.5）：AR 两格 ✅；原路线 a2d-sft 与 direct-dlm-sft 800 步后 ✅
  （4/6 DLM 行）；剩两格各有一 prompt >0.5，如实记录未修圆
- DPO：loss 0.693→0.140，held-out margin acc 0.825（500 对 × 200 步 smoke）

### 真实事故与修复

1. **debug run 覆盖正式产物**：验证 --name 前忘了加 --name，200 步 debug 覆盖了
   昨晚正式的 `ar-3.2m-mini_a2d-smoke`。正是 night-run 反模式表的那条。修复：
   --name + 强制 -smoke 后缀，正式格子矩阵阶段重跑恢复。
2. 多条 edit 因锚点漂移静默丢失 —— 发现后逐条重做，每条独立验证后才提交。
   教训：同一文件多次 edit 之间必须 grep 锚点，不能凭记忆。

### 评审轴（03:21 发射, 03:28/03:31 回收, 全部落实为 commit）

| agent | 模型 | 发现 | 处置 |
|---|---|---|---|
| review-check-cannot-fail | grok-4.7 | 7 个恒真检查 + 12 个弱断言 + RESULTS 数字无测试背书 | `d6b1bae` 全部修复；15 测试过 |
| rev-spec-conformance | glm-5.2 | 11 处契约偏差（高: a2d 命名丢 impl 段 / G3 混入第二变量; 中: T4a 弱化 / T4g 无执行点 / SFT 数据无生产者 / C4 只测 native） | `e2fc947` `b5650c8` `2d8ceed` 命名/参照系/T4a/T4f/T4g/sft-prepare 全修; 见下未修清单 |
| rev-bug-hunt | kimi-k3 | 429 用量超限, 未产出 | 换 deepseek-v4-pro 重发 |
| rev-bug-hunt2 | deepseek-v4-pro | 6 缺陷（高: native 双初始化 / lr warmup 错位; 中: tier1 head_dim=128 与 26.9M 锚点矛盾 / a2d+dpo 静默 fp32; 低: repl 硬编码 smoketest / 温度 0 NaN） | `e2fc947` 修 5; warmup 错位见下 |

### 评审后仍未修（诚实清单）

1. **lr warmup 错位（bug-hunt #2）**：native `lr_at` step0 = lr/warmup、峰在 warmup-1;
   HF 4.57 cosine_with_warmup step0 = 0、峰在 warmup。两路线 warmup 段每步 lr 不同。
   修法 = 把 `lr_at` 改成 HF 的 step/warmup 语义（test/01 [7] 期望值跟着改），
   影响全部已跑格子的 warmup 256 步——**留给正式 tier1 之前**，smoke 不改
   （改了会让所有已存 smoke 产物口径与新代码不一致）。
2. **native 与 HF 的逐位 init 不等**（deepseek 证实; HF 4.57 fast-init 走
   tensor.normal_(), 不可经 nn.init hook 复刻）。C1 是权重拷贝制不受影响;
   G1 逐步对齐开放点同前。
3. **C4 trans 侧 resume** 仍未做（HF Trainer checkpoint 关系）。
4. **两格 t2 在 800 步仍失败**：trans-sft-{mini,dllm}_a2d-sft（原路线复合）
   各有一个 prompt rep-4 ≈0.9。SPEC §6.3 阶梯止于 800——按「能 chat 即可」
   的口径这两格算到顶了, 结论如实写进 RESULTS（t1 全过, t3/t4 全过）。
5. **四门槛是报告制**：smoke 档不设门禁（§4.1 只验管线）; tier1 沿用时需给
   collect_results 加 --gate 之类的硬验收（不在今晚范围）。

### M22 warmup 语义统一（2026-10-07 中午补做, 用户 continue 指示）

评审发现的 lr warmup 错位已修：`lr_at` 改为 HF 语义（step0 = 0、峰在
warmup 整点），test/01 [7] 期望值同步。**全部矩阵产物已按新语义重跑**
（17 个格子命令 + collect_results 重新生成 RESULTS-smoke.md），代码与
盘上产物口径一致。15/15 测试 PASS。

新口径下的门槛分布（诚实）：t1 全过；t2 = 4/9 过（原路线复合与新路线
trans-mini_a2d-sft 的第三个 prompt rep-4 ≈ 0.9-0.97; dlm-sft 800 步档
0.66-0.75 也回落失败）。smoke 口径下的诚实结论：**双向去噪的复读率对
底座/步数敏感**, tier1 上才值得下结论（SPEC 6.4: rep-4 不能给实现排名）。

### 2026-10-14 补缺（两个缺口, 各带一个真 bug）

1. `dlm-example.py` 补齐（Direct-DLM 独立成格）。修掉：旧 `run_a2d` 分支写作
   `if impl == "mini" ... else dllm`，`--stage dlm --impl trans` 静默走 dllm loss。
2. `test/16-dpo.py` 补齐（19 项）。修掉：`load_dpo_pairs` 的 `max_pairs` 数的是行
   不是 pair —— 旧 `--dpo-pairs 1000` 实际只训 500 对。重跑 DPO：
   margin acc trans 0.700 / native 0.645（各 1,000 对 × 200 步）。
3. 测试总数 15 -> 16, 全 PASS。

### 纪律记录

- 审查发现 test/05 的期望值 9.505/0.525 本身是错的（真值 7.525/2.525）——
  「期望值恰好匹配 bug」型陷阱, 与恒真检查同罪。
- logs/run 与 checkpoints 的 pre-refactor 产物（*-example / -hf / 重复 family 段）
  全部归档到 `/mnt/extra/ckpt/_pre-refactor-noncompliant/`, 契约命名面干净。
