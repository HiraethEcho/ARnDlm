# minimind-dlm — DLM 原理与 A2D

**这套方法在本项目里称为 `minimind-dlm`。** 它有两条路：
`training/a2d_minimind/` 手动实现（这条路），`training/a2d_dllm/` 用 dllm 包做（另一条路）。
两者必须在同一 checkpoint 上跑出同一个 loss。

来源:
- `https://jingyaogong.github.io/space/dllm_intro` — 原理 + 交互式可视化
  （训练/推理两个模拟器；公式已全部收进本文，模拟器本身不摘抄）
- minimind discussion #618 (jingyaogong) — 完整实现
- 论文见文末

---

## 1. AR 的数学形式

```
p(x) = ∏_{i=1}^{N} p(x_i | x_<i)
```

文本被分解成**一条严格有序的决策链**。第 i 个 token 只依赖左侧上下文,
信息流单向。训练和推理目标完全一致: 都是 next-token prediction。

**AR 的三个问题:**

| 问题 | 说明 |
|---|---|
| 串行生成 O(N) | 生成 N 个 token 要 N 次 forward。KV cache 只降低单步开销, 改不了串行本质 |
| 固定顺序 | 很多任务天然适合「先定全局结构, 再填局部细节」 |
| **Exposure Bias** | 训练依赖 ground truth, 推理依赖模型自身输出, 分布偏差随步数累积 |

---

## 2. DLM 怎么做

```
p(x) = ∏_{i=1}^{N} p(x_i | x)      ← 条件是整个序列, 双向
```

**训练**: 随机遮住一部分 → 双向预测 → 只在被遮位置算 loss。
**推理**: 生成区初始化为全 MASK → 每步全序列 forward → 按置信度揭示。

**揭示顺序由置信度决定, 不是位置索引。** 模型会对同一个位置反复预测。

---

## 3. 从连续扩散到离散扩散

图像扩散用连续高斯噪声; 文本没有自然的连续噪声空间, 所以用 MASK 当离散噪声。

| 概念 | 连续 (DDPM) | 离散 (MDLM / dLM) |
|---|---|---|
| 噪声 | 高斯噪声 | 随机 MASK |
| 去噪目标 | 预测原始像素 | 预测被 MASK 的词元 |
| 前向 x₀→x_T | 越来越模糊 | 越来越多位置被掩码 |
| 反向 | 逐步去噪 | 逐步揭示 |

**没有 ELBO, 没有 log-SNR, 没有 weighting schedule。**
就是一个 linear-time schedule 的 weighted cross-entropy。

---

## 4. 训练: 4 步

```
Step 1  采 t  ~ U[0, 1)
Step 2  按比例 t 随机 mask **assistant 区** (prompt 区保持不变)
Step 3  双向 forward, 预测所有被遮位置
Step 4  只在 MASK 位置算 loss
```

```
L = E_t [ Σ_{i∈masked} (1/p_mask) · CE(f_θ(x_t)_i, x_0(i)) ] / N_valid
```

```python
loss = F.cross_entropy(logits, labels, reduction='none')
loss = (loss[corruption_mask] / p_mask[corruption_mask]).sum() / n_valid
```

**除以 `p_mask` 是为了平衡不同 t 下的梯度贡献:**
t 小时被 mask 的位置少但每个更关键, t 大时反之。不除的话 loss 会偏向大 t。

### 和 BERT / AR 的区别

| | 做法 |
|---|---|
| BERT | 固定 mask 比例 (15%) |
| **dLM** | 连续参数 t, 覆盖从低到高的**全部**损坏强度 |
| AR | 只学「预测下一个」 |
| **dLM** | 同时恢复**多个**被遮位置 |

### 训练数据格式

```
input_ids: [<user>] [你好] [<assistant>] [我是] [AI] [助手] [<pad>] [<pad>]
labels:    [-100]   [-100] [-100]       [我是] [AI] [助手] [-100] [-100]
                        ↑ 只有这些位置参与训练
```

**SFT 的 mask 区域 = assistant 回答区。** prompt 不能被 mask ——
否则模型看不到自己在回答什么。

---

## 5. 推理: 两个关键点

```
初始化:  [prompt 固定] [M M M M M M M M]   ← 生成区全 MASK

每一步:
  1. 对**全序列** forward (不只是当前 block)
  2. 算每个 MASK 位置的置信度
  3. 揭示 top-k, k = 剩余 MASK 数 / 剩余步数   ← 动态计算, 不是固定常数
  4. 循环
```

**关键点 1 — 每步揭示数动态计算:**
```
n_unmask = remaining_masks / remaining_steps
```
不是固定 top-k。

**关键点 2 — 每步都全序列 forward。**
双向注意力要用全部已知上下文, 不能只算当前 block。
**无 KV cache。** 这是 DLM 推理比 AR 慢的根源 (实测: 双向比因果慢 38%)。

---

## 6. block_size 是连续旋钮, 不是模式开关

| `block_size` | 行为 | 速度 / 质量 |
|---|---|---|
| `= max_new_tokens` | 全序列一次性去噪 | 最快, 质量取决于步数和采样 |
| **8 ~ 32** | 分块并行去噪 | **工程上常用的折中** |
| `= 1` | 退化成逐 token 自回归 | 最慢, 但最接近 AR |

**`block_size=1` 时 dLM 精确退化成 AR。** 这是理解 dLM 的关键直觉 ——
它是一个连续谱的一端。

---

## 7. 采样控制

### Gumbel-Max (比 softmax+multinomial 更稳)

```python
def add_gumbel_noise(logits, temperature):
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)
```

低精度数值条件下比 `softmax` + `multinomial` 稳定, 适合离散采样。
**`temperature=0` 时直接返回原 logits, 退化成 argmax (贪心解码)。**

### CFG (classifier-free guidance)

```python
logits = logits_uncond + (1 + s) * (logits_cond - logits_uncond)
```

需要**两次 forward** (有条件 / 无条件), 成本翻倍。

### 调参建议

| 参数 | 建议值 | 说明 |
|---|---|---|
| `block_size` | **8-32** | 越小越稳越慢, 越大越快但对 steps 越敏感 |
| `steps` | **≥ block_size** | 太少则每步揭示过多, 质量下降 |
| `temperature` | **0.2-0.8** | 过高散乱, 过低重复 |
| `top_k` | **20-100** | 截断候选空间 |
| `cfg_scale` | **0.5-2.0** | 过高会 mode collapse |

### 跟 `dllm` 包对拍时必知的两处语义差异

本项目同时用两种实现：手写版（按上面）和 `dllm` 包。两者**参数同名但含义不同**，
不先对齐就对拍，会把配置差异当成实现 bug。

| | 手写版（本文件） | `dllm` 包 |
|---|---|---|
| **`steps`** | **每块**的步数 | **全序列**预算，会按块数摊薄<br>`steps = ceil(steps / num_blocks)`<br>（`dllm/samplers/mdlm.py`） |
| **`t` 分布**（训练） | `t ~ U[0,1)` | `t = ε + (1-ε)·U[0,1)`，`ε = 1e-3`<br>（`MDLMConfig.time_epsilon`） |

**后果 1**——同一个 `--dlm-steps 8`：`dllm` 跑8 次 forward，手写版跑 32 次
（4 块 × 8）。任何跨实现的解码速度对比，**必须先换算到同一工作量**，否则测的是
「谁配置的步数少」。

**后果 2**——`E[1/t]` 相差 **0.1%**（6.91 vs 6.92），远小于对拍阈值；但**对拍必须
让两边抽到同一个 `t`**，做法是两边都用 `epsilon=1e-3` 并重新播种。

---

## 8. A2D: 两步转换

```
已训练的 AR 权重
   ↓ 1. 关闭因果约束   is_causal=True → False
   ↓ 2. 扩展词表        vocab += [MASK]
dLM 模型 → 只需微调
```

**继承 AR 的全部语言知识。** 可以冻结 FFN 只调 attention。

---

## 9. AR vs dLM 完整对比

| 维度 | AR LM | Masked Diffusion LM |
|---|---|---|
| 注意力 | 单向 (causal) | 双向 (bidirectional) |
| 生成顺序 | 从左到右逐 token | 按置信度, 最确定的先填 |
| 训练目标 | next-token prediction | masked-token prediction |
| 推理步数 | O(N) | O(T) = steps × blocks |
| **编辑能力** | 通常需额外机制 | **天然支持 infilling** |
| 训练成本 | 一般从零训 | **可从 AR 权重迁移** |
| 全局重评估 | 无 | **有** |

> 结论不是「dLM 已取代 AR」, 而是: dLM 把语言生成从严格的顺序链式过程,
> 扩展为可迭代、可全局重评估的推理过程。

---

## 10. 关键实验结论: 只需要训 Q/K

作者原话:

> FFN 层主要负责存储知识, 而 AR → dLM 的核心变化在于注意力模式
> (从因果变为双向), 因此冻结 FFN 仅训练 Attention 层即可取得不错的效果。
> 更极端地, 只训练 Q、K 投影也能获得一个可用的 dLM。

训练命令里直接给了 `--freeze_ffn` 开关。

**为什么合理:** Q/K projection 决定「每个位置看哪里」。AR 只学会了「往左看」,
现在要「往右也看」, 没学过。FFN 和 V/O 的功能不变。

**可证伪** — 值得重做消融 (全参 / freeze_ffn / 只 QK 三档)。

---

## 11. 实测结果 (作者自曝, 负面)

- 63.91M 参数, 1.6GB SFT 数据, 训练 1 小时
- 生成结果**很差**, 大量重复

```
💬: 请介绍一下你自己
🧠: 作为我我我具备学习和解决问题的能力,并能够完成各种任务。我能够完成各种
    任务,包括但不限于学习知识、解决问题、、学习、写作、等。
```

作者自己的判断:

> dLM 与 AR 在生成质量上仍存在显著差距。Discrete Diffusion 不像 Continuous
> Diffusion 那样有完备的数学框架, 其本质更接近 Masked Prediction 的延伸。
> 工程层面: 双向注意力要求每一步去噪都对 Block 内所有 token 做全量
> Attention, 无法通过 KV Cache 增量解码。CoT 推理、并发解码远未解决。

**别期待 A2D 之后模型变强。** 期待的是「学到 diffusion 的机制」。

---

## 12. 作者的项目结构 —— 直接对应本项目的 T2

```
dllm/
├── model/model_qwen3_dllm.py    ← 是 Qwen3 版!
├── dataset/lm_dataset.py
├── trainer/train_dllm.py
└── eval_dllm.py
```

```sh
# 训练
torchrun --nproc_per_node 2 trainer/train_dllm.py \
    --learning_rate 1e-4 --freeze_ffn --use_compile 1 \
    --batch_size 16 --epochs 5

# 推理
python eval_dllm.py --block_size 8 --steps 16 --temperature 0.3 --top_k 50
```

**作者已经做过 Qwen3 版。** 我们的 T2 线 (对 Qwen3-0.6B 做 A2D) 是有先例的。

---

## 13. 开放问题 (作者的清单)

- **长文本 block 策略** — 如何让 block 边界语义化, 而不是定长切分
- **与 speculative decoding 结合** — draft/verify 思路 + 多步去噪
- **多模态离散扩散** — 文本/图像/音频统一 token 化后共享同一套范式
- **训练效率** — 能否用更少步数逼近 AR 的生成质量

---

## 14. 参考文献

| 论文 | arXiv |
|---|---|
| Simple and Effective Masked Diffusion Language Models (**MDLM**) | 2406.07524 |
| Large Language Diffusion Models (**LLaDA**) | 2502.09992 |
| Scaling Diffusion Language Models via Adaptation from AR (**Dream**) | ICLR 2025 |
| Diffusion of Thoughts (**DoT**) | 2402.07754 |
| dLLM: Simple Diffusion Language Modeling (代码) | github.com/ZHZisZZ/dllm |
| MiniMind discussion #618 | minimind/discussions/618 |

权重: `dllm_768.pth` (ModelScope / HF `jingyaogong/minimind-3-pytorch`)
