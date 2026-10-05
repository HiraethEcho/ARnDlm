# 一般的训练脚本调用流程

不特指本项目 —— 任何「单机写循环训一个 LM」的代码都长这个形状。

## 一、启动（解释器阶段）

| 步 | 干什么 | 资源 |
|---|---|---|
| 1 | import：解析配置文件、读分词器 | 纯 CPU |
| 2 | 设定随机数（python / numpy / torch 三处） | CPU |
| 3 | 构造模型：配置 → nn.Module，权重随机初始化 | CPU 先生成 |

**此刻显存占用 0** —— 模型还在 CPU。

## 二、进入训练前的准备

| 步 | 干什么 | 资源 |
|---|---|---|
| 1 | 数据集 tokenize、拼接、切定长块 | 磁盘读 + RAM（一次性） |
| 2 | 模型 `.to(device)` | CPU→GPU 一次拷贝 |
| 3 | 建 optimizer（参数分组，tied 权重去重） | 无 state，不占显存 |

optimizer 的 state（动量类）在**第一次 `step()` 时**才分配，不是建 optimizer 时。

## 三、训练主循环（每一步）

```
取一批数据 → 搬到 GPU → autocast 前向算 loss → backward → clip 梯度
→ optimizer.step() → zero_grad → (同步 + 计时)
```

- 逐步显存账：权重(4B/参) + 梯度(4B/参) + optimizer state(8B/参) + 激活(峰值临时)。
- 磁盘在步间是**零活动**：语料已整个在 RAM，写盘只在循环外。
- 计时口径：参数更新的时间才进 train-only 分母；验证/采样只进 wall-clock。

## 四、循环外（仍属一次运行）

| 动作 | 备注 |
|---|---|
| 验证 loss | 冻结批次 + fp32 + no_grad（bf16 舍入盖过要测的差异） |
| 采样 | `generate()`，切 eval 模式，结束还原 train |
| 存 checkpoint | 权重 GPU→CPU→磁盘 + 优化器/步数/数据位置/RNG 快照 |
| 写日志 | summary + 每步一行 csv |

## 五、可复现性三件套

1. 起点相同：seed 在**模型初始化之前**打
2. 中间可存：RNG 状态、optimizer、批次位置都要能落盘
3. 存的能续：resume 后逐位等价于一次跑完（任何 eval/采样消耗过随机数的事实要考虑进去）
