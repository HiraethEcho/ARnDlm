"""sft_a2d_hf — 扩散 SFT: 掩码扩散目标 + assistant-only 监督。

对应教学计划 Step B2 (`example/teach/15-sft-train.md`) §B2.4。
**和 `../sft_ar_hf/` 共用同一份数据, 只换 objective。**

    data.py              复用 sft_ar_hf 的, 不重写
    loss.py              A2D loss, 但掩码只在 assistant span 内采样
    train.py             入口, 靠 --base 切权重来源

**为什么两个 base 共用一份代码。** SFT 阶段的 base (权重来源) 和
objective (训练目标) 是**两个正交的维度**: 同一份代码用 `--base` 换掉
上游 checkpoint, 用脚本本身决定 objective。写成两个各写一遍的脚本,
它们迟早会漂移, 而漂移看起来像模型 bug。

G3 比较的就是同一对 base 下两种 objective 的差别, 所以这两个维度必须正交。

**和 pretrain 阶段 A2D 唯一的差别**: 掩码**只在 assistant span 内采样**,
prompt 区域永远可见。否则模型学的是「补全任意前缀」, 不是「回答问题」。
掩码调度本身不变。

上游 cell 名**由程序带着**(查表), 不要从脚本自己的名字反推 ——
反推遇到命名规律之外的格子会崩, 而且崩之前会静默加载错权重。

依赖: transformers 4.57.0, ../../diffusion, ../sft_ar_hf (只借数据部分)"""
