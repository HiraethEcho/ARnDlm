"""a2d — 用 minimind-dlm 的思路手动实现 A2D。

对应教学计划 Step D5 (`example/teach/08-a2d-minimind.md`)。原理全在 notes/03-minimind-dlm.md。

A2D = AR-to-Diffusion, 两步:
    1. is_causal: True -> False        关掉因果约束
    2. vocab += [MASK]                 扩词表

然后用 DLM 目标微调:
    corrupt(input_ids, t)  按 p_mask = 1 - alpha(t) 随机替换成 MASK
    dllm_loss               只在被 mask 位置算, 除以 p_mask 重加权

    convert.py             AR 权重 -> DLM 权重 (上面那两步)
    corruption.py          加噪
    loss.py                加权 CE
    train.py               微调循环
    sample.py              分块去噪采样 (block_size / steps / temperature / top_k / cfg)

**除以 p_mask 不能省。** 不除的话 loss 偏向大 t, 不同 t 之间梯度贡献失衡。

**必须在同一 checkpoint 上与 dllm-a2d/ 跑出同一个 loss**, 否则不知道是谁错了。

依赖: torch (原生), ../../diffusion, ../../utils
"""
