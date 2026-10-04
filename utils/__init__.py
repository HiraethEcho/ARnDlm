"""utils -- 所有路线共同踩在上面的底座：配置、数据、种子、设备、训练循环、checkpoint、
指标、日志。

范围纠正（2026-10-05，step A0）：这一层**不是**「训练旁边的杂活」。`loop.py` 和
`data.py` 直接站在 loss 路径上 —— loss 函数收到的那个 batch 由这里产出，每一个结论里的
数字都在这里计时。实验版对自己也是这个说法：只有一份循环，A/B 差值才可归因。

模块（每个对应 example/common/ 里的一个文件；各自的建设顺序写在各文件自己的 docstring 里）：

    version.py          configs/<version>/<tier>.yaml -> dict；格子名 -> 输出目录
    config.py           ModelCfg / TrainCfg + 其余模块共用的常量
    seed.py             torch / numpy / random 三处都要设，少一处这次运行就不确定；
                        外加给 resume 用的 rng 快照
    device.py           ROCm 下的设备与 dtype 选择；宁可报错，不要静默退回 CPU
    data.py             memmap -> 定长打包块；能被 checkpoint 恢复的批次顺序
                        （DataLoader 的 sampler 恢复不了）
    loop.py             优化器 / LR 调度 / 计时区间 / 验证 / resume
    checkpoint.py       safetensors + 训练状态 + 参数普查；tied 存储要去重，
                        否则 safetensors 拒绝落盘
    metrics.py          两个口径的 tok/s（train-only 和 wall）、重复率
    logger.py           logs/run/<name>/summary.json（唯一真相）+ metrics.csv。
                        SwanLab 可选：没 export SWANLAB_API_KEY 就跳过，不要抛异常。
    tokenizer.py        encode / decode，默认不加 special token
    cli.py              --steps / --batch-size / --max-blocks / --data-dir / --resume

    web.py              不属于 step A0。它以后读 summary.json 做可视化。

**tok/s 必须同时报两个口径。** 验证和采样会污染 wall-clock 的分母 ——
notes/05-measurement.md 的 Bug 1。

**本版本读哪份配置。** configs/practical/*.yaml，后缀为空（实验版加 `-example`）。
checkpoints/ 和 logs/run/ 与 main worktree 共用，所以格子名不要互相覆盖。
现有 tier：`smoketest`（代码跑不跑得通）和 `tier1`（真正的对照，还没训）。

dtype 约定：`DTYPE` 来自配置的 `runtime:` 段，由 device.py 消费。别处不许自己选 dtype。
"""
