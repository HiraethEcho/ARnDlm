"""utils — 功能性代码: 日志、web 展示、设备、随机种子、checkpoint。

与 base 的区别: base 是「训练要用的算法」, utils 是「训练过程要记录和
展示的东西」。utils 不参与 loss 计算。

    logger.py           写 logs/run/<name>/summary.json (唯一真相) + metrics.csv
                        SwanLab 可选: 没 export SWANLAB_API_KEY 就跳过
    web.py              训练可视化 —— 读 summary.json 起一个本地 web 页面
    device.py           ROCm 下 device / dtype 的选择
    seed.py             torch / numpy / random 三处都要设, 少一处就不确定
    checkpoint.py       存 / 续。存 model + optimizer + rng, 否则续训不是同一条
    metrics.py          tok/s (train-only 和 wall 两个口径)、重复率

**tok/s 必须同时报两个口径。** 验证和采样会污染 wall-clock 分母。
见 notes/05-measurement.md 的 Bug 1。
"""
