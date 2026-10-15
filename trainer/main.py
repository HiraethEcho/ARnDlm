"""训练主干：读配置 -> build -> 训练/验证/采样 -> 存盘 -> 摘要。

计时口径、日志、checkpoint 落盘都在这条主路上；换 trans/native 不改这里。
参考实现：src/training/main-example.py
"""
