"""common/seed.py -- random / numpy / torch 三处随机源统一设定，外加可快照可恢复。

同一 seed 两次运行必须同序列；训练结束后快照供 resume 续接。
参考实现：common/seed-example.py
"""
