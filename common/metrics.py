"""common/metrics.py -- 两个口径的吞吐（train-only / wall）、rep-4 重复率、loss 首尾窗口。

测量口径是 SPEC 6.4 的硬要求；不用 torch。
参考实现：common/metrics-example.py
"""
