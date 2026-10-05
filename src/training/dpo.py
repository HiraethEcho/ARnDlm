"""src/training/dpo.py -- DPO 偏好对（chosen / rejected 同 prompt）的训练（SPEC 网格 rl 列）。

只在 AR 行上做（DLM 无序列 logprob，需要 ELBO 形式化才可能）。参考实现：
src/training/dpo-example.py
"""
