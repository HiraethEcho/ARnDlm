"""ARnDlm — 七条实现路线共用的一层壳。

每个子包对应 ROADMAP.md 里的一条路线, 互不 import:
依赖方向永远是单向的 —— base <- utils <- {transformer, minimind}
                                     <- a2d <- dllm-a2d <- dllm

A2D 是在已有 AR 权重上做的, 所以 a2d/ 依赖 transformer/ 或 minimind/ 提供的
AR 模型, 但不依赖它们的训练循环。
"""
