"""src/models/native/model.py -- 手写 Qwen-style decoder（RoPE / RMSNorm / SwiGLU / GQA / QK-Norm / tied）。

验收：与 trans 同权重同 batch logits max|Δ| <= 1e-5（SPEC C1）。
参考实现：src/models/native/model-example.py
"""
