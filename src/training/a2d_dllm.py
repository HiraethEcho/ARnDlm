"""src/training/a2d_dllm.py -- A2D，调 dllm 包的 MDLMTrainer（G5 的 dllm 侧；转换零漂移 C5）。

5.x 的 create_causal_mask 签名变了，transformers 钉 4.57.0。
参考实现：src/training/a2d_dllm-example.py
"""
