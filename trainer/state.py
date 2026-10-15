"""训练状态的存/取：step、optimizer、批次位置、RNG 快照（C4 resume）。

权重走 transformers 自己的 safetensors；这里只管额外的 pickle 状态。
参考实现：src/training/state-example.py
"""
