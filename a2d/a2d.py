"""A2D 共享层：corruption / 加权 loss / 块状揭示采样器（mini 与 dllm、trans 与 native 四条组合共用这一份）。

复制一份会让 C2/C3 无法把 loss 差归因到任何东西（copy-instead-of-import 反模式）。
参考实现：common/a2d-example.py
"""
