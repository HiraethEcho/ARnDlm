"""common/device.py -- 设备选择（ROCm 下仍是 cuda API）与 dtype 转换、autocast 上下文。

宁可报错也不静默退回 CPU；eval 强制 fp32（跨实现对拍是硬要求）。
参考实现：common/device-example.py
"""
