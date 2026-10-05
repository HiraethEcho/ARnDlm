"""common/device-example.py -- common/device.py 的参考实现。

设备选择（ROCm 下仍是 cuda API）+ dtype 名到 torch.dtype 的转换 + autocast 上下文。
宁可报错也不静默退回 CPU（torch 默认 CPU 且 cuda.is_available()=True 时毫无异常，
是 legacy 踩过的坑）。eval 强制 fp32：调用方硬传 "float32"，本模块不 import config。

约定：本模块不读配置。训练传 config.DTYPE，eval 传 "float32"（2026-10-04 A 方案）。
"""
from __future__ import annotations

import contextlib

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def pick_device() -> torch.device:
    """有 cuda 用 cuda；没有就报错，绝不静默回 CPU。"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuda 不可用（ROCm 环境下也是查 cuda API）。"
            "项目不做 CPU 路线 —— 宁可停，也不要在 CPU 上静默跑出假数字。"
        )
    return torch.device("cuda")


def to_dtype(name: str) -> torch.dtype:
    """"bfloat16" -> torch.bfloat16。未知名字报错，不猜。"""
    if name not in _DTYPES:
        raise ValueError(f"未知 dtype: {name!r} (可选: {sorted(_DTYPES)})")
    return _DTYPES[name]


def autocast(name: str):
    """按 dtype 名开 autocast。eval 传 "float32" 得到 nullcontext（fp32 就是关掉混合精度）。"""
    if name == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=to_dtype(name))


if __name__ == "__main__":
    dev = pick_device()
    print(f"device = {dev} ({torch.cuda.get_device_name(0)})")
    with autocast("bfloat16"):
        x = torch.randn(4, 4, device=dev)
        y = x @ x
    print(f"bf16 autocast 下 matmul dtype = {y.dtype}")
    with autocast("float32"):
        z = x @ x
    print(f"fp32(eval) 下 matmul dtype = {z.dtype}")
