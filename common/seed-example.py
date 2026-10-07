"""common/seed-example.py -- common/seed.py 的参考实现。

random / numpy / torch 三处随机源统一设定；可快照可恢复（C4 resume 的根）。
同 seed 同输入必须逐位同序列。

注意：numpy 有两套 RNG（遗留 np.random.* 与新 Generator）。本项目全部用遗留
np.random.*（data 层的 permutation 用它），Generator 不在此管理。
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """三个随机源一次设齐。cuda 存在时也设（ROCm 走 cuda API）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 同时覆盖 cpu 与所有 cuda 设备
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot() -> dict:
    """全部 RNG 状态快照。存进 checkpoint（state.py），resume 时 restore。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore(state: dict) -> None:
    """按 snapshot() 的产物恢复。键集合必须匹配，防止静默丢随机源。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


if __name__ == "__main__":
    seed_all(42)
    a = [torch.randn(3), np.random.rand(3), random.random()]
    snap = snapshot()
    b = [torch.randn(3), np.random.rand(3), random.random()]
    restore(snap)
    c = [torch.randn(3), np.random.rand(3), random.random()]
    for x, y, name in zip(b, c, ("torch", "numpy", "python")):
        same = torch.equal(x, y) if isinstance(x, torch.Tensor) else (list(x) == list(y) if name == "numpy" else x == y)
        print(f"{name}: restore 后逐位一致 = {same}")
