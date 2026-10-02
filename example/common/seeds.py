"""Seeding, device and dtype -- the three things that silently differ between runs.

logs/dev/01-local-state.md records the worst one: torch tensors default to CPU,
`torch.cuda.is_available()` still returns True, and nothing looks wrong. So
device selection *prints* what it picked and `require_cuda` makes it fatal.
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed every RNG the training path can touch.

    numpy needs its own call: a torch.Generator seeds torch only, and the data
    pipeline draws permutation indices from torch, not numpy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """Snapshot enough RNG state to make a resumed run bit-identical."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def pick_device(require_cuda: bool = True) -> torch.device:
    """Return the training device, refusing to silently fall back to CPU."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require_cuda=True but torch.cuda.is_available() is False")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu only'})")
    return device


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def autocast(device: torch.device, name: str = "bfloat16"):
    """Context manager for mixed precision; a no-op on CPU.

    Master weights stay fp32 (bf16 Adam states lose too many small updates at
    these step counts); only the matmuls run in bf16.
    """
    if device.type != "cuda" or name == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_DTYPES[name])