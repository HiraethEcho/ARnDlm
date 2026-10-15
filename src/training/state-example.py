"""src/training/state-example.py -- trainer/state.py 的参考实现。

训练状态的存/取: step、optimizer、批次位置、RNG 快照（C4 resume 的另一半;
权重走 save_pretrained 的 safetensors, 这里只管额外状态 pickle）。
"""
from __future__ import annotations

from pathlib import Path

import torch

# ---- 连字符文件名不能被 import: 显式装载（x.py 填完后用正常 import）----
import importlib.util as _ilu
_ROOT = Path(__file__).resolve().parents[2]
_spec = _ilu.spec_from_file_location("seed_example", _ROOT / "common/seed-example.py")
_seed = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_seed)
restore, snapshot = _seed.restore, _seed.snapshot


def save_state(path: str | Path, *, step: int, optimizer: torch.optim.Optimizer,
               data_position: tuple[int, int]) -> None:
    """train_state.pt: 一切「权重之外、resume 之内」的东西。"""
    opt_state = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "data_position": list(data_position),  # (epoch, within) —— 与 step 互为校验
        "rng": snapshot(),
        "optimizer": opt_state,
    }, path)


def load_state(path: str | Path, optimizer: torch.optim.Optimizer) -> dict:
    """恢复 RNG 与 optimizer; 返回 {"step", "data_position"}。文件不存在 -> ValueError。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"train_state 不存在: {p} (无状态可续, 应从头训)")
    blob = torch.load(p, weights_only=False)
    restore(blob["rng"])
    optimizer.load_state_dict(blob["optimizer"])
    return {"step": blob["step"], "data_position": tuple(blob["data_position"])}
