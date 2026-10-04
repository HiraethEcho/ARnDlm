"""utils/checkpoint.py -- 存/读模型权重 (safetensors) 和训练状态 (torch.save)。

格式与实现路线无关: A2D 或 native 的格子都要能读同一份文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _unique_state_dict(model: torch.nn.Module) -> dict:
    # state_dict 里 tied 权重出现两个键; 共享存储的只留第一个
    out, seen = {}, {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.device, tensor.shape)
        if key in seen:
            continue
        seen[key] = name
        out[name] = tensor.detach().to("cpu").contiguous()
    return out


def save_model(model: torch.nn.Module, out_dir: Path, config: dict, meta: dict | None = None) -> Path:
    # 权重 + config + meta 三个文件落到同一目录
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _unique_state_dict(model)
    save_file(sd, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir


def load_state_dict(path: Path | str) -> dict:
    # 参数可以是目录也可以是文件本身
    path = Path(path)
    f = path / "model.safetensors" if path.is_dir() else path
    return load_file(str(f))


def _storage(t: torch.Tensor):
    # 判断两个键是否同一块存储
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))


def load_into(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict:
    # 载入权重; tied 的别名键当作已满足, 其余缺键/多键报错
    model_state = model.state_dict()
    absent = [k for k in model_state if k not in state]
    aliased = {k for k in absent
               if any(_storage(model_state[k]) == _storage(model_state[j]) for j in state)}
    missing = [k for k in absent if k not in aliased]
    unexpected = [k for k in state if k not in model_state]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.load_state_dict(state, strict=False)
    return {"tensors": len(state), "aliased_tied": sorted(aliased)}


def describe(model: torch.nn.Module) -> dict:
    # 参数普查: 去重后的参数量 / 可训练量 / state_dict 键数 / 存储数 / 是否 tied
    seen, unique = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            unique += p.numel()
    sd = model.state_dict()
    stored = _unique_state_dict(model)
    return {
        "params_unique": unique,
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict_entries": len(sd),
        "stored_tensors": len(stored),
        "tied": len(stored) < len(sd),
    }


def save_train_state(path: Path | str, **state) -> Path:
    # step / optimizer / 流位置 / RNG 这些 pickle 对象用 torch.save 存
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "train_state.pt")
    return path


def load_train_state(path: Path | str) -> dict:
    return torch.load(Path(path) / "train_state.pt", weights_only=False)
