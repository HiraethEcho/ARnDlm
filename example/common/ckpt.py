"""Checkpoint I/O: a plain state_dict + config + train state, for every row.

Deliberately not `model.save_pretrained()`: the A2D and native rows need to load
each other's weights (C1 compares transformers Qwen3 against the native torch
port on the *same* tensors), so the on-disk format has to be row-agnostic.
safetensors is used because it refuses to serialise shared storage silently --
`tied_word_embeddings` would otherwise produce two keys pointing at one tensor.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _unique_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Drop tensors that share storage with an earlier key (tied embeddings)."""
    out, seen = {}, {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.device, tensor.shape)
        if key in seen:
            continue
        seen[key] = name
        out[name] = tensor.detach().to("cpu").contiguous()
    return out


def save_model(model: torch.nn.Module, out_dir: Path, config: dict, meta: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _unique_state_dict(model)
    save_file(sd, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir


def load_state_dict(path: Path | str) -> dict[str, torch.Tensor]:
    """Accept either a checkpoint dir or the safetensors file itself."""
    path = Path(path)
    f = path / "model.safetensors" if path.is_dir() else path
    return load_file(str(f))


def load_config(path: Path | str) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def save_train_state(path: Path | str, **state) -> Path:
    """step / optimizer / batch-stream position / RNG -- everything resume needs.

    torch.save (not safetensors): optimiser and RNG states are pickled python
    objects, not a tensor dict.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "train_state.pt")
    return path


def load_train_state(path: Path | str) -> dict:
    return torch.load(Path(path) / "train_state.pt", weights_only=False)