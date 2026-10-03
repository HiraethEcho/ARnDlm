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


def load_into_any(model: torch.nn.Module, path: Path | str,
                  prefix: str = "model.") -> dict:
    """Load a checkpoint, tolerating one extra wrapper level.

    `dlm-3.2m-dllm` wraps its backbone in a `Bidirectional` module to flip the
    attention mask, so its state_dict carries an extra `model.` level that a plain
    `Qwen3ForCausalLM` does not have. SFT-ing that cell means the wrapper is the
    only difference.

    Deciding by inspection rather than by guessing: try the keys as they are, and
    only strip the prefix if that is what makes them fit. A blanket "strip
    `model.`" is wrong in the other direction -- a plain Qwen3 checkpoint's keys
    *all* start with `model.`, so stripping leaves `embed_tokens.weight` and the
    load fails on the first key. Both failures are silent in the sense that the
    right answer is not obvious from the error.
    """
    state = load_state_dict(path)
    want = set(model.state_dict())
    if all(k in want for k in state):
        return load_into(model, state)

    stripped = {k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state.items()}
    if all(k in want for k in stripped):
        if len(stripped) != len(state):
            raise ValueError(f"stripping {prefix!r} lost keys: "
                             f"{len(state)} -> {len(stripped)}")
        return load_into(model, stripped)

    raise ValueError(
        f"{path}: {len(state)} keys fit neither as-is nor with {prefix!r} "
        f"stripped. First keys: {sorted(state)[:3]} vs model {sorted(want)[:3]}")


def _storage(t: torch.Tensor):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))


def load_into(model: torch.nn.Module, state: dict[str, torch.Tensor],
              atol: float = 1e-5) -> dict:
    """Load with a completeness check that tolerates tied aliases.

    A tied checkpoint stores one tensor per storage, so `lm_head.weight` is
    absent on disk and refilled from the embedding -- `strict=True` (and a naive
    `strict=False` missing-key check) would reject every correct tied model.
    The returned report is what the cells print.
    """
    model_state = model.state_dict()
    absent = [k for k in model_state if k not in state]
    aliased = {k for k in absent
               if any(_storage(model_state[k]) == _storage(model_state[j]) for j in state)}
    missing = [k for k in absent if k not in aliased]
    unexpected = [k for k in state if k not in model_state]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.load_state_dict(state, strict=False)
    # No gap measurement here. An earlier version reported "max_abs_gap": 0.0 as a
    # literal -- a field that always reads "no drift" and so can never be the
    # reason a mismatch is caught. The check that matters is the strict
    # missing/unexpected comparison above, which raises.
    return {"tensors": len(state), "aliased_tied": sorted(aliased)}


def load_config(path: Path | str) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def describe(model: torch.nn.Module) -> dict:
    """Parameter census for meta.json.

    `params_unique` de-duplicates tied storage defensively: `parameters()`
    already collapses shared Parameters, but `state_dict()` does not -- it keeps
    both `model.embed_tokens.weight` and `lm_head.weight`. That difference is
    exactly what safetensors would refuse to serialise, so both numbers are
    reported and `tied` says whether they disagree.
    """
    seen, unique = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            unique += p.numel()
    return {
        "params_unique": unique,
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict_entries": len(model.state_dict()),
        "stored_tensors": len(_unique_state_dict(model)),
        "tied": len(_unique_state_dict(model)) < len(model.state_dict()),
    }


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