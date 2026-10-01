"""Reload a checkpoint into a fresh model and compare logits.

Acceptance item 3 in example/PLAN.md §6 says a checkpoint must load back
identically. `test_hf_ar.py` proves that on CPU with a hand-built checkpoint;
this runs the same comparison on the real training run, in fp32, on the
evaluation batches -- which is the only place a bug that only appears after
200 steps (a dtype left in bf16, an optimizer-only buffer, a dropped tied
tensor) would show up.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .ckpt import load_state_dict
from .data import PackedData


@torch.no_grad()
def logits_delta(model_a: torch.nn.Module, model_b: torch.nn.Module,
                 data: PackedData, batches, device) -> float:
    """Largest absolute logit difference over `batches`. Both models in eval."""
    worst = 0.0
    for idx in batches:
        batch = data.batch(idx).to(device)
        worst = max(worst, float((model_a(batch).logits
                                  - model_b(batch).logits).abs().max()))
    return worst


def verify_checkpoint(ckpt_dir: Path | str, build_fn, live_model: torch.nn.Module,
                      data: PackedData, batches, device) -> dict:
    """Rebuild the model from scratch, load the checkpoint, compare it to the
    model that is still in memory.

    The control matters: rebuilding twice and comparing those would return 0.0
    for the wrong reason. `live_model` is the only independent reference.
    """
    fresh = build_fn().to(device).float().eval()
    state = load_state_dict(ckpt_dir)
    model_state = fresh.state_dict()

    # A tied checkpoint legitimately omits `lm_head.weight`: safetensors stores
    # one tensor per storage, and `load_state_dict` fills the alias from the
    # embedding. Counting that as "missing" would fail every tied model, so
    # only names that are absent *and* unaliased count.
    absent = [k for k in model_state if k not in state]
    # state_dict() returns detached *views*, so `is` never matches; compare
    # storage identity instead.
    def _storage(t):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))

    aliased = {k for k in absent
               if any(_storage(model_state[k]) == _storage(model_state[j]) for j in state)}
    missing = [k for k in absent if k not in aliased]
    unexpected = [k for k in state if k not in model_state]
    if missing or unexpected:
        return {"ok": False, "reason": f"missing={missing} unexpected={unexpected}"}
    fresh.load_state_dict(state, strict=False)   # aliases fill in from the embedding

    live = live_model.to(device).float().eval()
    delta = logits_delta(fresh, live, data, batches, device)
    return {"ok": delta <= 1e-5, "max_logit_delta": delta, "tensors": len(state),
            "aliased_tied": sorted(aliased)}