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

from .ckpt import load_into, load_state_dict
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
    try:
        report = load_into(fresh, load_state_dict(ckpt_dir))
    except RuntimeError as exc:
        return {"ok": False, "reason": str(exc)}

    # The live model is only a reference: do not leave it moved to another
    # device, cast, or stuck in eval mode for whatever the caller does next.
    was_training = live_model.training
    live = live_model.to(device).float().eval()
    delta = logits_delta(fresh, live, data, batches, device)
    if was_training:
        live_model.train()
    return {"ok": delta <= 1e-5, "max_logit_delta": delta,
            "tensors": report["tensors"], "aliased_tied": report["aliased_tied"]}