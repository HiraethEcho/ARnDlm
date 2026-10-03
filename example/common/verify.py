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
from .data import PackedData, ids_of


@torch.no_grad()
def logits_delta(model_a: torch.nn.Module, model_b: torch.nn.Module,
                 data: PackedData, batches, device) -> float:
    """Largest absolute logit difference over `batches`. Both models in eval."""
    if len(batches) == 0:
        raise ValueError("logits_delta called with no batches -- it would "
                         "return 0.0 and read as 'identical'")
    worst = 0.0
    for idx in batches:
        batch = data.batch(idx).to(device)
        # An SFT batch is [2, B, L]; the model only ever sees the ids row.
        inputs = ids_of(batch)
        worst = max(worst, float((model_a(inputs).logits
                                  - model_b(inputs).logits).abs().max()))
    return worst


def verify_checkpoint(ckpt_dir: Path | str, build_fn, live_model: torch.nn.Module,
                      data: PackedData, batches, device) -> dict:
    """Rebuild the model from scratch, load the checkpoint, compare it to the
    model that is still in memory.

    The control matters: rebuilding twice and comparing those would return 0.0
    for the wrong reason. `live_model` is the only independent reference.

    `live_model` must already be on `device` and in fp32. An earlier version
    called `.to(device).float().eval()` on it and then restored only the train
    flag -- `to`/`float` mutate in place, so the comment promising not to move
    or cast the reference was describing something the code did not do. It was
    harmless at every call site (they all pass loop.device, and the master
    weights are fp32 by design), and would not have been on the next one. So
    those two requirements are asserted instead of silently applied.

    The train flag *is* handled: `Loop.fit` leaves the model in train mode, and
    this is called straight after it, so dropping into eval and putting it back
    is the only mutation done here.
    """
    live_device = next(live_model.parameters()).device
    want = torch.device(device)
    # `torch.device("cuda")` has index None while a real tensor is always
    # "cuda:0", so comparing the two objects directly would reject every caller.
    if live_device.type != want.type or (want.index is not None
                                         and live_device.index != want.index):
        raise RuntimeError(f"verify_checkpoint: live model is on {live_device}, "
                           f"not {torch.device(device)}. Moving it would mutate "
                           f"the caller's model.")
    if next(live_model.parameters()).dtype != torch.float32:
        raise RuntimeError("verify_checkpoint: live model must be fp32; casting "
                           "it in place would mutate the caller's model.")

    fresh = build_fn().to(device).float().eval()
    try:
        report = load_into(fresh, load_state_dict(ckpt_dir))
    except RuntimeError as exc:
        return {"ok": False, "reason": str(exc)}

    was_training = live_model.training
    live_model.eval()
    try:
        delta = logits_delta(fresh, live_model, data, batches, device)
    finally:
        live_model.train(was_training)
    return {"ok": delta <= 1e-5, "max_logit_delta": delta,
            "tensors": report["tensors"], "aliased_tied": report["aliased_tied"]}