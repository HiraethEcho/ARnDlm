"""dlm-3.2m-dllm -- Direct-DLM driven by dllm's MDLMTrainer.

Per SPEC 1.2 the `dlm-` family has exactly two implementations, `hf` (this
project's own masked CE on a transformers model) and `dllm` (the package's
trainer). There is no `native` variant: a third modelling implementation would
re-ask the question G1 already answered.

The interesting part is the adapter. `MDLMTrainer.compute_loss` calls

    model(input_ids=noised, attention_mask=None)

and a plain `Qwen3ForCausalLM` would answer causally -- a masked diffusion
model that only sees its left context, which trains fine and is not a diffusion
model at all. So the model is wrapped in a thin module that turns the missing
mask into an explicit all-zeros 4D mask, the same trick the hand-written row
uses. The upstream loss code is untouched; only the model's answer to "is this
causal?" is fixed.

That also makes this cell the missing third arm of the G5 comparison: dlm-hf and
dlm-dllm must agree on the same corruption, exactly as the two A2D rows do.
"""
from __future__ import annotations

import torch
from torch import nn

from common import config as C
from base.hf.a2d_minimind import dlm_forward, sample_dlm_row
from base.hf.a2d_dllm import DLLMLoss
from base.hf.direct_dlm import make_direct_loss

CELL = "dlm-3.2m-dllm"
BASE = "Qwen-mini-HF"
METHOD = "Direct-DLM"


class Bidirectional(nn.Module):
    """Wrap a causal HF model so every position attends to every position.

    Keeps `input_ids` / `attention_mask` as keyword arguments, which is the
    calling convention `MDLMTrainer.compute_loss` uses; a 2D mask still means
    "padding only" and a 4D one is passed through.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        t = input_ids.shape[1]
        if attention_mask is None or attention_mask.ndim == 2:
            attention_mask = torch.zeros(input_ids.shape[0], 1, t, t,
                                        device=input_ids.device, dtype=self.model.dtype)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def make_dllm_loss(mask_id: int = C.MASK, epsilon: float = 1e-3):
    """dllm's loss driving the wrapped model, with the Qwen-aware stub ids."""
    harness = DLLMLoss(special_ids={"mask_token_id": mask_id})

    def loss_fn(model, batch: torch.Tensor):
        return harness(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}
    return loss_fn


# Re-exports: a Direct-DLM cell built through the dllm package needs the same
# objective pieces as the hand-written one, and both live in those two modules.
# Kept in __all__ so `from ... import *` in a notebook still sees them; nothing
# here calls them directly.
__all__ = ["CELL", "BASE", "METHOD", "Bidirectional", "make_dllm_loss",
           "make_direct_loss", "dlm_forward", "sample_dlm_row"]