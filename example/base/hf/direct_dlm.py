"""Cell 5/8 -- Direct-DLM: the same objective, no AR checkpoint.

The only difference from cell 3 is the initialisation, so the loss and the
decoder are the same functions -- they live in `common/a2d.py` and are imported
here rather than copied. A copy would let the two cells drift, and the whole
point of the pair is that exactly one variable differs.
"""
from __future__ import annotations

import torch

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_t
from base.hf.a2d_minimind import dlm_forward          # identical for both cells

CELL = "dlm-3.2m-hf"
BASE = "Qwen-mini-HF"
METHOD = "Direct-DLM"


def make_direct_loss(model, generator: torch.Generator | None = None,
                     epsilon: float = 0.0, logits_fn=None):
    """The MDLM objective. `epsilon` > 0 reproduces dllm's t range (for C2).

    `logits_fn` lets the native row reuse this objective with its own
    bidirectional forward instead of the transformers 4D-mask one.
    """
    logits_fn = logits_fn or dlm_forward(model)

    def loss_fn(model, batch: torch.Tensor):
        t = sample_t(batch.shape[0], batch.device, generator, epsilon=epsilon)
        t_safe = t.clamp_min(1e-3)          # 1/t at t~0 would explode
        x_t, masked = corrupt(batch, t, C.MASK, generator)
        return a2d_loss(logits_fn(x_t), batch, masked, t_safe), {
            "t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_direct(model, prompts=C.PROMPTS, gen_len: int = 32, block_size: int = 8,
                  steps: int = 8, temperature: float = 0.3, top_k: int = 50) -> dict:
    """Same decoder as the A2D cell; kept as a name so RESULTS rows read cleanly."""
    from base.hf.a2d_minimind import sample_dlm_row

    return sample_dlm_row(model, prompts=prompts, gen_len=gen_len,
                          block_size=block_size, steps=steps,
                          temperature=temperature, top_k=top_k)


def trainable_fingerprint(model) -> torch.Tensor:
    """A cheap signature of the initialisation, for the record.

    Two Direct-DLM runs that claim the same seed must have the same fingerprint;
    if they do not, the comparison is between two different initialisations and
    the loss difference means nothing.
    """
    return torch.cat([p.detach().reshape(-1)[:8] for p in model.parameters()])