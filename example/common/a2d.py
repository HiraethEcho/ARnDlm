"""MiniMind-DLM style A2D: corruption, weighted loss, block sampler.

Shared by both base rows on purpose. Row 1 and row 2 must differ only in the
model, so the objective and the sampler live here and each row supplies a
`logits_fn`. Two copies would make C2/C3 unable to attribute a loss difference
to anything.

Formulas follow notes/03-minimind-dlm.md §4-§5, which also note that this is
the same family as MDLM -- no ELBO, no log-SNR, no weighting schedule beyond the
1/p_mask reweighting:

    t ~ U[0,1)                       per sample, one scalar
    x_t = MASK with probability t    per token
    L   = sum_i [ 1_{masked_i} * CE(f(x_t)_i, x_0(i)) / t ] / N_valid

Two details that are easy to get wrong and that test/04 already pins down:

* the denominator is the number of *valid* tokens, not the number of masked
  ones -- otherwise the loss drifts with the mask ratio while still falling;
* `p_mask` is a per-sample **scalar**, not a per-token random draw. Passing a
  [B, L] vector instead mixes the weights inside a sample and the closed-form
  comparison fails.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import MASK

EPS = 1e-6          # dllm/core/schedulers/alpha.py hardcodes the same value


def sample_t(batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """One corruption level per sample, uniform in [0, 1)."""
    return torch.rand(batch_size, device=device, generator=generator)


def corrupt(x0: torch.Tensor, t: torch.Tensor, mask_id: int = MASK,
            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask each token with probability t[b]. Returns (x_t, masked)."""
    probs = torch.rand(x0.shape, device=x0.device, generator=generator)
    masked = probs < t[:, None]                       # [B, L], t is per-sample
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def a2d_loss(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor,
             t: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Reweighted cross entropy on masked positions only.

    Dividing by t balances the gradient contribution across corruption levels:
    a small t means few masked positions but each one matters more, and
    without the division the loss would be dominated by large t.
    """
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          x0.reshape(-1), reduction="none").view_as(x0)
    weight = (1.0 / (t + EPS)).unsqueeze(1)           # [B, 1] -> broadcasts over L
    per_token = nll * weight * masked * valid
    return per_token.sum() / valid.sum().clamp_min(1)


def gumbel_argmax(logits: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-max sampling: returns (token, confidence).

    Numerically stabler than softmax+multinomial at low precision
    (notes/03 §7). temperature=0 degenerates to argmax and returns confidence 1.
    """
    if temperature <= 0:
        conf, token = F.softmax(logits, dim=-1).max(dim=-1)
        return token, conf
    logits = logits.to(torch.float64)
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64).clamp_(1e-20, 1.0)
    gumbel = -torch.log(-torch.log(u))
    perturbed = logits / temperature + gumbel
    conf, token = perturbed.max(dim=-1)
    # Confidence has to be the model's actual probability of the token it
    # committed to, not the perturbed score -- otherwise the reveal order is
    # driven by the noise and not by the model.
    probs = F.softmax(logits, dim=-1)
    confidence = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    return token, confidence


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if not top_k or top_k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


@torch.no_grad()
def sample_dlm(logits_fn, prompt_ids: torch.Tensor, gen_len: int, *,
               mask_id: int = MASK, block_size: int = 8, steps: int = 8,
               temperature: float = 0.3, top_k: int = 50,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Block-wise confidence decoding (notes/03 §5).

    * every step re-runs the **whole** sequence -- bidirectional attention needs
      all known context and there is no KV cache to reuse
    * the number revealed per step is `remaining / remaining_steps`, computed
      dynamically, not a fixed top-k
    * positions outside the current block are frozen even if they look more
      confident, which is what makes block_size a quality/speed knob rather
      than a mode switch
    """
    device = prompt_ids.device
    total = prompt_ids.shape[1] + gen_len
    x = torch.cat([prompt_ids,
                   torch.full((prompt_ids.shape[0], gen_len), mask_id,
                              device=device, dtype=prompt_ids.dtype)], dim=1)
    prompt_len = prompt_ids.shape[1]

    for start in range(prompt_len, total, block_size):
        end = min(start + block_size, total)
        steps_left = steps
        while steps_left > 0:
            if not bool((x[:, start:end] == mask_id).any()):
                break

            logits = top_k_filter(logits_fn(x), top_k)
            token, confidence = gumbel_argmax(logits, temperature)

            # Only the current block may be revealed this step, and the budget
            # is per sequence: "remaining / remaining_steps" from notes/03 §5.
            for b in range(x.shape[0]):
                active = x[b, start:end] == mask_id
                n_active = int(active.sum())
                if n_active == 0:
                    continue
                n_unmask = max(1, math.ceil(n_active / steps_left))
                conf = confidence[b, start:end].clone()   # block slice, not the row
                conf[~active] = -float("inf")
                order = torch.argsort(conf, descending=True)[:n_unmask]
                chosen = order[active[order]]
                x[b, start + chosen] = token[b, start + chosen]
            steps_left -= 1
    return x