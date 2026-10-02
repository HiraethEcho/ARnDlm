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


def sample_t(batch_size: int, device, generator: torch.Generator | None = None,
             epsilon: float = 0.0) -> torch.Tensor:
    """One corruption level per sample, uniform in [0, 1).

    `epsilon` shifts the range to [epsilon, 1), which is what dllm's
    MDLMTrainer does (`t = time_epsilon + (1 - time_epsilon) * rand`). The two
    differ by ~0.1% in E[1/t], but for the C2 comparison the *draw order* has to
    match as well, so C2 passes the same epsilon and reseeds.
    """
    t = torch.rand(batch_size, device=device, generator=generator)
    return epsilon + (1.0 - epsilon) * t


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


def gumbel_argmax(logits: torch.Tensor, temperature: float = 1.0,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-max sampling: returns (token, confidence).

    Numerically stabler than softmax+multinomial at low precision
    (notes/03 §7). temperature=0 degenerates to argmax. `generator` makes the
    draw reproducible, which C2 needs when two implementations sample the same
    prompt. Note a CUDA tensor needs a CUDA generator.

    Always float32, even when the logits are bf16: `rand` at bf16 can round u
    to exactly 1.0, and `-log(-log(1.0))` is +inf, which would hand every
    surviving candidate an infinite perturbation.
    """
    if temperature <= 0:
        conf, token = F.softmax(logits.float(), dim=-1).max(dim=-1)
        return token, conf
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32,
                   generator=generator).clamp_(1e-20, 1.0)
    gumbel = -torch.log(-torch.log(u))
    perturbed = logits.float() / temperature + gumbel
    token = perturbed.argmax(dim=-1)
    # Confidence has to be the model's actual probability of the token it
    # committed to, not the perturbed score -- otherwise the reveal order is
    # driven by the noise and not by the model.
    confidence = F.softmax(logits.float(), dim=-1).gather(
        -1, token.unsqueeze(-1)).squeeze(-1)
    return token, confidence


def top_k_filter(logits: torch.Tensor, top_k: int,
                  suppress: int | tuple | None = None) -> torch.Tensor:
    """Keep the k best candidates; optionally forbid specific ids outright.

    `suppress` matters: MASK lives inside the vocabulary, so a masked position
    can sample MASK, "reveal" it, and burn one unit of the reveal budget while
    the position stays masked. The loop would then exit with MASK in the output.
    """
    if suppress is not None:
        ids = (suppress,) if isinstance(suppress, int) else tuple(suppress)
        logits = logits.clone()
        logits[..., list(ids)] = -float("inf")
    if not top_k or top_k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


@torch.no_grad()
def sample_dlm(logits_fn, prompt_ids: torch.Tensor, gen_len: int, *,
               mask_id: int = MASK, block_size: int = 8, steps: int = 8,
               temperature: float = 0.3, top_k: int = 50,
               max_context: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Block-wise confidence decoding (notes/03 §5).

    * every step re-runs the **whole** sequence -- bidirectional attention needs
      all known context and there is no KV cache to reuse
    * the number revealed per step is `remaining / remaining_steps`, computed
      dynamically, not a fixed top-k
    * positions outside the current block are frozen even if they look more
      confident, which is what makes block_size a quality/speed knob rather
      than a mode switch

    The reveal bookkeeping stays on the GPU and never reads a tensor back to
    the host: `int(mask.any())` costs ~20 ms of synchronisation stall per step
    against 1.3 ms for the forward itself (measured). The number of remaining
    masks is therefore tracked in python integers, derived from the reveal
    budget rather than queried from the tensor.
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    total = prompt_len + gen_len
    if max_context is not None and total > max_context:
        # Every step feeds the whole sequence, so exceeding the context is not a
        # truncation the model can absorb -- RoPE positions run past the trained
        # range. Fail loudly instead of emitting plausible-looking garbage.
        raise ValueError(f"prompt {prompt_len} + gen {gen_len} exceeds "
                         f"max_context {max_context}")
    x = torch.cat([prompt_ids,
                   torch.full((prompt_ids.shape[0], gen_len), mask_id,
                              device=device, dtype=prompt_ids.dtype)], dim=1)

    for start in range(prompt_len, total, block_size):
        end = min(start + block_size, total)
        # The generation region starts fully masked, so the first count is exact.
        remaining = [end - start] * x.shape[0]
        steps_left = steps
        while steps_left > 0 and max(remaining) > 0:
            full_logits = logits_fn(x)
            # Sample from the truncated distribution, but score confidence on the
            # untouched logits: top-k renormalises inside the surviving
            # candidates, which would reorder positions whose top-k mass differs.
            token, _ = gumbel_argmax(top_k_filter(full_logits, top_k, suppress=mask_id),
                                     temperature, generator)
            confidence = F.softmax(full_logits.float(), dim=-1).gather(
                -1, token.unsqueeze(-1)).squeeze(-1)

            for b in range(x.shape[0]):
                if remaining[b] == 0:
                    continue
                active = x[b, start:end] == mask_id
                # Only the current block may be revealed, and the budget is per
                # sequence: "remaining / remaining_steps" (notes/03 §5).
                n_unmask = max(1, math.ceil(remaining[b] / steps_left))
                conf = torch.where(active, confidence[b, start:end],
                                   torch.full_like(confidence[b, start:end], -float("inf")))
                chosen = conf.topk(min(n_unmask, conf.numel())).indices
                x[b, start + chosen] = token[b, start + chosen]
                remaining[b] -= int(chosen.numel())
            steps_left -= 1
    return x