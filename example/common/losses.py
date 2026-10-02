"""Losses shared by both base rows.

Row 1 (transformers) and row 2 (native torch) must produce the same numbers, so
the objective lives here rather than inside either row's training script -- two
copies would eventually drift, and the drift would look like a model bug.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sft import IGNORE


def ar_loss(model, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Next-token cross entropy on a packed block.

    A block is [BOS] doc [EOS] [BOS] doc [EOS] ... so shifting by one keeps each
    document self-contained, and the loss on the EOS position is what teaches
    the model to stop.
    """
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, {"target_tokens": int(targets.numel())}


def sft_ar_loss(model, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Next-token cross entropy on **assistant tokens only** (GOAL.md Q4).

    `batch` is [2, B, L]: row 0 input ids, row 1 labels with IGNORE outside the
    assistant spans. Three details are load-bearing:

    * targets come from the **labels** row, not the ids row. Scoring ids would
      train the model to reproduce the token it was just handed.
    * the shift is applied to both rows together, so a label at index i is scored
      from the logits at i-1. A label at index 0 has nothing before it and can
      never be scored -- `block_atomic` keeps BOS unsupervised for exactly this.
    * `reduction='mean'` divides by the number of supervised tokens **in the
      batch**. Blocks carry different assistant fractions, so a per-sample mean
      would weight a 10%-supervised block like a 90% one.

    Returns a zero loss (not a nan gradient) when nothing is supervised:
    `F.cross_entropy` with every target ignored returns nan, and one nan step
    destroys the run while the loss curve still looks plausible for a while.
    """
    ids, labels = batch[0], batch[1]
    inputs, targets = ids[:, :-1], labels[:, 1:]
    n_target = int((targets != IGNORE).sum())
    if n_target == 0:
        zero = model(inputs).logits.sum() * 0.0        # keeps the graph alive
        return zero, {"target_tokens": 0, "skipped": True}
    logits = model(inputs).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE)
    return loss, {"target_tokens": n_target, "skipped": False}