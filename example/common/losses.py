"""Losses shared by both base rows.

Row 1 (transformers) and row 2 (native torch) must produce the same numbers, so
the objective lives here rather than inside either row's training script -- two
copies would eventually drift, and the drift would look like a model bug.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


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