"""Cell 6 -- dLLM's objective on the native torch base.

dllm's `MDLMTrainer.compute_loss` is written against any callable model: it
only needs `model(input_ids=..., attention_mask=...)` to return something with
`.logits`. The native model satisfies that once it accepts the keyword and is
switched to bidirectional attention.

This is the cell that shows the dllm path is not transformers-specific -- and
the one place where a package and a hand-written model meet at an interface
neither author designed for.
"""
from __future__ import annotations

from base.hf.a2d_dllm import DLLMLoss
from base.native.a2d_minimind import (DLLM_CELL, build_bidirectional,
                                       sample_native_dlm)

METHOD = "dLLM-A2D"


def make_dllm_loss(model, epsilon: float = 1e-3):
    """dllm's loss, unmodified, driving the native model."""
    harness = DLLMLoss(time_epsilon=epsilon)

    def loss_fn(model, batch):
        return harness(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}
    return loss_fn