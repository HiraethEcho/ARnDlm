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

METHOD = "dLLM-A2D"


def make_dllm_loss(model, epsilon: float = 1e-3):
    """dllm's loss, unmodified, driving the native model.

    Requires the model to already be bidirectional. dllm's MDLMTrainer forwards
    with no mask, and the native port takes bidirectionality through
    `self._last_is_causal` rather than a mask tensor -- so a factory that quietly
    accepted a causal model would train a causal model under a cell name that
    says A2D. That is exactly what happened once: `train_a2d_from_sft.py` used
    the plain builder and four `*-sft-dllm_a2d` cells came out causal. Checked
    here so the next one fails immediately instead of showing up as a conclusion.
    """
    if getattr(model, "_last_is_causal", True):
        raise ValueError(
            "make_dllm_loss needs a bidirectional model: call "
            "model.set_causal(False) first. dllm's loss forwards with no mask, "
            "so causality would come from the model and this would silently be "
            "an AR cell named A2D.")
    harness = DLLMLoss(time_epsilon=epsilon)

    def loss_fn(model, batch):
        return harness(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}
    return loss_fn