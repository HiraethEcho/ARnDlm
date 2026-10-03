"""Cells 6-8 -- the DLM methods on the native torch base.

Row 2 exists so the A2D columns have a second base to run on (SPEC 2.3) and so
the B version has a from-scratch model to compare against (teach/03). The
objective and the decoder are imported from `common/a2d.py`; what this row
contributes is the bidirectional forward and, for cell 6, dllm's loss applied to
a model that is not a transformers model.
"""
from __future__ import annotations

import torch

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_dlm, sample_t
from common.tokenizer import decode, encode
from base.native.model import build_native_model

A2D_CELL = "ar-3.2m-native-mini_a2d"
DLLM_CELL = "ar-3.2m-native-dllm_a2d"
METHOD = "MiniMind-DLM-A2D"


def native_forward(model, is_causal: bool = False):
    """Bidirectional logits for the native model.

    The native model takes `is_causal` directly, so there is no mask tensor to
    build -- which is why a bidirectional forward costs the same here as the
    transformers row's 4D-zero-mask path rather than more.
    """
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids, is_causal=is_causal).logits
    return logits_fn


def make_a2d_loss(model, generator: torch.Generator | None = None,
                  epsilon: float = 0.0):
    """Same objective as the transformers row, different forward.

    Accepts both batch shapes for the same reason row 1 does: [B, L] for a
    pretraining/A2D cell, [2, B, L] for an SFT cell whose row 1 carries IGNORE
    outside the assistant spans. Row 1 got this first and the native row kept only
    the [B, L] path for a while, which showed up as a shape error at dim 1 --
    `batch.shape[0]` was the channel axis. `test_dlm_sft.py` §3 now exercises both
    rows' factories with both shapes.
    """
    from common.sft import IGNORE

    def loss_fn(model, batch: torch.Tensor):
        if batch.dim() == 3:
            ids, labels = batch[0], batch[1]
            valid = labels != IGNORE
        else:
            ids, labels, valid = batch, batch, None
        t = sample_t(ids.shape[0], ids.device, generator, epsilon=epsilon)
        t_safe = t.clamp_min(1e-3)
        x_t, masked = corrupt(ids, t, C.MASK, generator, valid=valid)
        return a2d_loss(native_forward(model)(x_t), ids, masked, t_safe, valid=valid), {
            "t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_native_dlm(model, prompts=C.PROMPTS, gen_len: int = 32, block_size: int = 8,
                      steps: int = 8, temperature: float = 0.3,
                      top_k: int = 50,
                      prompt_ids: list[list[int]] | None = None,
                      prompt_labels: list | None = None) -> dict:
    """Block-wise sampling for the native row.

    `prompt_ids` takes over from `prompts` when the prefix cannot be spelled as
    one string -- the SFT cells, whose turn separator is EOS. Without it those
    cells sample from `user\n{q}\nassistant\n`, a prefix the model was never
    trained on, and a 200-step model answers it with an immediate EOS: rep-4
    comes back `nan` on an empty generation and the cell still reports OK.
    """
    ids, texts, raw_texts = [], [], []
    logits_fn = native_forward(model)
    device = next(model.parameters()).device      # nn.Module has no .device
    for prompt in (prompt_ids if prompt_ids is not None else
                   [[*encode(p, bos=True)] for p in prompts]):
        prompt_ids = torch.tensor([prompt], device=device)
        out = sample_dlm(logits_fn, prompt_ids, gen_len, mask_id=C.MASK,
                         block_size=block_size, steps=steps,
                         temperature=temperature, top_k=top_k)
        generated = out[0, prompt_ids.shape[1]:]
        raw_texts.append(decode(generated))
        if C.EOS in generated:
            generated = generated[: int((generated == C.EOS).nonzero()[0])]
        ids.append(generated.tolist())
        texts.append(decode(generated))
    labels = list(prompts) if prompt_labels is None else list(prompt_labels)
    return {"prompts": labels, "ids": ids, "texts": texts,
            "raw_texts": raw_texts,
            "params": {"gen_len": gen_len, "block_size": block_size, "steps": steps,
                       "temperature": temperature, "top_k": top_k, "kv_cache": False}}


def build_bidirectional(seed: int = 42):
    """A native model already switched to bidirectional attention.

    Setting it once here rather than per forward keeps the AR row and the DLM
    rows from accidentally sharing a model object.
    """
    model = build_native_model(C.TIER0, seed=seed)
    model.set_causal(False)
    return model


def trainable_fingerprint(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1)[:8] for p in model.parameters()])