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
from base.native.model import BASE, build_native_model

A2D_CELL = "native-minimind-a2d-example"
DLLM_CELL = "native-dllm-a2d-example"
DIRECT_CELL = "native-direct-dlm-example"
METHOD = "MiniMind-DLM-A2D"
DIRECT_METHOD = "Direct-DLM"


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
    """Same objective as the transformers row, different forward."""
    def loss_fn(model, batch: torch.Tensor):
        t = sample_t(batch.shape[0], batch.device, generator, epsilon=epsilon)
        t_safe = t.clamp_min(1e-3)
        x_t, masked = corrupt(batch, t, C.MASK, generator)
        return a2d_loss(native_forward(model)(x_t), batch, masked, t_safe), {
            "t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_native_dlm(model, prompts=C.PROMPTS, gen_len: int = 32, block_size: int = 8,
                      steps: int = 8, temperature: float = 0.3,
                      top_k: int = 50) -> dict:
    ids, texts, raw_texts = [], [], []
    logits_fn = native_forward(model)
    device = next(model.parameters()).device      # nn.Module has no .device
    for prompt in prompts:
        prompt_ids = torch.tensor([encode(prompt, bos=True)], device=device)
        out = sample_dlm(logits_fn, prompt_ids, gen_len, mask_id=C.MASK,
                         block_size=block_size, steps=steps,
                         temperature=temperature, top_k=top_k)
        generated = out[0, prompt_ids.shape[1]:]
        raw_texts.append(decode(generated))
        if C.EOS in generated:
            generated = generated[: int((generated == C.EOS).nonzero()[0])]
        ids.append(generated.tolist())
        texts.append(decode(generated))
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
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