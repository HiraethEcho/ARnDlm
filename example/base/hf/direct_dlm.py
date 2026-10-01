"""Cell 5/8 -- Direct-DLM: the same objective, no AR checkpoint.

The only difference from cell 3 is the initialisation. Everything else --
objective, schedule, data, decode settings -- is identical, so the pair
(cell 3, cell 5) isolates one variable: does starting from AR weights buy
anything at this scale?

Expect it not to. notes/03 §11 reports the same: A2D inherits the AR
*mechanism*, and from-scratch DLM has to learn it. The smoke run only needs the
pipeline to hold together; the interesting column is the eval loss gap.
"""
from __future__ import annotations

import torch

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_dlm, sample_t
from common.tokenizer import decode, encode

CELL = "hf-direct-dlm-example"
BASE = "Qwen-mini-HF"
METHOD = "Direct-DLM"


def dlm_forward(model):
    """Same bidirectional path as the A2D cell (see a2d_minimind.py)."""
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        mask4d = torch.zeros(input_ids.shape[0], 1, t, t,
                             device=input_ids.device, dtype=model.dtype)
        return model(input_ids, attention_mask=mask4d).logits
    return logits_fn


def make_direct_loss(model, generator: torch.Generator | None = None,
                     epsilon: float = 0.0):
    """The MDLM objective. `epsilon` > 0 reproduces dllm's t range (for C2)."""
    def loss_fn(model, batch: torch.Tensor):
        t = sample_t(batch.shape[0], batch.device, generator, epsilon=epsilon)
        t_safe = t.clamp_min(1e-3)          # 1/t at t~0 would explode
        x_t, masked = corrupt(batch, t, C.MASK, generator)
        return a2d_loss(dlm_forward(model)(x_t), batch, masked, t_safe), {
            "t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_direct(model, prompts=C.PROMPTS, gen_len: int = 32, block_size: int = 8,
                  steps: int = 8, temperature: float = 0.3, top_k: int = 50) -> dict:
    ids, texts, raw_texts = [], [], []
    logits_fn = dlm_forward(model)
    for prompt in prompts:
        prompt_ids = torch.tensor([encode(prompt, bos=True)], device=model.device)
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


def trainable_fingerprint(model) -> torch.Tensor:
    """A cheap signature of the initialisation, for the record.

    Two Direct-DLM runs that claim the same seed must have the same fingerprint;
    if they do not, the comparison is between two different initialisations and
    the loss difference means nothing.
    """
    return torch.cat([p.detach().reshape(-1)[:8] for p in model.parameters()])