"""Cell 3/7 -- MiniMind-DLM A2D on top of an AR checkpoint.

The A2D conversion has two steps (notes/03 §8) and only one of them touches
this file's logic:

1. causal -> bidirectional attention. For the transformers row that means
   passing a 4D all-zero attention mask: `create_causal_mask` in 4.57 returns a
   prepared 4D mask unchanged (masking_utils.py:768), so a zero mask *is* the
   "no mask" request. The 5.x trick of flipping `layer.self_attn.is_causal`
   does not exist here.
2. vocab += MASK. Not done: MASK is id 27, a reserved buffer token (SPEC 3.2).
   Adding a row would change the embedding shape and silently invalidate every
   weight copied from the AR checkpoint.

No weight changes at all. That is the whole point of A2D.
"""
from __future__ import annotations

import torch

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_dlm, sample_t
from common.tokenizer import decode, encode

CELL = "ar-3.2m-hf-mini_a2d"
BASE = "Qwen-mini-HF"
METHOD = "MiniMind-DLM-A2D"


def dlm_forward(model):
    """Bidirectional logits for a transformers Qwen3, with an explicit mask."""
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        mask4d = torch.zeros(input_ids.shape[0], 1, t, t,
                             device=input_ids.device, dtype=model.dtype)
        return model(input_ids, attention_mask=mask4d).logits
    return logits_fn


def make_a2d_loss(model, generator: torch.Generator | None = None):
    """`loss_fn(model, batch)` shaped for common.loop.Loop.

    Accepts both batch shapes on purpose: [B, L] for a pretraining/A2D cell
    (every position is supervised) and [2, B, L] for an SFT cell (row 0 ids, row
    1 labels with IGNORE outside assistant spans). The same closure serves both
    because a second copy of this function would drift -- which is the failure
    the long-run skill records as "copy instead of import for the second
    caller".
    """
    from common.sft import IGNORE

    def loss_fn(model, batch: torch.Tensor):
        if batch.dim() == 3:
            ids, labels = batch[0], batch[1]
            valid = labels != IGNORE
        else:
            ids, labels, valid = batch, batch, None
        t = sample_t(ids.shape[0], ids.device, generator)
        x_t, masked = corrupt(ids, t, C.MASK, generator, valid=valid)
        # t=0 would divide by ~1e-6; that sample contributes nothing anyway.
        t = t.clamp_min(1e-3)
        logits = dlm_forward(model)(x_t)
        return a2d_loss(logits, ids, masked, t, valid=valid), {"t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_dlm_row(model, prompts=C.PROMPTS, gen_len: int = 32, block_size: int = 8,
                   steps: int = 8, temperature: float = 0.3, top_k: int = 50) -> dict:
    """Block-wise decoding, one sequence at a time (batch 1 keeps the reveal
    budget per sequence unambiguous).

    Both the raw sequence and the EOS-truncated text are kept: the truncated one
    can be empty after a 200-step smoke run, and "empty" then looks like a
    broken decoder when it is really an untrained model.
    """
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
                       "temperature": temperature, "top_k": top_k,
                       "kv_cache": False}}