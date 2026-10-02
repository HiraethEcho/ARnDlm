"""Qwen-mini-HF: Tier 0 through transformers, plus AR sampling.

Seeding happens here, before the model is constructed, not in the training
loop: `Loop.__init__` seeds too late to influence initialisation, and getting
that wrong makes two supposedly comparable runs start from different weights.
"""
from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from common.config import BOS, EOS, PROMPTS, ModelCfg
from common.model import build_qwen3_config
from common.seeds import seed_all
from common.tokenizer import decode, encode

CELL = "hf-ar-example"
BASE = "Qwen-mini-HF"
METHOD = "AR"


def build_model(cfg: ModelCfg, seed: int = 42) -> Qwen3ForCausalLM:
    seed_all(seed)
    return Qwen3ForCausalLM(build_qwen3_config(cfg))


@torch.no_grad()
def sample_ar(model: Qwen3ForCausalLM, prompts=PROMPTS,
              max_new_tokens: int = 32, temperature: float = 0.3,
              top_k: int = 50) -> dict:
    """Autoregressive sampling through `generate`, greedy-ish by construction.

    top_k=50 with temperature 0.3 is the working range from
    notes/03-minimind-dlm.md §7; the smoke run is not judging quality, it needs
    text that is reproducible enough to compare repetition rates.

    Train mode is restored on the way out: this is called from inside a training
    run, and leaving dropout active would silently change every later step.
    """
    was_training = model.training
    model.eval()
    ids, texts = [], []
    for prompt in prompts:
        prompt_ids = torch.tensor([encode(prompt, bos=True)], device=model.device)
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=BOS,
            eos_token_id=EOS,
            pad_token_id=0,
        )
        generated = out[0, prompt_ids.shape[1]:]
        ids.append(generated.tolist())
        texts.append(decode(generated))
    if was_training:
        model.train()
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
            "params": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                       "top_k": top_k}}