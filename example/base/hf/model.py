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
from common.sft import encode_prompt
from common.tokenizer import decode, encode
CELL = "ar-3.2m-hf"
BASE = "Qwen-mini-HF"
METHOD = "AR"

def build_model(cfg: ModelCfg, seed: int = 42) -> Qwen3ForCausalLM:
    seed_all(seed)
    return Qwen3ForCausalLM(build_qwen3_config(cfg))

@torch.no_grad()
def sample_ar(model: Qwen3ForCausalLM, prompts=PROMPTS,
              max_new_tokens: int = 32, temperature: float = 0.3,
              top_k: int = 50, prompt_ids: list[list[int]] | None = None) -> dict:
    """Autoregressive sampling through `generate`, greedy-ish by construction.

    top_k=50 with temperature 0.3 is the working range from
    notes/03-minimind-dlm.md §7; the smoke run is not judging quality, it needs
    text that is reproducible enough to compare repetition rates.

    Train mode is restored on the way out: this is called from inside a training
    run, and leaving dropout active would silently change every later step.

    `prompt_ids` takes over from `prompts` when the caller needs a prefix that
    cannot be spelled as one string -- the SFT rows, whose turn separator is EOS
    rather than a newline. See common/sft.py::encode_prompt.
    """
    was_training = model.training
    model.eval()
    ids, texts = [], []
    for prompt in (prompt_ids if prompt_ids is not None else
                   [[*encode(p, bos=True)] for p in prompts]):
        prompt_ids_t = torch.tensor([prompt], device=model.device)
        out = model.generate(
            input_ids=prompt_ids_t,
            attention_mask=torch.ones_like(prompt_ids_t),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=BOS,
            eos_token_id=EOS,
            pad_token_id=0,
        )
        generated = out[0, prompt_ids_t.shape[1]:]
        ids.append(generated.tolist())
        texts.append(decode(generated))
    if was_training:
        model.train()
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
            "params": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                       "top_k": top_k}}

# Prompts for the SFT rows. The questions come from the corpus itself so the
# model's answer is a response to something in-distribution; the two fixed
# pretraining prompts stay the AR rows' business. Sampling uses the same
# temperature/top_k as every other cell so rep-4 stays comparable within a row.
SFT_PROMPTS = ("你是一个 helpful 的助手。用户问：什么是深度学习？",
               "请用一句话解释什么是大语言模型。")

def sample_sft(model: Qwen3ForCausalLM, prompts=SFT_PROMPTS, max_new_tokens: int = 32,
               temperature: float = 0.3, top_k: int = 50) -> dict:
    """`sample_ar` with the exact prefix an SFT block starts with.

    The prefix is built at the token level by `encode_prompt`, not as an
    f-string: the corpus spells a turn as `[user\\n][question][EOS][assistant\\n]`,
    and `f"user\\n{p}\\nassistant\\n"` puts a newline where the corpus has EOS.
    Same token count, different tokens, no error -- and every sampled text and
    rep-4 number would then describe a prefix the model never trained on.

    The header is plain text rather than a `<|im_assistant|>` token because the
    6400-vocab tokenizer has no role tokens and SPEC 3.3 forbids adding any
    (common/sft.py documents this).
    """
    # `prompts` still goes in: sample_ar labels its output with them, and the
    # repetition table in RESULTS.md prints whatever is listed here. Passing only
    # `prompt_ids` would silently label SFT samples with the AR pretraining
    # prompts, since that parameter has its own default.
    return sample_ar(model, prompts=list(prompts),
                     max_new_tokens=max_new_tokens, temperature=temperature,
                     top_k=top_k,
                     prompt_ids=[encode_prompt(p) for p in prompts])
