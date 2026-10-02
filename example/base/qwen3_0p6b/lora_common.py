"""Row 3 -- shared plumbing for the two LoRA cells.

The Loop expects a PackedData-like object with `.batch(indices)` and
`.block_size`, so the Qwen-tokenizer blocks are wrapped in the same shape. The
objective and the decoder stay in `common/a2d.py`; only the forward differs,
because the base model is a peft-wrapped A2DQwen3.

Mask id 151662 (`<|fim_pad|>`, reused rather than added -- see
common_qwen.py and SPEC.md 3.2).
"""
from __future__ import annotations

import torch

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_dlm, sample_t
from common.data import BlockStream, PackedData
from common.tokenizer import decode


def packed(blocks: torch.Tensor) -> PackedData:
    return PackedData(blocks=blocks, n_docs=blocks.shape[0],
                      n_tokens=blocks.shape[0] * blocks.shape[1],
                      block_size=blocks.shape[1], prepared_dir=torch.device("cpu"))


def forward_fn(model):
    """Bidirectional logits through the peft wrapper.

    A peft model forwards **kwargs to the wrapped model, so attention_mask
    reaches A2DQwen3Model, where a 2D mask means "mask the padding only".
    """
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        mask4d = torch.zeros(input_ids.shape[0], 1, t, t,
                             device=input_ids.device, dtype=torch.float32)
        return model(input_ids=input_ids, attention_mask=mask4d).logits
    return logits_fn


def make_minimind_loss(model, mask_id: int, generator: torch.Generator | None = None,
                       epsilon: float = 0.0):
    def loss_fn(model, batch: torch.Tensor):
        t = sample_t(batch.shape[0], batch.device, generator, epsilon=epsilon)
        t_safe = t.clamp_min(1e-3)
        x_t, masked = corrupt(batch, t, mask_id, generator)
        return a2d_loss(forward_fn(model)(x_t), batch, masked, t_safe), {
            "t_mean": float(t.mean())}
    return loss_fn


@torch.no_grad()
def sample_lora_dlm(model, tokenizer, prompt_ids, gen_len: int, *, mask_id: int,
                    block_size: int = 8, steps: int = 8, temperature: float = 0.3,
                    top_k: int = 50) -> dict:
    ids, texts = [], []
    logits_fn = forward_fn(model)
    for ids_in in prompt_ids:
        x = torch.tensor([ids_in], device=next(model.parameters()).device)
        out = sample_dlm(logits_fn, x, gen_len, mask_id=mask_id, block_size=block_size,
                         steps=steps, temperature=temperature, top_k=top_k)
        generated = out[0, x.shape[1]:]
        if tokenizer.eos_token_id in generated:
            # generated is a tensor: list.index() does not exist on it.
            hit = (generated == tokenizer.eos_token_id).nonzero()
            generated = generated[: int(hit[0])] if hit.numel() else generated
        ids.append(generated.tolist())
        texts.append(tokenizer.decode(generated, skip_special_tokens=True))
    return {"prompts": ["<fixed prompt 1>", "<fixed prompt 2>"], "ids": ids,
            "texts": texts,
            "params": {"gen_len": gen_len, "block_size": block_size, "steps": steps,
                       "temperature": temperature, "top_k": top_k, "kv_cache": False}}


def prompts_for(tokenizer) -> list[list[int]]:
    """SPEC's fixed prompts, encoded with the Qwen tokenizer."""
    return [tokenizer.encode(text, add_special_tokens=False) for text in C.PROMPTS]


def stream_for(blocks: torch.Tensor, batch_size: int, seed: int) -> BlockStream:
    return BlockStream(blocks.shape[0], batch_size, seed)