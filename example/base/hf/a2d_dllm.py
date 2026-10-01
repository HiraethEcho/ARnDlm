"""Cell 4/7 -- dLLM-A2D on top of an AR checkpoint, through the dllm package.

This cell exists to exercise the *upstream* path, so it deliberately does not
reuse the hand-written corruption/loss from cell 3:

* conversion uses `dllm.pipelines.a2d`'s A2DQwen3 classes, which are a Qwen3
  copy with the causal mask deleted (the patched modeling_qwen3.py:50-95). On
  4.57 there is no `layer.self_attn.is_causal` flag to flip -- the 5.x trick
  used in ~/Projects/dlm does not exist here.
* the loss is `MDLMTrainer.compute_loss`, called for real rather than
  reimplemented. Constructing a full Trainer wants a dataset, a real tokenizer
  and output-dir scaffolding, none of which is on the code path under test, so
  the harness below builds the object with `__new__` and fills in the five
  attributes `compute_loss` reads.
* sampling is `MDLMSampler`.

What A2D changes is not one weight: the config is copied field by field
(`tie_word_embeddings` included) and the state dict loads with no missing or
unexpected keys. Acceptance C5 checks exactly that.
"""
from __future__ import annotations

import torch

from common import config as C
from common.ckpt import load_state_dict, save_model
from common.model import build_qwen3_config
from common.tokenizer import decode, encode

CELL = "hf-dllm-a2d-example"
BASE = "Qwen-mini-HF"
METHOD = "dLLM-A2D"

TIED_ALIASES = ("lm_head.weight",)     # absent from a tied checkpoint by design


def build_a2d_model(cfg=C.TIER0, ar_ckpt=None, seed: int = 42):
    """A2DQwen3LMHeadModel, optionally with the row-1 weights loaded."""
    import dllm
    from dllm.pipelines.a2d import A2DQwen3Config

    from common.seeds import seed_all

    src = build_qwen3_config(cfg)
    for key in ("auto_map", "architectures"):          # as dllm/pipelines/a2d/convert.py
        if hasattr(src, key):
            delattr(src, key)
    target = A2DQwen3Config(**{k: v for k, v in src.to_dict().items()
                               if k != "model_type"})
    seed_all(seed)
    model = dllm.pipelines.a2d.A2DQwen3LMHeadModel(target)

    if ar_ckpt is not None:
        state = load_state_dict(ar_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k not in TIED_ALIASES]
        if missing or unexpected:
            raise RuntimeError(f"A2D conversion lost weights: "
                               f"missing={missing} unexpected={unexpected}")
    return model


class _ProcessingStub:
    """The four attributes MDLMTrainer.compute_loss reads off the tokenizer."""

    padding_side = "right"
    mask_token_id = C.MASK
    bos_token_id = C.BOS
    eos_token_id = C.EOS


class _MeterStub:
    def update(self, **kwargs) -> None:
        pass


class DLLMLoss:
    """dllm's own MDLMTrainer.compute_loss, without constructing a Trainer."""

    def __init__(self, time_epsilon: float = 1e-3, loss_weight_type: str = "scheduler",
                 loss_norm_type: str = "token"):
        from dllm.core.schedulers import LinearAlphaScheduler
        from dllm.core.trainers.mdlm import MDLMTrainer

        self.trainer = MDLMTrainer.__new__(MDLMTrainer)
        self.trainer.scheduler = LinearAlphaScheduler()
        self.trainer.time_epsilon = time_epsilon
        self.trainer.loss_weight_type = loss_weight_type
        self.trainer.loss_norm_type = loss_norm_type
        self.trainer.right_shift_logits = False
        self.trainer.processing_class = _ProcessingStub()
        self.trainer.meter = _MeterStub()

    def __call__(self, model, batch: torch.Tensor) -> torch.Tensor:
        # labels == input_ids with no attention mask: every position is valid,
        # and A2DQwen3Model turns a missing mask into "all valid, bidirectional".
        return self.trainer.compute_loss(model, {"input_ids": batch, "labels": batch})


def make_a2d_loss(loss: DLLMLoss):
    def loss_fn(model, batch: torch.Tensor):
        return loss(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}
    return loss_fn


@torch.no_grad()
def sample_dllm(model, prompts=C.PROMPTS, max_new_tokens: int = 32, block_size: int = 8,
                steps: int = 8, temperature: float = 0.3) -> dict:
    """MDLMSampler over the project's prompts."""
    from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig

    sampler = MDLMSampler(model=model, tokenizer=_ProcessingStub())
    prompt_ids = [torch.tensor(encode(p, bos=True), device=model.device) for p in prompts]
    cfg = MDLMSamplerConfig(max_new_tokens=max_new_tokens, block_size=block_size,
                            steps=steps, temperature=temperature)
    out = sampler.sample(prompt_ids, config=cfg)
    sequences = out.sequences if hasattr(out, "sequences") else out

    ids, texts = [], []
    for i, prompt in enumerate(prompts):
        plen = len(encode(prompt, bos=True))
        gen = sequences[i, plen: plen + max_new_tokens]
        if C.EOS in gen:
            gen = gen[: int((gen == C.EOS).nonzero()[0])]
        ids.append(gen.tolist())
        texts.append(decode(gen))
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
            "params": {"sampler": "dllm.MDLMSampler", "max_new_tokens": max_new_tokens,
                       "block_size": block_size, "steps": steps,
                       "temperature": temperature}}


def save_converted(model, out_dir, config: dict, meta: dict | None = None):
    """Save an A2D model in the same row-agnostic format as every other cell."""
    return save_model(model, out_dir, config, meta)