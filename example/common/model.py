"""Build the Qwen3 model config from the frozen Tier-0 shape.

transformers 4.57 only (AGENTS.md: dllm pins it, and 5.x changes the
create_causal_mask signature that dllm's A2D models are written against).
Every field we care about is set explicitly -- defaults differ between
versions and a silently-defaulted field would show up as a phantom A/B delta.
"""
from __future__ import annotations

import torch
from transformers import Qwen3Config

from .config import ModelCfg


def build_qwen3_config(cfg: ModelCfg) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.context_length,
        initializer_range=0.02,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=cfg.tie_word_embeddings,
        rope_theta=cfg.rope_theta,
        rope_scaling=None,
        use_sliding_window=False,        # sliding attention changes the mask; keep it off
        sliding_window=None,
        max_window_layers=0,
        attention_dropout=0.0,
        use_cache=True,
    )


def count_params(model: torch.nn.Module) -> int:
    """Unique parameters. SPEC 3.1: tied embeddings must be counted once."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)