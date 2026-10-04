"""models/hf_qwen/config.py -- 把 ModelCfg 换成 transformers 的 Qwen3Config, 字段全部显式写。

版本锁 transformers 4.57.0。验收: 同 cfg 下与实验版 build_qwen3_config 完全一致。
"""

from __future__ import annotations

from transformers import Qwen3Config

from utils.config import ModelCfg


def build_qwen3_config(cfg: ModelCfg) -> Qwen3Config:
    # 每个字段都写出来, 不依赖库的默认值 (默认值随版本变)
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
        use_sliding_window=False,          # 滑动窗口会改 causal mask, 关掉
        sliding_window=None,
        max_window_layers=0,
        attention_dropout=0.0,             # 有 dropout 就没法逐位对拍
        use_cache=True,
    )
