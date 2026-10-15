"""src/models/trans-example.py -- models/model-trans.py 的参考实现。

ModelCfg -> Qwen3Config -> Qwen3ForCausalLM（transformers 4.57.0, SPEC §5 钉版）。
字段全部显式写 —— 默认值随版本变, 静默走默认的字段会变成幽灵 A/B 差。
种子在构造之前打（构造即初始化, 循环里补种已经晚了）。
"""
from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def build_qwen3_config(m) -> Qwen3Config:
    """m 是 config-example 的 MODEL dataclass（鸭子类型: 只要字段在就行）。"""
    return Qwen3Config(
        vocab_size=m.vocab_size,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_hidden_layers=m.num_layers,
        num_attention_heads=m.num_heads,
        num_key_value_heads=m.num_kv_heads,
        head_dim=m.head_dim,
        hidden_act="silu",
        max_position_embeddings=m.context_length,
        initializer_range=0.02,
        rms_norm_eps=m.rms_norm_eps,
        tie_word_embeddings=m.tie_word_embeddings,
        rope_theta=m.rope_theta,
        rope_scaling=None,
        use_sliding_window=False,   # sliding attention 改 mask; 关掉
        sliding_window=None,
        max_window_layers=0,
        attention_dropout=0.0,
        use_cache=True,
    )


def build(m, seed: int = 42) -> Qwen3ForCausalLM:
    # 构造即初始化, 初始化只消耗 torch RNG -> manual_seed 足够;
    # 全局 seed_all(seed) 由 main 在进入前调（numpy/random 与训练循环有关, 与建模型无关）
    if seed is not None:
        torch.manual_seed(seed)
    return Qwen3ForCausalLM(build_qwen3_config(m))


def count_params(model: torch.nn.Module) -> int:
    """唯一参数计数。tied embedding 只算一次（SPEC 3.1）。"""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


@torch.no_grad()
def sample(model, prompt_ids: list[int], max_new_tokens: int = 32,
           temperature: float = 0.3, top_k: int = 50,
           pad_id: int = 0, bos_id: int = 1, eos_id: int = 2) -> dict:
    """AR 采样走 generate。训练中被调用, 退出时恢复 train 态（别留 dropout）。"""
    was_training = model.training
    model.eval()
    prefix = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    generated = out[0, prefix.shape[1]:].tolist()
    if was_training:
        model.train()
    return {"ids": generated}
