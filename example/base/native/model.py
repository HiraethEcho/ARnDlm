"""Qwen-mini in plain torch -- row 2 of SPEC 2.3, no transformers.

The point of this file is that it must be *the same model* as
`transformers.Qwen3ForCausalLM`: same parameter names, same shapes, same
arithmetic, so that loading a row-1 checkpoint into it is a dict lookup rather
than a rename table. Module and attribute names therefore mirror
`transformers/models/qwen3/modeling_qwen3.py` exactly.

Order of operations inside attention is the part people get wrong, so it is
spelled out in Attention.forward:

    project -> q_norm/k_norm on head_dim -> RoPE -> GQA repeat -> sdpa -> o_proj

QK-Norm comes *before* RoPE (modeling_qwen3.py:203-204), and it is applied on
the head dimension, not the hidden size.

`is_causal=False` is not a diff to a different model -- it is the one switch
A2D flips (notes/03-minimind-dlm.md §8). Everything else stays identical, which
is exactly why the A2D rows can share this file with the AR row.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from common.config import PROMPTS, ModelCfg
from common.tokenizer import decode, encode

CELL = "minimind-ar-example"
BASE = "Qwen-mini-MiniMind"
METHOD = "AR"


@dataclass
class CausalLMOutput:
    """Minimal stand-in for transformers' output object.

    The training loop only ever touches `.logits`; depending on the real
    dataclass would tie the native row back to the package it exists to avoid.
    """

    logits: torch.Tensor


class RMSNorm(nn.Module):
    """modeling_qwen3.py:44 -- compute the variance in fp32, cast back after."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """modeling_qwen3.py:320 -- inv_freq from theta, cos/sin duplicated."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        # Explicit int64 arange then a float cast: matching transformers keeps
        # the frequencies bit-identical, and C1 is a 1e-5 comparison.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                   .float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device).float()
        freqs = pos[:, None] * self.inv_freq[None, :].float()      # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                   # [T, head_dim]
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]   # [1, T, head_dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin are [1, T, head_dim]; unsqueeze(1) lines them up with [B, H, T, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, kv_heads, T, D] -> [B, kv_heads*n_rep, T, D].

    repeat_interleave, not repeat: query head h must see key/value head
    h // n_rep. Using `repeat` would pair the wrong heads and the model would
    still train, just badly -- the kind of bug C1 exists to catch.
    """
    if n_rep == 1:
        return x
    b, kv, t, d = x.shape
    return x[:, :, None].expand(b, kv, n_rep, t, d).reshape(b, kv * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        h, hd, kv = cfg.hidden_size, cfg.head_dim, cfg.num_kv_heads
        self.q_proj = nn.Linear(h, cfg.num_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, kv * hd, bias=False)
        self.v_proj = nn.Linear(h, kv * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos, sin, is_causal: bool) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        # sdpa's default scale is 1/sqrt(head_dim) -- the same value
        # Qwen3Attention passes as `scaling`.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and t > 1)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    """modeling_qwen3.py:70 -- SwiGLU, gate and up in parallel."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, is_causal: bool) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, is_causal)
        h = self.post_attention_layernorm(x)
        return x + self.mlp(h)


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.cfg = cfg

    def forward(self, input_ids: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(input_ids.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, is_causal)
        return self.norm(x)


class NativeQwen3ForCausalLM(nn.Module):
    """Same signature as transformers' version for the paths this project uses."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight   # tied, SPEC 3.1
        self.apply(self._init_weights)
        self._last_is_causal = True       # what A2D flips

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_causal(self, is_causal: bool) -> None:
        """A2D step 1 (notes/03 §8): same weights, bidirectional attention."""
        self._last_is_causal = is_causal

    def forward(self, input_ids: torch.Tensor,
                is_causal: bool | None = None,
                attention_mask: torch.Tensor | None = None) -> CausalLMOutput:
        """Accepts transformers' keyword signature on purpose.

        dllm's MDLMTrainer.compute_loss calls `model(input_ids=...,
        attention_mask=...)`, so this row can be trained by the same upstream
        code as row 1. `attention_mask` is accepted and ignored: the native
        model has no padding to mask, because packing drops the ragged tail
        instead of padding it.
        """
        causal = self._last_is_causal if is_causal is None else is_causal
        return CausalLMOutput(logits=self.lm_head(self.model(input_ids, causal)))


def build_native_model(cfg: ModelCfg, seed: int = 42) -> NativeQwen3ForCausalLM:
    from common.seeds import seed_all

    seed_all(seed)
    return NativeQwen3ForCausalLM(cfg)


@torch.no_grad()
def sample_native(model: NativeQwen3ForCausalLM, prompts=PROMPTS, max_new_tokens: int = 32,
                  temperature: float = 0.3, top_k: int = 50,
                  max_context: int | None = None) -> dict:
    """Autoregressive sampling without `generate`, and without a KV cache.

    Same recipe as row 1 (top_k 50, temperature 0.3) so the two rows' samples are
    comparable. Each step re-runs the whole prefix: at Tier 0 that is cheap, and
    it keeps this row free of transformers' cache machinery.
    """
    was_training = model.training
    model.eval()

    max_context = max_context or model.cfg.context_length
    ids, texts = [], []
    for prompt in prompts:
        prefix = torch.tensor([encode(prompt, bos=True)], device=next(model.parameters()).device)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            window = prefix[:, -max_context:]
            # is_causal=True explicitly: this is AR sampling, and the model
            # object may have been switched to bidirectional for an A2D cell.
            logits = model(window, is_causal=True).logits[0, -1].float()
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = torch.softmax(values / temperature, dim=-1)
                nxt = int(indices[torch.multinomial(probs, 1)])
            generated.append(nxt)
            prefix = torch.cat([prefix, torch.tensor([[nxt]], device=prefix.device)], dim=1)
        ids.append(generated)
        texts.append(decode(generated))
    if was_training:
        model.train()
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
            "params": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                       "top_k": top_k, "kv_cache": False}}
