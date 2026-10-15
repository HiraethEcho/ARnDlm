"""src/models/native/model-example.py -- models/model.py 的参考实现。

纯 torch.nn 手写 Qwen-style decoder（RoPE / RMSNorm / SwiGLU / GQA / QK-Norm / tied）。
必须与 transformers.Qwen3ForCausalLM 是同一个模型: 同参数名、同形状、同算术 ——
这样 C1 对拍就是 state_dict 直拷, 不是改写表。模块命名镜像 modeling_qwen3.py。

attention 内部顺序（容易错的就这一处）:
    project -> q_norm/k_norm(head_dim 上) -> RoPE -> GQA repeat -> sdpa -> o_proj
QK-Norm 在 RoPE 之前, 且作用在 head 维（modeling_qwen3.py:203-204）。

is_causal=False 不是换模型 —— 是 A2D 唯一要动的开关（权重一个不动, SPEC 3.1）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalLMOutput:
    """transformers 输出对象的最小替身。训练循环只碰 .logits。"""

    logits: torch.Tensor

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class RMSNorm(nn.Module):
    """variance 在 fp32 里算, 算完转回（modeling_qwen3.py:44）。"""

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
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        # 先 int64 arange 再 .float(): 与 transformers 逐位同频, C1 才有 1e-5 的余地
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                    .float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device).float()
        freqs = pos[:, None] * self.inv_freq[None, :].float()      # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                    # [T, head_dim]
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # [1, T, head_dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    """cos/sin [1, T, head_dim] -> unsqueeze(1) 对齐 [B, H, T, D]。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, kv, T, D] -> [B, kv*n_rep, T, D]。

    必须 repeat_interleave 语义: query 头 h 看的是 key 头 h // n_rep。
    用 torch.repeat 会配错头 —— 模型照样训得动, 只是差, C1 就是为抓它存在的。
    """
    if n_rep == 1:
        return x
    b, kv, t, d = x.shape
    return x[:, :, None].expand(b, kv, n_rep, t, d).reshape(b, kv * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.head_dim = m.head_dim
        self.num_heads = m.num_heads
        self.num_kv_heads = m.num_kv_heads
        self.n_rep = m.num_heads // m.num_kv_heads
        self.q_proj = nn.Linear(m.hidden_size, m.num_heads * m.head_dim, bias=False)
        self.k_proj = nn.Linear(m.hidden_size, m.num_kv_heads * m.head_dim, bias=False)
        self.v_proj = nn.Linear(m.hidden_size, m.num_kv_heads * m.head_dim, bias=False)
        self.o_proj = nn.Linear(m.num_heads * m.head_dim, m.hidden_size, bias=False)
        self.q_norm = RMSNorm(m.head_dim, m.rms_norm_eps)
        self.k_norm = RMSNorm(m.head_dim, m.rms_norm_eps)

    def forward(self, x, cos, sin, is_causal: bool) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        # sdpa 默认 scale = 1/sqrt(head_dim), 与 Qwen3Attention 的 scaling 同值
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and t > 1)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    """SwiGLU: gate 与 up 并列, silu(gate) * up 再 down。"""

    def __init__(self, m):
        super().__init__()
        self.gate_proj = nn.Linear(m.hidden_size, m.intermediate_size, bias=False)
        self.up_proj = nn.Linear(m.hidden_size, m.intermediate_size, bias=False)
        self.down_proj = nn.Linear(m.intermediate_size, m.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.self_attn = Attention(m)
        self.mlp = MLP(m)
        self.input_layernorm = RMSNorm(m.hidden_size, m.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(m.hidden_size, m.rms_norm_eps)

    def forward(self, x, cos, sin, is_causal: bool) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, is_causal)
        h = self.post_attention_layernorm(x)
        return x + self.mlp(h)


class Qwen3Model(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.embed_tokens = nn.Embedding(m.vocab_size, m.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(m) for _ in range(m.num_layers))
        self.norm = RMSNorm(m.hidden_size, m.rms_norm_eps)
        self.rotary = RotaryEmbedding(m.head_dim, m.rope_theta)
        self.cfg = m

    def forward(self, input_ids: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(input_ids.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, is_causal)
        return self.norm(x)


class NativeQwen3ForCausalLM(nn.Module):
    """签名与 transformers 版对齐（dllm 的 MDLMTrainer 只会 model(input_ids=, attention_mask=)）。

    attention_mask 接受并忽略: packing 丢弃残尾而非 padding, 无 padding 可 mask。
    """

    def __init__(self, m):
        super().__init__()
        self.cfg = m
        self.model = Qwen3Model(m)
        self.lm_head = nn.Linear(m.hidden_size, m.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.model.embed_tokens.weight   # tied (SPEC 3.1)
        # 注意顺序: tie 必须在 init 之后 —— 反过来 _init_weights 会把 embed
        # 重画一次（lm_head 与 embed 同存储）, 同 seed 下权重对不上 transformers
        # （transformers: 先 init 两份, tie 时 embed 的那份存活）。
        self._last_is_causal = True

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_causal(self, is_causal: bool) -> None:
        """A2D 开关: 同权重, 双向 attention。"""
        self._last_is_causal = is_causal

    def forward(self, input_ids: torch.Tensor,
                is_causal: bool | None = None,
                attention_mask: torch.Tensor | None = None) -> CausalLMOutput:
        causal = self._last_is_causal if is_causal is None else is_causal
        return CausalLMOutput(logits=self.lm_head(self.model(input_ids, causal)))


def build_native(m, seed: int = 42) -> NativeQwen3ForCausalLM:
    if seed is not None:
        torch.manual_seed(seed)  # 初始化只消耗 torch RNG; 全局 seed_all 由 main 负责
    return NativeQwen3ForCausalLM(m)
