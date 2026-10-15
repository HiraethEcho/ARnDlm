"""common/a2d-example.py -- a2d/a2d.py 的参考实现。

MiniMind-DLM 风格 A2D 的共享层：corruption / 加权 loss / 块状揭示采样器。
**trans 与 native、mini 与 dllm 四条组合共用这一份** —— 复制一份会让 C2/C3
无法把 loss 差归因到任何东西（copy-instead-of-import 反模式）。

公式（notes/03 §4-§5，与 MDLM 同族，无 ELBO）：
    t ~ U[0,1)                        每 sample 一个标量
    x_t = MASK 概率 t                 逐 token
    L   = Σ 1_masked · CE(x_t→x0) / t ÷ N_valid

两个容易错的点（legacy 踩过，test/12 钉死）：
* 分母是 **valid** token 数，不是 masked 数 —— 否则 loss 随 mask 比例漂移但照样降；
* p_mask 是 per-sample 标量 —— 传 [B,L] 向量会把权重混进 sample 内部，
  闭式对拍失效。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MASK = 27  # 预留槽位（SPEC 3.2），与 tokenizer-example / config-example 同源
EPS = 1e-6  # dllm/core/schedulers/alpha.py 硬编码同值


def sample_t(batch_size: int, device, generator: torch.Generator | None = None,
             epsilon: float = 1e-3) -> torch.Tensor:
    """每 sample 一个 corruption 水平，U[epsilon, 1)。

    epsilon 对齐 dllm 的 MDLMTrainer（t = time_epsilon + (1-time_epsilon)*rand）。
    C2 对拍要求两边同 epsilon 且 RNG 消耗顺序一致 —— generator 显式传。
    """
    t = torch.rand(batch_size, device=device, generator=generator)
    return epsilon + (1.0 - epsilon) * t


def corrupt(x0: torch.Tensor, t: torch.Tensor, mask_id: int = MASK,
            generator: torch.Generator | None = None,
            valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """每个 token 以概率 t[b] 变 MASK。返回 (x_t, masked)。

    valid 限制加噪位置（SFT 的 T4f：user 段是条件不是问题，加噪会教模型重建问题）。
    """
    probs = torch.rand(x0.shape, device=x0.device, generator=generator)
    masked = probs < t[:, None]  # [B, L]，t 是 per-sample 标量
    if valid is not None:
        masked = masked & valid
    return torch.where(masked, torch.full_like(x0, mask_id), x0), masked


def a2d_loss(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor,
             t: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """masked 位置上加权 CE：权重 1/t 平衡各 corruption 水平的梯度贡献。"""
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          x0.reshape(-1), reduction="none").view_as(x0)
    weight = (1.0 / (t + EPS)).unsqueeze(1)  # [B,1] 广播到 L
    per_token = nll * weight * masked * valid
    return per_token.sum() / valid.sum().clamp_min(1)


def gumbel_argmax(logits: torch.Tensor, temperature: float = 1.0,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-max 采样 -> (token, confidence)。温度 0 退化为 argmax。

    始终 fp32：bf16 的 rand 能把 u 舍入到 1.0，-log(-log(1.0)) = +inf。
    confidence 是模型对所提交 token 的真实概率（不是扰动分）——否则揭示顺序
    由噪声驱动而不是模型驱动。
    """
    if temperature <= 0:
        conf, token = F.softmax(logits.float(), dim=-1).max(dim=-1)
        return token, conf
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32,
                   generator=generator).clamp_(1e-20, 1.0)
    gumbel = -torch.log(-torch.log(u))
    perturbed = logits.float() / temperature + gumbel
    token = perturbed.argmax(dim=-1)
    confidence = F.softmax(logits.float(), dim=-1).gather(
        -1, token.unsqueeze(-1)).squeeze(-1)
    return token, confidence


def top_k_filter(logits: torch.Tensor, top_k: int,
                 suppress: int | tuple | None = None) -> torch.Tensor:
    """保留 top-k 候选；suppress 直接禁掉指定 id。

    suppress 必须禁 MASK：MASK 在词表内，采出 MASK 会「揭示」一个仍是 MASK 的
    位置，白烧揭示预算且输出里带 MASK。
    """
    if suppress is not None:
        ids = (suppress,) if isinstance(suppress, int) else tuple(suppress)
        logits = logits.clone()
        logits[..., list(ids)] = -float("inf")
    if not top_k or top_k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


@torch.no_grad()
def sample_dlm(logits_fn, prompt_ids: torch.Tensor, gen_len: int, *,
               mask_id: int = MASK, block_size: int = 8, steps: int = 8,
               temperature: float = 0.3, top_k: int = 50,
               max_context: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """块状置信度解码（notes/03 §5）。

    * 每步重跑**整条**序列 —— 双向 attention 需要全部已知上下文，无 KV cache
    * 每步揭示数 = remaining / remaining_steps，动态算，不是固定 top-k
    * 当前块之外的位置即使更自信也冻结 —— block_size 是质量/速度旋钮
    * 揭示簿记留在 python 整数（remaining 列表），不从 tensor 读回
    * 超出 max_context 直接报错 —— RoPE 位置会跑出训练范围，宁可停也不出垃圾
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    total = prompt_len + gen_len
    if max_context is not None and total > max_context:
        raise ValueError(f"prompt {prompt_len} + gen {gen_len} 超出 max_context {max_context}")
    x = torch.cat([prompt_ids,
                   torch.full((prompt_ids.shape[0], gen_len), mask_id,
                              device=device, dtype=prompt_ids.dtype)], dim=1)

    for start in range(prompt_len, total, block_size):
        end = min(start + block_size, total)
        remaining = [end - start] * x.shape[0]
        steps_left = steps
        while steps_left > 0 and max(remaining) > 0:
            full_logits = logits_fn(x)
            # 从截断分布采样，但置信度打在未截断 logits 上：
            # top-k 重归一化会改变幸存候选间的相对质量，重排位置就失真了
            token, _ = gumbel_argmax(top_k_filter(full_logits, top_k, suppress=mask_id),
                                     temperature, generator)
            confidence = F.softmax(full_logits.float(), dim=-1).gather(
                -1, token.unsqueeze(-1)).squeeze(-1)

            for b in range(x.shape[0]):
                if remaining[b] == 0:
                    continue
                active = x[b, start:end] == mask_id
                n_unmask = max(1, math.ceil(remaining[b] / steps_left))
                conf = torch.where(active, confidence[b, start:end],
                                   torch.full_like(confidence[b, start:end], -float("inf")))
                chosen = conf.topk(min(n_unmask, conf.numel())).indices
                x[b, start + chosen] = token[b, start + chosen]
                remaining[b] -= int(chosen.numel())
            steps_left -= 1
    return x
