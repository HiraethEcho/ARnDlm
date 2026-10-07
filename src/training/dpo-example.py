"""src/training/dpo-example.py -- src/training/dpo.py 的参考实现。

DPO 偏好对训练（SPEC §8 rl 列, 只在 AR 行）。数据 = minimind dpo.jsonl
{chosen: [user, assistant], rejected: [...]} —— 同 prompt 异回答（实测
17,166 行全部 roles 结构相同）。

loss = -logsigmoid( β·[(π_c - π_ref_c) - (π_r - π_ref_r)] )，logprob 只在
response 区间累计（与 SFT 的 T4b 同哲学: 不给 prompt 打分）。
ref = 权重的冻结副本（eval 态, 无梯度）。padding 语义: 序列右补齐到定长,
POLICY 与 REF、chosen 与 rejected 用同一个 padding 方式 —— 可比性来自一致,
不来自无 pad。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
BOS, EOS, PAD = 1, 2, 0
IGNORE = -100


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dpo_pairs(path: str | Path, max_pairs: int = 4000,
                   total_len: int = 120, resp_cap: int = 96) -> dict:
    """dpo.jsonl -> {'seq': [N, L] int64, 'resp_mask': [N, L] bool}。

    序列 = [BOS] user\\n{q} [EOS] assistant\\n {resp} [EOS]，右补 PAD 到 total_len。
    prompt 超长（挤掉 response 空间）的整对丢弃并计数。
    """
    tok = _load("common/tokenizer-example.py", "tokenizer_example_dpo").Tok()
    rows = []
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max_pairs:
                break
            d = json.loads(line)
            q = d["chosen"][0]["content"]
            prefix = tok.encode(f"user\n{q}") + [EOS] + tok.encode("assistant\n")
            budget = total_len - 1 - len(prefix) - 1  # BOS + 至少 1 个 response token
            if budget < 8:
                dropped += 1
                continue

            def enc_pair(msgs):
                resp = tok.encode(msgs[1]["content"])[:resp_cap][:budget] + [EOS]
                ids = [BOS] + prefix + resp
                ids = ids[:total_len]
                mask = [False] * len(ids)
                for i in range(len(prefix) + 1, len(ids)):
                    mask[i] = True  # response 区间（含 EOS）
                return ids, mask

            for key in ("chosen", "rejected"):
                ids, mask = enc_pair(d[key])
                if len(ids) < total_len:
                    pad = total_len - len(ids)
                    ids = ids + [PAD] * pad
                    mask = mask + [False] * pad
                rows.append((ids, mask))
    seqs = np.asarray([r[0] for r in rows], dtype=np.int64)
    masks = np.asarray([r[1] for r in rows], dtype=bool)
    return {"seq": seqs, "resp_mask": masks,
            "n_pairs": len(rows) // 2, "dropped": dropped}


class PairBatcher:
    """step -> pair 索引的纯函数（chosen 与 rejected 相邻成对: seq[2i], seq[2i+1]）。"""

    def __init__(self, n_pairs: int, batch_size: int, seed: int):
        self.n, self.B, self.seed = n_pairs, batch_size, seed

    def _perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def position(self, step: int):
        pos = step * self.B
        return pos // self.n, pos % self.n

    def batch(self, step: int) -> np.ndarray:
        epoch, within = self.position(step)
        cur = self._perm(epoch)[within:]
        if len(cur) >= self.B:
            return cur[:self.B]
        return np.concatenate([cur, self._perm(epoch + 1)[:self.B - len(cur)]])


def seq_logp(model, ids: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
    """每个序列在 response 区间的 logprob 和（fp32）。ids [B, L], mask [B, L]。"""
    logits = model(ids).logits[:, :-1].float()
    tgt = ids[:, 1:]
    m = resp_mask[:, 1:]
    lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          tgt.reshape(-1), reduction="none").view_as(tgt)
    return (lp * m).sum(-1)


def pair_batch(data: dict, idx: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一对 = 相邻两行: [2i]=chosen, [2i+1]=rejected -> 张量 [B, L]。"""
    ci = torch.from_numpy(idx * 2).long()
    ri = torch.from_numpy(idx * 2 + 1).long()
    seq = torch.from_numpy(data["seq"]).long()
    mask = torch.from_numpy(data["resp_mask"])
    return (seq[ci].to(device), seq[ri].to(device),
            mask[ci].to(device), mask[ri].to(device))


def dpo_loss(model, ref_model, seq_c, seq_r, mask_c, mask_r, beta: float = 0.1):
    with torch.no_grad():
        ref = seq_logp(ref_model, seq_c, mask_c), seq_logp(ref_model, seq_r, mask_r)
    pol_c, pol_r = seq_logp(model, seq_c, mask_c), seq_logp(model, seq_r, mask_r)
    margin = beta * ((pol_c - ref[0]) - (pol_r - ref[1]))
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()
    return loss, {"loss": loss.item(), "margin": float(margin.mean()), "acc": float(acc)}
