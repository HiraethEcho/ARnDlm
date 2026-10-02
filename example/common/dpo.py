"""DPO: preference pairs, and the objective that needs a reference model.

The data is already downloaded and its shape is ideal -- `data/raw/minimind_dataset/
dpo.jsonl`, 17,166 pairs, each a two-message conversation where chosen and
rejected share the prompt and differ only in the final assistant turn. All 200
sampled pairs have that shape.

Two facts that decide what is runnable here, both measured:

* **Only 10.5% of pairs fit a 128-token block.** The chosen response has a median
  of 577 tokens (prompt 40, chosen 577, rejected 626). Turn-atomic dropping keeps
  ~1,800 of the 17,166 pairs -- plenty for a smoke test, and the reason the
  corpus is sliced rather than exhausted.
* **A DLM has no sequence logprob**, which SPEC's own matrix says ("DLM 无序列
  logprob"). DPO's objective is built on `log pi(y|x)` as a product over
  autoregressive factors; a masked-diffusion model denoises all positions at
  once, so that product is not a quantity it defines. The AR rows can run this
  objective; the DLM rows need an ELBO formulation first. ISSUES.md OPEN-10.

So the loss here is the AR-shaped one, written so the pairing and the assistant
mask are explicit rather than implied by the caller's batch layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .sft import IGNORE


def sequence_logprob(logits: torch.Tensor, ids: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
    """log pi(y | x) summed over the supervised positions. One scalar per row.

    Position i predicts token i+1, so a supervised label at index i is scored
    from the logits at i-1. Summing over *labels* rather than over ids is what
    makes this assistant-only: a DPO model that were allowed to score the prompt
    too could lower the loss by shortening the question.
    """
    logits = logits[:, :-1]
    targets = labels[:, 1:]
    mask = targets != IGNORE
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)


def dpo_loss(policy, reference, ids: torch.Tensor, labels: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """`-log sigmoid(beta * ((lp_c - lp_ref_c) - (lp_r - lp_ref_r)))`.

    The batch must hold pairs: row 2i and row 2i+1 are the chosen and rejected
    completions of the same prompt. An odd length is an error rather than a
    silently dropped row, because a dropped row is a 50/50 sign flip in the
    accuracy and nobody reads the accuracy at Tier 0.

    `beta=0` reduces to log 2 whatever the models say, which is what
    `test_dpo.py` §2 checks: it is the cheapest proof that the whole expression
    is being evaluated rather than a constant.
    """
    if ids.shape[0] % 2:
        raise ValueError(f"DPO needs pairs; got {ids.shape[0]} rows")
    n_sup = int((labels != IGNORE).sum())
    if n_sup == 0:
        zero = policy(input_ids=ids).logits.sum() * 0.0
        return zero, {"accuracy": 0.5, "margin": 0.0, "target_tokens": 0,
                      "skipped": True}

    pol = sequence_logprob(policy(input_ids=ids).logits, ids, labels)
    with torch.no_grad():
        ref = sequence_logprob(reference(input_ids=ids).logits, ids, labels)

    pol_c, pol_r = pol[0::2], pol[1::2]
    ref_c, ref_r = ref[0::2], ref[1::2]
    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    return loss, {"accuracy": float((logits > 0).float().mean()),
                  "margin": float((pol_c - ref_c - pol_r + ref_r).mean()),
                  "target_tokens": n_sup, "skipped": False}


def dpo_blocks(jsonl: Path, block_size: int, max_pairs: int = 0,
               max_blocks: int = 0):
    """`dpo.jsonl` -> chosen and rejected blocks, sharing prompts.

    Layout: rows 2i and 2i+1 of the returned pair stack are the chosen and
    rejected completions of pair i. Supervised positions are the final assistant
    turn only -- the same rule as SFT, and for the same reason.
    """
    from .sft import Turn, _pack
    from .tokenizer import encode
    from . import config as C

    ids_rows: list[list[int]] = []
    labs_rows: list[list[int]] = []
    n_pairs = n_dropped = 0
    with Path(jsonl).open() as fh:
        for line in fh:
            if max_pairs and n_pairs >= max_pairs:
                break
            pair = json.loads(line)
            made = []
            for side in ("chosen", "rejected"):
                turns = [Turn(role=m["role"], content=m["content"])
                         for m in pair[side]]
                blocks = _pack(turns, block_size)
                if not blocks:
                    made = None
                    break
                blk, spans = blocks[0]
                made.append((blk, spans))
            n_pairs += 1
            if made is None:
                n_dropped += 1
                continue
            for blk, spans in made:
                lab = [IGNORE] * len(blk)
                for a, b in spans:
                    lab[a:b] = blk[a:b]
                ids_rows.append(blk + [C.PAD] * (block_size - len(blk)))
                labs_rows.append(lab + [IGNORE] * (block_size - len(blk)))
            if max_blocks and len(ids_rows) >= max_blocks:
                break

    blocks = torch.tensor(ids_rows, dtype=torch.long) if ids_rows else \
        torch.zeros((0, block_size), dtype=torch.long)
    labels = torch.tensor(labs_rows, dtype=torch.long) if labs_rows else \
        torch.zeros((0, block_size), dtype=torch.long)
    return blocks, labels, {"n_pairs": n_pairs, "n_rows": blocks.shape[0],
                            "n_dropped_pairs": n_dropped}
