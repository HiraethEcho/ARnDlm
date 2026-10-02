#!/usr/bin/env python
"""test_dpo -- does the DPO objective compute the DPO objective?

DPO's loss is small but has three places where a plausible implementation is
wrong, and each one produces a finite, plausible loss:

  1 the sequence logprob must be summed over **assistant tokens only**. Scoring
    the prompt too rewards shortening the question, and the model can drive that
    loss down without becoming a better answerer.
  2 the reference model's logprob is subtracted. Dropping it leaves a
    contrastive loss with no anchor.
  3 the batch must hold *pairs*: row 2i and row 2i+1 are the chosen and rejected
    completions of one prompt. Shuffling them pairs unrelated prompts and still
    gives a number.

Two cheap proofs that the whole expression is being evaluated rather than a
constant: `beta=0` must give exactly log 2 whatever the models say, and a policy
built to favour the chosen completion must reach accuracy 1.0 while the mirror
image reaches 0.0.

Run:  .venv/bin/python example/tests/test_dpo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dpo import IGNORE, dpo_loss, sequence_logprob

FAILS: list[str] = []
LOG2 = 0.6931471805599453


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


class TableLM(torch.nn.Module):
    """Returns a caller-supplied logit table.

    A *uniform additive shift* cannot stand in for a reference difference:
    log_softmax is shift-invariant, so the two models come out identical and every
    DPO term collapses to zero. That is what the first version of this test did,
    and it made three checks pass vacuously at exactly log 2.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, inputs=None, input_ids=None):
        ids = inputs if inputs is not None else input_ids
        return type("O", (), {"logits": self.table[:, :ids.shape[1]]
                              .expand(ids.shape[0], -1, -1)})()


def random_lm(B: int, L: int, V: int, seed: int) -> TableLM:
    g = torch.Generator().manual_seed(seed)
    return TableLM(torch.randn(1, L, V, generator=g))


def favouring_lm(ids: torch.Tensor, labels: torch.Tensor, B: int, L: int,
                 V: int, strength: float = 8.0) -> TableLM:
    """Puts `strength` on the target token wherever its row is supervised.

    The table is [B, L, V], **not** [1, L, V]. With a shared batch slot the rows
    overwrite each other and only the last one survives, so "a policy that
    favours chosen" and "a policy that favours rejected" came out nearly
    identical and both reported accuracy 1.0.
    """
    table = torch.zeros(B, L, V)
    for r in range(B):
        for i in range(1, L):
            if labels[r, i] != IGNORE:
                table[r, i - 1, int(ids[r, i])] = strength
    return TableLM(table)


def make_batch(B: int = 3, L: int = 10, V: int = 32):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(1, V, (B, L), generator=g)
    labels = torch.full_like(ids, IGNORE)
    labels[:, 4:] = ids[:, 4:]                  # only the tail is supervised
    return ids, labels


def pair_batch(ids, labels):
    """Rows 2i/2i+1 are chosen/rejected for one prompt."""
    B = ids.shape[0] // 2
    pi = torch.stack([ids[2 * i] for i in range(B)] + [ids[2 * i + 1] for i in range(B)])
    pl = torch.stack([labels[2 * i] for i in range(B)] + [labels[2 * i + 1] for i in range(B)])
    return pi, pl


# ------------------------------------------------------- 1. the logprob itself

def section_logprob() -> None:
    print("\n[1] sequence_logprob sums assistant tokens and nothing else")
    ids, labels = make_batch()
    model = random_lm(3, ids.shape[1], 32, seed=0)
    lp = sequence_logprob(model(ids).logits, ids, labels)
    check("one scalar per row", tuple(lp.shape) == (3,), str(tuple(lp.shape)))
    check("finite", bool(torch.isfinite(lp).all()),
          str([round(float(x), 4) for x in lp]))

    logits = model.table[0]
    rows = [i for i in range(ids.shape[1]) if labels[0, i] != IGNORE]
    ref = sum(F.log_softmax(logits[i - 1], -1)[int(ids[0, i])] for i in rows)
    check("matches a hand-written reference", abs(float(lp[0]) - float(ref)) < 1e-5,
          f"{float(lp[0]):.6f} vs {float(ref):.6f}")

    prompt_only = sum(F.log_softmax(logits[i - 1], -1)[int(ids[0, i])]
                      for i in range(1, 4))
    check("it is not the prompt's logprob instead",
          abs(float(lp[0]) - float(prompt_only)) > 1e-3,
          f"{float(lp[0]):.6f} vs prompt {float(prompt_only):.6f}")
    check("the prompt is a minority of the block",
          len(rows) < ids.shape[1] - 1, f"{len(rows)} of {ids.shape[1] - 1}")


# ------------------------------------------------------------- 2. the loss

def section_loss() -> None:
    print("\n[2] dpo_loss")
    ids, labels = make_batch(B=4)
    B, L, V = 4, ids.shape[1], 32
    pi, pl = pair_batch(ids, labels)

    pol, ref = random_lm(B, L, V, seed=1), random_lm(B, L, V, seed=2)
    loss, extra = dpo_loss(pol, ref, pi, pl, beta=0.1)
    check("finite scalar", loss.ndim == 0 and torch.isfinite(loss), str(float(loss)))
    check("the loss is not the degenerate log 2", abs(float(loss) - LOG2) > 1e-4,
          f"{float(loss):.6f}")
    check("reports accuracy in [0, 1]", 0.0 <= extra["accuracy"] <= 1.0,
          f"acc {extra['accuracy']:.3f}")
    check("reports the margin", "margin" in extra, str(sorted(extra)))

    flat, _ = dpo_loss(pol, ref, pi, pl, beta=0.0)
    # 1e-6, not 1e-9: the whole path runs in float32 (see sequence_logprob's
    # `.float()`), so log 2 is only good to about eight digits.
    check("beta=0 gives log 2", abs(float(flat) - LOG2) < 1e-6,
          f"{float(flat):.10f} vs {LOG2:.10f}")

    good_policy = favouring_lm(pi, pl, B, L, V)
    good, good_extra = dpo_loss(good_policy, ref, pi, pl, beta=0.1)
    check("a policy that favours `chosen` scores better", float(good) < float(loss),
          f"good {float(good):.6f} vs {float(loss):.6f}")
    check("and reaches accuracy 1.0", good_extra["accuracy"] == 1.0,
          f"acc {good_extra['accuracy']}")

    # The mirror image: put the same policy's rows in the opposite order, so
    # `chosen` and `rejected` swap roles. The margin must be exactly negated and
    # the accuracy exactly mirrored. This is a stronger statement than building a
    # hand-made anti-policy, and it does not depend on a helper getting the
    # semantics right.
    swapped_ids = torch.stack([pi[1], pi[0], pi[3], pi[2]])
    bad, bad_extra = dpo_loss(good_policy, ref, swapped_ids,
                              torch.stack([pl[1], pl[0], pl[3], pl[2]]), beta=0.1)
    check("swapping chosen/rejected negates the margin",
          abs(bad_extra["margin"] + good_extra["margin"]) < 1e-4,
          f"{good_extra['margin']:.6f} vs {bad_extra['margin']:.6f}")
    check("and mirrors the accuracy", bad_extra["accuracy"] == 0.0,
          f"acc {bad_extra['accuracy']}")
    check("the reversed pair scores worse than the random policy",
          float(bad) > float(loss), f"reversed {float(bad):.6f} vs {float(loss):.6f}")

    moved, _ = dpo_loss(pol, random_lm(B, L, V, seed=99), pi, pl, beta=0.1)
    check("the reference term is actually subtracted", abs(float(moved) - float(loss)) > 1e-6,
          f"{float(moved):.6f} vs {float(loss):.6f}")

    bigger, _ = dpo_loss(pol, ref, pi, pl, beta=1.0)
    check("beta scales the margin", abs(float(bigger) - float(loss)) > 1e-6,
          f"beta=1 {float(bigger):.6f} vs beta=0.1 {float(loss):.6f}")


# ------------------------------------------------------- 3. the pair layout

def section_pairing() -> None:
    print("\n[3] the batch really is pairs")
    ids, labels = make_batch(B=4)
    B, L, V = 4, ids.shape[1], 32
    pol, ref = random_lm(B, L, V, seed=3), random_lm(B, L, V, seed=4)
    straight, _ = dpo_loss(pol, ref, ids, labels, beta=0.1)

    swapped = torch.stack([ids[0], ids[1], ids[1], ids[0]])
    swapped_lab = torch.stack([labels[0], labels[1], labels[0], labels[1]])
    crossed, _ = dpo_loss(pol, ref, swapped, swapped_lab, beta=0.1)
    check("swapping chosen/rejected changes the answer",
          abs(float(straight) - float(crossed)) > 1e-6,
          f"{float(straight):.6f} vs {float(crossed):.6f}")

    try:
        dpo_loss(pol, ref, ids[:3], labels[:3], beta=0.1)
        check("an odd batch is rejected", False, "it returned a number")
    except ValueError as exc:
        check("an odd batch is rejected", True, str(exc)[:60])


# ----------------------------------------------------------- 4. degenerate

def section_degenerate() -> None:
    print("\n[4] an all-masked completion")
    ids, labels = make_batch(B=2)
    dead = torch.full_like(labels, IGNORE)
    L, V = labels.shape[1], 32
    loss, extra = dpo_loss(random_lm(2, L, V, seed=5), random_lm(2, L, V, seed=6),
                           ids, dead, beta=0.1)
    check("no supervised token -> finite, not nan", torch.isfinite(loss),
          f"{float(loss):.6f}")
    check("and it is flagged", extra.get("skipped") is True, str(extra))


# ------------------------------------------------------ 5. the real corpus

def section_corpus() -> None:
    print("\n[5] the real dpo.jsonl packs into pairs")
    src = C.ROOT / "data" / "raw" / "minimind_dataset" / "dpo.jsonl"
    if not src.exists():
        print("  SKIP  dpo.jsonl not present")
        return
    from common.dpo import dpo_blocks

    blocks, labels, stats = dpo_blocks(src, C.BLOCK_SIZE, max_pairs=400)
    check("an even number of rows", blocks.shape[0] % 2 == 0, str(stats))
    check("some pairs survived", stats["n_rows"] > 0, str(stats))
    check("supervision is a minority of each block",
          bool(((labels != IGNORE).sum(dim=1)
                < C.BLOCK_SIZE - 1).all()),
          str(int((labels != IGNORE).sum(dim=1).max())))
    check("padding is never supervised",
          int(((blocks == 0) & (labels != IGNORE)).sum()) == 0)


from common import config as C  # noqa: E402  (after sys.path setup)


def main() -> int:
    section_logprob()
    section_loss()
    section_pairing()
    section_degenerate()
    section_corpus()
    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())