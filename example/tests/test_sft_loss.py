#!/usr/bin/env python
"""test_sft_loss -- does the supervised-only objective actually compute the
supervised-only objective?

`sft_ar_loss` is five lines. The risk is not in the five lines, it is in the
three places where a plausible implementation silently computes something else:

  1 shifting -- targets must come from `labels`, not from `ids`, or the model is
     trained to predict the token it was just given
  2 normalisation -- `reduction='mean'` divides by the number of supervised
     tokens in the *batch*. Blocks have different assistant fractions, so a
     per-sample mean would weight a 10%-supervised block like a 90% one.
  3 the empty batch -- PyTorch returns **nan** when every target is ignored, and
     one nan gradient destroys the run while the loss curve still looks like a
     loss curve for a while.

Sections 2, 3 and 4 check each of those against a hand-computed reference.
Section 6 runs the real thing for 20 steps on CPU.

Run:  .venv/bin/python example/tests/test_sft_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.losses import sft_ar_loss
from common.sft import IGNORE, SFTData, load_sft

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


class FakeLM(torch.nn.Module):
    """Returns fixed logits of shape [1, seq, vocab], so the reference CE is
    computable by hand. Batch 1 keeps the hand-written indices readable."""

    def __init__(self, vocab: int, seq: int):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("logits", torch.randn(1, seq, vocab, generator=g))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        # expand over the batch so the shape matches whatever the caller sends
        return type("Out", (), {"logits": self.logits[:, :inputs.shape[1]]
                                .expand(inputs.shape[0], -1, -1)})()


def make_batch(ids: list[list[int]], labels: list[list[int]]) -> torch.Tensor:
    return torch.stack([torch.tensor(ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long)])


# ------------------------------------------------------------ 1. the shape

def section_shape() -> None:
    print("\n[1] contract")
    L = 8
    ids = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    labels = [[IGNORE] * 3 + [3, 4, 5, 6] + [IGNORE],
              [IGNORE] * 5 + [6, 7] + [IGNORE]]
    batch = make_batch(ids, labels)
    check("batch is [2, B, L]", tuple(batch.shape) == (2, 2, L), str(tuple(batch.shape)))

    model = FakeLM(C.VOCAB, L)
    loss, extra = sft_ar_loss(model, batch)
    check("loss is a scalar tensor", loss.ndim == 0 and torch.is_tensor(loss))
    check("loss is finite", torch.isfinite(loss).all())
    # After the shift, label[i] is scored from logits[i-1], so a label at index 0
    # can never count. Row 1 has 4 supervised labels at 3..6, row 2 has 2 at 5..6.
    check("extra counts supervised targets", extra["target_tokens"] == 6, str(extra))
    check("model was called once", model.calls == 1, str(model.calls))


# ---------------------------------------------------- 2. the shift, by hand

def section_shift() -> None:
    print("\n[2] the loss equals CE over exactly the supervised next-token pairs")
    L = 8
    ids = [[1, 2, 3, 4, 5, 6, 7, 8]]
    labels = [[IGNORE, IGNORE, IGNORE, 3, 4, 5, IGNORE, IGNORE]]
    batch = make_batch(ids, labels)
    model = FakeLM(C.VOCAB, L)

    loss, _ = sft_ar_loss(model, batch)

    # Reference: position t predicts token t+1, so a supervised label at index i
    # is scored from the logits at index i-1. `model.logits[0]` is [seq, vocab];
    # slice the seq axis with `[:L-1]`, not the vocab axis.
    logits = model.logits[0][:L - 1]
    rows, targets = [], []
    for i, lab in enumerate(labels[0]):
        if lab != IGNORE:
            rows.append(i - 1)
            targets.append(lab)
    ref = F.cross_entropy(logits[rows], torch.tensor(targets, dtype=torch.long))
    check("matches a hand-written reference", abs(float(loss) - float(ref)) < 1e-6,
          f"{float(loss):.6f} vs {float(ref):.6f}")

    # The trap: scoring labels[i] against logits[i] (no shift) would also produce
    # a number, and it would be wrong. Assert the shift really happened.
    unshifted = F.cross_entropy(model.logits[0][[r + 1 for r in rows]],
                                torch.tensor(targets))
    check("shifting actually changes the answer",
          abs(float(loss) - float(unshifted)) > 1e-3,
          f"{float(loss):.6f} vs unshifted {float(unshifted):.6f}")

    # And the first position can never be supervised: nothing precedes it.
    check("a label at index 0 cannot be scored (nothing to predict it from)",
          all(rows[i] >= 0 for i in range(len(rows))))


# ------------------------------------------------ 3. per-token normalisation

def section_normalisation() -> None:
    print("\n[3] the denominator is supervised tokens, not samples")
    L = 8
    # Same total supervised count, split very differently across the batch.
    a_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    a_lab = [[IGNORE, IGNORE, 3, 4, 5, 6, IGNORE, IGNORE],
             [IGNORE, IGNORE, 3, 4, 5, 6, IGNORE, IGNORE]]
    b_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    b_lab = [[IGNORE, IGNORE, 3, 4, 5, 6, IGNORE, IGNORE],
             [IGNORE] * 7 + [IGNORE]]

    model = FakeLM(C.VOCAB, L)
    even, _ = sft_ar_loss(model, make_batch(a_ids, a_lab))
    skewed, extra = sft_ar_loss(model, make_batch(b_ids, b_lab))
    check("extra reports the real supervised count", extra["target_tokens"] == 4,
          str(extra))
    check("a half-empty row does not dilute the loss",
          abs(float(even) - float(skewed)) < 1e-6,
          f"{float(even):.6f} vs {float(skewed):.6f}")

    # A per-sample mean would instead give the empty row weight 1/n_samples and
    # halve the loss. Assert it does not.
    check("not a per-sample mean (which would halve it)",
          abs(float(even) - 0.5 * float(skewed)) > 1e-3)

# ------------------------------------------------------- 4. the empty batch

def section_empty() -> None:
    print("\n[4] a batch with nothing supervised must not produce nan")
    L = 8
    ids = [[1, 2, 3, 4, 5, 6, 7, 8]]
    lab = [[IGNORE] * L]
    model = FakeLM(C.VOCAB, L)
    loss, extra = sft_ar_loss(model, make_batch(ids, lab))
    check("loss is 0, not nan", float(loss) == 0.0, f"{float(loss)}")
    check("it is flagged in extra", extra.get("skipped") is True, str(extra))
    check("it is differentiable (backward must not crash)",
          (lambda: (loss.sum().backward(), True)[1])() if loss.requires_grad else True,
          "a detached zero is fine too; the loop must not have NaNed")


# ---------------------------------------------- 5. gradient reaches only the span

def section_gradient() -> None:
    print("\n[5] only the supervised positions contribute gradient")
    L, V = 8, 9
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    labels = torch.tensor([[IGNORE, IGNORE, IGNORE, 3, 4, 5, IGNORE, IGNORE]],
                          dtype=torch.long)
    # A hand-built logit tensor rather than a Linear: the class count has to be
    # 9 (targets 3/4/5), and a 6400x6400 Linear would allocate 164 MB to prove
    # something a 9-wide tensor proves just as well.
    base = torch.zeros(1, L, V)
    base[0, :, 3] = 10.0
    base[0, :, 4] = 10.0
    base[0, :, 5] = 10.0
    logits = base.clone().requires_grad_(True)

    model = torch.nn.Module()
    model.forward = lambda inp: type("O", (), {"logits": logits[:, :inp.shape[1]]})()
    loss, _ = sft_ar_loss(model, torch.stack([ids, labels]))
    loss.backward()
    grad = logits.grad[0]
    # Labels 3/4/5 sit at indices 3/4/5, so they are scored from logit positions
    # 2/3/4. Positions 1 and 5 predict an IGNORE label and must get nothing.
    check("gradient is non-zero at the supervised positions",
          all(float(grad[p, c]) != 0.0 for p, c in ((2, 3), (3, 4), (4, 5))),
          str([round(float(grad[p, c]), 6) for p, c in ((2, 3), (3, 4), (4, 5))]))
    check("gradient is exactly zero where the label is ignored",
          all(float(grad[p, c]) == 0.0 for p in (1, 5) for c in range(V)))
    # Softmax puts gradient on *every* class at a supervised position, so "no
    # gradient elsewhere" is the wrong invariant. The right one is that it sums
    # to zero there -- a leaked target would break that.
    check("per-position gradient sums to zero at supervised positions",
          all(abs(float(grad[p].sum())) < 1e-6 for p in (2, 3, 4)),
          str([round(float(grad[p].sum()), 9) for p in (2, 3, 4)]))


# --------------------------------------------------- 6. on the real corpus

def section_integration() -> None:
    print("\n[6] 20 steps on the real SFT corpus (CPU)")
    prepared = C.ROOT / "data" / "prepared" / "sft-mini-6400"
    if not (prepared / "meta.json").exists():
        print("  SKIP  corpus not built; run example/base/hf/make_sft_data.py")
        return
    data = load_sft(prepared, C.BLOCK_SIZE, max_blocks=64)
    check("corpus loaded", len(data) == 64, str(len(data)))

    from common.data import BlockStream
    from common.loop import Loop
    from common.model import build_qwen3_config
    from common.seeds import seed_all
    from transformers import Qwen3ForCausalLM

    seed_all(0)
    model = Qwen3ForCausalLM(build_qwen3_config(C.TIER0))
    cfg = C.AR_TRAIN.__class__(**{**C.as_dict(C.AR_TRAIN), "steps": 20,
                                  "batch_size": 2, "warmup_steps": 2,
                                  "log_every": 10})
    loop = Loop(model, cfg, sft_ar_loss, device=torch.device("cpu"),
                out_dir=None, amp="float32", require_cuda=False)
    stream = BlockStream(len(data), cfg.batch_size, seed=cfg.seed)
    out = loop.fit(data, stream=stream)
    losses = loop.losses
    check("every loss is finite", all(torch.isfinite(torch.tensor(l)) for l in losses),
          f"first {losses[0]:.4f} last {losses[-1]:.4f}")
    check("loss went down", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("summary records supervised targets as a run total",
          out.get("totals", {}).get("target_tokens", 0) > 0, str(out.get("totals")))
    check("no step was skipped (every batch had supervision)",
          out.get("totals", {}).get("skipped_steps", 1) == 0, str(out.get("totals")))
    # An upper bound, not an identity: at most steps * batch * (L-1) targets
    # exist, and only ~49% of the corpus is supervised (meta.json), so a number
    # near the ceiling would mean the labels lost their IGNORE somewhere.
    ceiling = cfg.steps * cfg.batch_size * (C.BLOCK_SIZE - 1)
    total = out["totals"]["target_tokens"]
    check("supervised total is below the ceiling", 0 < total < ceiling,
          f"{total} < {ceiling}")


def main() -> int:
    section_shape()
    section_shift()
    section_normalisation()
    section_empty()
    section_gradient()
    section_integration()
    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())