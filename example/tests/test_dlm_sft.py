#!/usr/bin/env python
"""test_dlm_sft -- does A2D training actually restrict itself to the assistant
spans when the batch carries SFT labels?

Both A2D objectives had to learn a second batch shape ([2, B, L] with labels)
without breaking their existing [B, L] path. Three things can go wrong and all
three produce a finite, plausible loss:

  1 the hand-written `corrupt` masks the user's half too, so the model is
    conditioned on a corrupted prompt instead of being asked to predict the
    answer;
  2 the hand-written loss's `valid` is not threaded through, so its denominator
    counts all 128 tokens while only ~62 are supervised;
  3 dllm's trainer ignores the labels and masks everything -- which is the one
    place where nothing in this repository can fix it, because that code is
    upstream.

Section 4 is the important one: it runs dllm's real `compute_loss` on the same
corruption twice, once with labels and once without, and asserts the label-
restricted number differs. If a future dllm upgrade drops `labels != -100`, that
check fails instead of quietly making the cell mean something else.

Run:  .venv/bin/python example/tests/test_dlm_sft.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_t
from common.sft import IGNORE

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def make_batch(L: int = 12, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(36, 100, (3, L), generator=g)
    labels = torch.full_like(ids, IGNORE)
    for r in range(3):                        # one supervised span per row
        labels[r, 5:9] = ids[r, 5:9]
    return ids, labels


# --------------------------------------------------------- 1. corrupt's valid

def section_corrupt() -> None:
    print("\n[1] corrupt respects `valid`")
    ids, labels = make_batch()
    valid = labels != IGNORE

    g = torch.Generator().manual_seed(1)
    x_t, masked = corrupt(ids, torch.tensor([0.9, 0.9, 0.9]), C.MASK, g)
    check("without valid, masks land anywhere", int(masked.sum()) > int(valid.sum()))

    g = torch.Generator().manual_seed(1)
    x_t2, masked2 = corrupt(ids, torch.tensor([0.9, 0.9, 0.9]), C.MASK, g, valid=valid)
    check("with valid, no mask outside a span", not bool((masked2 & ~valid).any()),
          f"{int((masked2 & ~valid).sum())} outside")
    check("with valid, masks still land inside spans", bool(masked2.any()),
          f"{int(masked2.sum())} of {int(valid.sum())} inside")
    check("the prompt is byte-identical to the input",
          bool((x_t2[~valid] == ids[~valid]).all()))
    check("masked positions carry the MASK id",
          bool((x_t2[masked2] == C.MASK).all()))

    # Same seed must give the same draw, or C2/C3 stop being reproducible.
    g = torch.Generator().manual_seed(1)
    _, again = corrupt(ids, torch.tensor([0.9, 0.9, 0.9]), C.MASK, g, valid=valid)
    check("the draw is reproducible under the same seed", bool(torch.equal(masked2, again)))


# ----------------------------------------------------------- 2. the denominator

def section_denominator() -> None:
    print("\n[2] a2d_loss divides by the supervised tokens, not all of them")
    torch.manual_seed(0)
    logits = torch.randn(3, 12, C.VOCAB)
    ids, labels = make_batch()
    valid = labels != IGNORE
    t = torch.tensor([0.5, 0.5, 0.5])
    _, masked = corrupt(ids, t, C.MASK, torch.Generator().manual_seed(2), valid=valid)

    n_val = int(valid.sum())
    check("the supervised set is a minority of the block", n_val < ids.numel(),
          f"{n_val} of {ids.numel()}")

    a = a2d_loss(logits, ids, masked, t, valid=valid)
    b = a2d_loss(logits, ids, masked, t)
    check("passing valid changes the value", abs(float(a) - float(b)) > 1e-4,
          f"valid {float(a):.6f} vs all {float(b):.6f}")

    # Reference: the mean of the weighted nll over supervised AND masked positions.
    per = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none"
    ).view_as(ids)
    want = float((per * (1.0 / (t + 1e-6)).unsqueeze(1) * masked * valid).sum() / n_val)
    check("matches a hand-written reference", abs(float(a) - want) < 1e-6,
          f"{float(a):.6f} vs {want:.6f}")


# -------------------------------------------- 3. the batch-shape compatibility

def section_batch_shapes() -> None:
    print("\n[3] both objectives accept [B, L] and [2, B, L]")
    from base.hf.a2d_minimind import make_a2d_loss
    from base.hf.a2d_dllm import DLLMLoss, make_a2d_loss as make_dllm_loss
    from base.hf.direct_dlm_dllm import Bidirectional
    from base.hf.model import build_model

    ids, labels = make_batch()
    stacked = torch.stack([ids, labels])
    model = build_model(C.TIER0, seed=0)
    ours = make_a2d_loss(model)

    torch.manual_seed(0)
    v_flat, _ = ours(model, ids)
    check("[B, L] still works on the pretraining path", torch.isfinite(v_flat).all(),
          f"{float(v_flat):.4f}")
    torch.manual_seed(0)
    v_stack, _ = ours(model, stacked)
    check("[2, B, L] works on the SFT path", torch.isfinite(v_stack).all(),
          f"{float(v_stack):.4f}")
    check("the two shapes give different values (valid is threaded through)",
          abs(float(v_flat) - float(v_stack)) > 1e-6,
          f"{float(v_flat):.4f} vs {float(v_stack):.4f}")

    dllm_model = Bidirectional(build_model(C.TIER0, seed=0))
    loss = DLLMLoss()
    torch.manual_seed(0)
    d_flat = float(loss(dllm_model, ids))
    torch.manual_seed(0)
    d_stack = float(loss(dllm_model, stacked))
    check("dllm accepts both shapes and finite-returns",
          torch.isfinite(torch.tensor([d_flat, d_stack])).all(),
          f"flat {d_flat:.4f} stack {d_stack:.4f}")
    check("dllm's two shapes differ, so the labels reach it", abs(d_flat - d_stack) > 1e-6,
          f"{d_flat:.4f} vs {d_stack:.4f}")

    # Both rows, both objectives. The native hand-written factory kept only the
    # [B, L] path for a while and failed at dim 1 with the channel axis read as
    # the batch -- a shape error, and nothing in the hf-only test could see it.
    from base.native.a2d_minimind import make_a2d_loss as native_minimind_loss
    from base.native.a2d_dllm import make_dllm_loss as native_dllm_loss
    from base.native.model import build_native_model

    nat = build_native_model(C.TIER0, seed=0)
    for name, factory in (("native minimind", native_minimind_loss),
                          ("native dllm", native_dllm_loss)):
        fn = factory(nat)
        torch.manual_seed(0)
        a = fn(nat, ids)[0]
        torch.manual_seed(0)
        b = fn(nat, stacked)[0]
        check(f"{name} accepts [B, L] and [2, B, L]", torch.isfinite(a) and torch.isfinite(b),
              f"{float(a):.4f} / {float(b):.4f}")
        check(f"{name}: the two shapes differ (valid threaded through)",
              abs(float(a) - float(b)) > 1e-6, f"{float(a):.4f} vs {float(b):.4f}")


# ------------------------------------------- 4. does dllm honour labels != -100?

def section_dllm_honours_labels() -> None:
    print("\n[4] dllm's trainer really uses labels != -100 (upstream behaviour)")
    import dllm
    from dllm.pipelines.a2d import A2DQwen3Config
    from transformers import AutoConfig

    from base.hf.a2d_dllm import DLLMLoss
    from transformers import Qwen3Config

    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=128, vocab_size=C.VOCAB,
                      max_position_embeddings=64, tie_word_embeddings=True)
    target = A2DQwen3Config(**{k: v for k, v in cfg.to_dict().items()
                               if k != "model_type"})
    small = dllm.pipelines.a2d.A2DQwen3LMHeadModel(target).eval()
    loss = DLLMLoss()

    ids, labels = make_batch(L=12, seed=3)
    torch.manual_seed(11)
    unrestricted = float(loss(small, ids))
    torch.manual_seed(11)
    restricted = float(loss(small, torch.stack([ids, labels])))

    check("both are finite", torch.isfinite(torch.tensor([unrestricted, restricted])).all(),
          f"{unrestricted:.4f} / {restricted:.4f}")
    check("restricting to the spans changes dllm's loss",
          abs(unrestricted - restricted) > 1e-6,
          f"unrestricted {unrestricted:.6f} vs restricted {restricted:.6f}")

    # The decisive version: change nothing but the labels, with the corruption
    # held fixed by reseeding. If dllm ignored labels entirely these would be equal.
    all_ignored = torch.full_like(labels, IGNORE)
    torch.manual_seed(11)
    zero_labels = float(loss(small, torch.stack([ids, all_ignored])))
    check("ignoring every label collapses the loss to ~0 (nothing to predict)",
          abs(zero_labels) < 1e-3, f"{zero_labels:.8f}")


def main() -> int:
    section_corrupt()
    section_denominator()
    section_batch_shapes()
    section_dllm_honours_labels()
    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())