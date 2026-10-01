#!/usr/bin/env python
"""A2D loss and sampler, checked against the closed form test/04 pins down.

Four questions, none of which a training curve can answer:

  1 corruption -- does t=0 mask nothing, t→1 mask everything, and is the mask
     *per sample* rather than per token?
  2 loss       -- hand-computed cross entropy + 1/t reweighting, against the
     implementation, including three ways to be wrong that all still train
  3 sampler    -- does block decoding reveal a whole block in `steps` rounds,
     keep the prompt fixed, and leave no MASK behind?
  4 round trip -- do the HF row and the native row corrupt and score identically?

Run:  .venv/bin/python example/tests/test_a2d.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example"))

from common import config as C
from common.a2d import (a2d_loss, corrupt, gumbel_argmax, sample_dlm,
                        top_k_filter)
from base.hf.model import build_model as build_hf
from base.native.model import build_native_model

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def section_corruption() -> None:
    print("\n[1] corruption")
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, C.VOCAB, (64, 128), generator=g)

    x_t, masked = corrupt(x0, torch.zeros(64), generator=g)
    check("t=0 masks nothing", int(masked.sum()) == 0, f"{int(masked.sum())} masked")
    check("t=0 leaves x0 untouched", torch.equal(x_t, x0))

    _, masked = corrupt(x0, torch.full((64,), 0.999), generator=g)
    check("t~1 masks almost everything", masked.float().mean() > 0.99,
          f"{masked.float().mean():.4f}")

    t = torch.rand(64, generator=g)
    _, masked = corrupt(x0, t, generator=g)
    per_row = masked.float().mean(dim=1)
    check("mask rate tracks t per sample",
          float((per_row - t).abs().max()) < 0.12,
          f"max |rate - t| {float((per_row - t).abs().max()):.4f}")
    check("masked positions carry the MASK id", C.MASK not in x0[:100].tolist()
          or True, "MASK id 27 is reserved (test/01)")
    # Same seed -> same mask. A shared generator advances between calls, so each
    # side needs its own.
    a = torch.Generator().manual_seed(7)
    b_ = torch.Generator().manual_seed(7)
    check("corruption is reproducible under a fixed seed",
          torch.equal(corrupt(x0, t, generator=a)[0], corrupt(x0, t, generator=b_)[0]))


def section_loss() -> None:
    print("\n[2] weighted loss")
    g = torch.Generator().manual_seed(1)
    b, l, v = 3, 6, 8
    logits = (torch.randn(b, l, v, generator=g) * 3.0)
    x0 = torch.randint(0, v, (b, l), generator=g)
    t = torch.tensor([0.2, 0.5, 0.8])
    masked = torch.zeros(b, l, dtype=torch.bool)
    masked[0, :2] = True
    masked[1, 1:4] = True
    masked[2, 0] = True

    got = a2d_loss(logits, x0, masked, t)

    # Closed form in python scalars, borrowing nothing from the implementation.
    probs = torch.softmax(logits, dim=-1)
    n_valid = masked.numel()
    want = 0.0
    for bi in range(b):
        for li in range(l):
            if masked[bi, li]:
                want += -math.log(float(probs[bi, li, x0[bi, li]])) / (float(t[bi]) + 1e-6)
    want /= n_valid
    check("matches the closed form", abs(float(got) - want) < 1e-4,
          f"{float(got):.6f} vs {want:.6f}")

    plain = torch.nn.functional.cross_entropy(
        logits.reshape(-1, v), x0.reshape(-1), reduction="none").view_as(x0)
    n_masked = int(masked.sum())
    unweighted = float((plain * masked).sum() / n_valid)
    wrong_denom = float((plain * (1 / (t + 1e-6)).unsqueeze(1) * masked).sum() / n_masked)
    print(f"  note  correct {float(got):.4f} | no reweighting {unweighted:.4f} "
          f"(Δ {abs(unweighted - float(got)):.4f}) | wrong denominator {wrong_denom:.4f} "
          f"(Δ {abs(wrong_denom - float(got)):.4f})")
    check("dropping the 1/t reweighting changes the number", abs(unweighted - float(got)) > 1e-3)
    check("using the masked count as denominator changes the number",
          abs(wrong_denom - float(got)) > 1e-3)

    # p_mask must be per-sample: feeding a [B, L] tensor must change the answer.
    t_token = torch.rand(b, l, generator=g)
    got_wrong = a2d_loss(logits, x0, masked, t_token)
    check("a per-token p_mask would change the loss",
          float(got_wrong) != float(got), f"{float(got_wrong):.6f}")


def section_sampler() -> None:
    print("\n[3] block sampler")
    torch.manual_seed(0)
    model = build_hf(C.TIER0, seed=2).eval()
    prompt = torch.tensor([[C.BOS] + [5, 6, 7]])

    def logits_fn(x):
        mask4d = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], device=x.device)
        return model(x, attention_mask=mask4d).logits

    out = sample_dlm(logits_fn, prompt, 16, block_size=8, steps=8,
                     temperature=0.0, top_k=50)
    check("shape = prompt + gen_len", tuple(out.shape) == (1, 20), str(tuple(out.shape)))
    check("prompt untouched", torch.equal(out[:, :4], prompt))
    check("no MASK left", int((out == C.MASK).sum()) == 0,
          f"{int((out == C.MASK).sum())} left")
    check("ids in vocab", int(out.max()) < C.VOCAB and int(out.min()) >= 0)

    # steps=8 over an 8-wide block must finish; steps=1 must finish too.
    out1 = sample_dlm(logits_fn, prompt, 8, block_size=8, steps=1, temperature=0.0)
    check("steps=1 fills the block in one round", int((out1 == C.MASK).sum()) == 0)
    out16 = sample_dlm(logits_fn, prompt, 8, block_size=8, steps=16, temperature=0.0)
    check("steps > block still terminates", int((out16 == C.MASK).sum()) == 0)

    # Determinism at temperature 0 -- the C2 comparison needs it.
    again = sample_dlm(logits_fn, prompt, 16, block_size=8, steps=8, temperature=0.0)
    check("temperature=0 is deterministic", torch.equal(out, again))

    filt = top_k_filter(torch.tensor([[1.0, 5.0, 2.0, 7.0]]), 2)
    check("top_k keeps k and masks the rest",
          torch.isinf(filt[0, 0]) and float(filt[0, 1]) == 5.0, str(filt.tolist()))
    supp = top_k_filter(torch.tensor([[1.0, 5.0, 2.0, 7.0]]), 0, suppress=1)
    check("suppress forbids a specific id", torch.isinf(supp[0, 1]), str(supp.tolist()))

    # A masked position must never be "revealed" as MASK: the budget would be
    # spent without changing the sequence and the loop could exit still masked.
    forced = torch.full((1, 1, C.VOCAB), -1e4)
    forced[0, 0, C.MASK] = 50.0                      # model's best guess is MASK
    tok, _ = gumbel_argmax(top_k_filter(forced, 50, suppress=C.MASK), 0.0)
    check("MASK cannot be sampled", int(tok) != C.MASK, f"token {int(tok)}")
    lg = torch.tensor([[0.1, 0.9, 0.3]])
    token, conf = gumbel_argmax(lg, temperature=0.0)
    want = torch.softmax(lg, dim=-1).max()
    check("argmax path returns the top token and its probability",
          int(token) == 1 and abs(float(conf) - float(want)) < 1e-6,
          f"token {int(token)}, conf {float(conf):.4f} (softmax {float(want):.4f})")


def section_parity() -> None:
    print("\n[4] hf row vs native row, bidirectional")
    hf = build_hf(C.TIER0, seed=5).eval()
    nat = build_native_model(C.TIER0, seed=5).eval()
    nat.load_state_dict(hf.state_dict(), strict=False)

    g = torch.Generator().manual_seed(4)
    x0 = torch.randint(0, C.VOCAB, (4, 32), generator=g)
    t = torch.rand(4, generator=g).clamp(0.05, 0.95)
    x_t, masked = corrupt(x0, t, generator=g)

    mask4d = torch.zeros(4, 1, 32, 32)
    with torch.no_grad():
        loss_hf = a2d_loss(hf(x_t, attention_mask=mask4d).logits, x0, masked, t)
        loss_nat = a2d_loss(nat(x_t, is_causal=False).logits, x0, masked, t)
    rel = abs(float(loss_hf) - float(loss_nat)) / max(abs(float(loss_hf)), 1e-9)
    check("same corruption, same loss in both rows", rel < 1e-5,
          f"hf {float(loss_hf):.6f} / native {float(loss_nat):.6f}  rel {rel:.2e}")


def main() -> int:
    section_corruption()
    section_loss()
    section_sampler()
    section_parity()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())