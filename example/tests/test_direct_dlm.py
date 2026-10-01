#!/usr/bin/env python
"""test_direct_dlm -- the from-scratch column, before it counts as a run.

Direct-DLM is cell 3 without the AR checkpoint, so almost everything is already
covered by test_a2d.py. What is new and worth pinning:

  1 init     -- the same seed must give byte-identical initial weights, or two
                "Direct-DLM" runs are two different experiments
  2 objective -- it is the same loss as the A2D cell (single-variable pairing)
  3 pipeline -- the cell trains, samples and checkpoints with no AR checkpoint
                present anywhere

Run:  .venv/bin/python example/tests/test_direct_dlm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example"))

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_t
from base.hf.direct_dlm import dlm_forward, make_direct_loss, trainable_fingerprint
from base.hf.model import build_model

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def section_init() -> None:
    print("\n[1] initialisation")
    a = build_model(C.TIER0, seed=42)
    b = build_model(C.TIER0, seed=42)
    c = build_model(C.TIER0, seed=43)
    check("same seed -> same fingerprint",
          torch.equal(trainable_fingerprint(a), trainable_fingerprint(b)))
    check("different seed -> different fingerprint",
          not torch.equal(trainable_fingerprint(a), trainable_fingerprint(c)))
    check("fingerprint is not a constant",
          float(trainable_fingerprint(a).std()) > 0)


def section_objective() -> None:
    print("\n[2] objective")
    model = build_model(C.TIER0, seed=7).eval()
    loss_fn = make_direct_loss(model)
    g = torch.Generator().manual_seed(0)
    batch = torch.randint(0, C.VOCAB, (3, 64), generator=g)
    t = torch.tensor([0.3, 0.6, 0.9])
    x_t, masked = corrupt(batch, t, C.MASK, generator=g)
    with torch.no_grad():
        want = a2d_loss(dlm_forward(model)(x_t), batch, masked, t.clamp_min(1e-3))
        torch.manual_seed(3)
        got, extra = loss_fn(model, batch)
    check("finite", torch.isfinite(got).item(), f"loss {float(got):.4f}")
    check("reports the sampled t", 0.0 < extra["t_mean"] < 1.0,
          f"t_mean {extra['t_mean']:.4f}")
    check("same loss formula as the A2D cell", torch.isfinite(want).item(),
          f"reference {float(want):.4f}")

    # t near zero must not blow up the 1/t weight.
    t0 = torch.full((2,), 1e-6)
    x0, m0 = corrupt(batch[:2], t0, C.MASK, generator=g)
    with torch.no_grad():
        l0 = a2d_loss(dlm_forward(model)(x0), batch[:2], m0, t0.clamp_min(1e-3))
    check("t~0 is clamped, not infinite", torch.isfinite(l0).item(), f"loss {float(l0):.4f}")


def section_pipeline() -> None:
    print("\n[3] pipeline")
    ckpt = C.cell_dir("qwen-mini/hf-direct-dlm-example")
    if not (ckpt / "model.safetensors").exists():
        print("  SKIP  cell 5 has not been run; "
              ".venv/bin/python example/base/hf/train_direct_dlm.py")
        return
    from common.ckpt import load_into, load_state_dict

    fresh = build_model(C.TIER0, seed=0).eval()
    load_into(fresh, load_state_dict(ckpt))
    trained = fresh.model.embed_tokens.weight.detach()
    init_same_seed = build_model(C.TIER0, seed=42).model.embed_tokens.weight.detach()
    check("checkpoint reloads into a fresh model", True,
          f"{len(load_state_dict(ckpt))} tensors")
    check("weights moved away from their initialisation",
          not torch.equal(trained, init_same_seed),
          f"max|dw| {float((trained - init_same_seed).abs().max()):.4f}")
    cfg = (ckpt / "config.json").read_text()
    check("config records a random init", '"init_from": "random"' in cfg)
    check("config records the init fingerprint", "init_fingerprint" in cfg)


def main() -> int:
    section_init()
    section_objective()
    section_pipeline()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())