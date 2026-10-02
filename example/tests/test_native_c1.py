#!/usr/bin/env python
"""C1 -- does the native torch port reproduce transformers bit-for-bit?

Structural checks live here; the numeric comparison is delegated to
`example/check_ab.py c1` so there is exactly one implementation of the
tolerance and the layer report. Running it as a subprocess keeps the test
independent of how check_ab imports things.

  1 build   -- does the port have row 1's parameter names, count and tie?
  2 c1      -- max|logit delta| <= 1e-5, via check_ab (random weights, so the
               test needs no checkpoints)
  3 mask    -- does the A2D switch really switch? causal must leave the past
               bit-identical, bidirectional must move it
  4 sample  -- does the no-KV-cache sampler produce valid ids and text?

Run:  .venv/bin/python example/tests/test_native_c1.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example"))

from common import config as C
from base.hf.model import build_model
from base.native.model import build_native_model, sample_native

TOL = 1e-5
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def _batch(n: int = 2, t: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, C.VOCAB, (n, t), generator=g)


def section_build() -> None:
    print("\n[1] build")
    hf = build_model(C.TIER0, seed=1).eval()
    nat = build_native_model(C.TIER0, seed=1).eval()

    hf_keys, nat_keys = set(hf.state_dict()), set(nat.state_dict())
    check("same parameter count",
          sum(p.numel() for p in hf.parameters()) == sum(p.numel() for p in nat.parameters()),
          f"{sum(p.numel() for p in hf.parameters()):,}")
    check("state_dict key sets match", hf_keys == nat_keys,
          f"only hf {sorted(hf_keys - nat_keys)[:3]} / only native {sorted(nat_keys - hf_keys)[:3]}")
    check("tied lm_head", nat.lm_head.weight is nat.model.embed_tokens.weight)
    check("forward returns .logits",
          hasattr(nat(_batch()).logits, "shape") and tuple(nat(_batch()).logits.shape)[-1] == C.VOCAB)


def section_c1() -> None:
    print("\n[2] c1 (delegated to check_ab)")
    proc = subprocess.run([sys.executable, str(ROOT / "example" / "check_ab.py"),
                           "c1", "--random", "--cpu"],
                          capture_output=True, text=True, cwd=str(ROOT))
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("[c1]")]
    print("  " + (tail[-1] if tail else "no output"))
    check("check_ab c1 passes on fresh weights", proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")


def section_mask() -> None:
    print("\n[3] bidirectional switch")
    hf = build_model(C.TIER0, seed=1).eval()
    nat = build_native_model(C.TIER0, seed=1).eval()
    nat.load_state_dict(hf.state_dict(), strict=False)
    batch = _batch()
    with torch.no_grad():
        causal = nat(batch, is_causal=True).logits
        bidir = nat(batch, is_causal=False).logits
    check("causal path matches transformers",
          float((causal - hf(batch).logits).abs().max()) <= TOL,
          f"{float((causal - hf(batch).logits).abs().max()):.3e}")
    check("is_causal=False changes the output",
          float((causal - bidir).abs().max()) > 1e-3,
          f"{float((causal - bidir).abs().max()):.3e}")

    edited = batch.clone()
    edited[:, -1] = (edited[:, -1] + 1) % C.VOCAB
    with torch.no_grad():
        past_causal = float((hf(edited).logits[:, :-1] - hf(batch).logits[:, :-1]).abs().max())
        past_bidir = float((nat(edited, is_causal=False).logits[:, :-1]
                            - bidir[:, :-1]).abs().max())
    check("causal: the past is untouched by the last token", past_causal == 0.0,
          f"{past_causal:.3e}")
    check("bidirectional: the past does see the last token", past_bidir > 1e-3,
          f"{past_bidir:.3e}")

    nat.set_causal(False)
    with torch.no_grad():
        check("set_causal(False) switches the default path",
              float((nat(batch).logits - bidir).abs().max()) == 0.0)
    nat.set_causal(True)


def section_sample() -> None:
    print("\n[4] sample")
    nat = build_native_model(C.TIER0, seed=3)
    out = sample_native(nat, max_new_tokens=16)
    check("ids in vocab", all(0 <= t < C.VOCAB for ids in out["ids"] for t in ids))
    check("length matches max_new_tokens", all(len(i) == 16 for i in out["ids"]))
    check("no KV cache advertised", out["params"]["kv_cache"] is False)
    greedy_a = sample_native(nat, prompts=("Once upon a time",), temperature=0.0,
                             max_new_tokens=16)
    greedy_b = sample_native(nat, prompts=("Once upon a time",), temperature=0.0,
                             max_new_tokens=16)
    check("temperature=0 is deterministic", greedy_a["ids"] == greedy_b["ids"])
    check("text decodes", all(isinstance(t, str) and t for t in out["texts"]))


def main() -> int:
    section_build()
    section_c1()
    section_mask()
    section_sample()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())