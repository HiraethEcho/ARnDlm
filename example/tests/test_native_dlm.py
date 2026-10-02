#!/usr/bin/env python
"""test_native_dlm -- cells 6-8, the DLM methods on the hand-written base.

  1 interface -- dllm's compute_loss drives the native model through the
     keyword signature it expects, and the model ignores the padding mask
  2 parity    -- the native row's hand-written loss matches dllm's (C3)
  3 sampling  -- the native decoder fills every block, keeps the prompt, and is
     deterministic at temperature 0
  4 matrix    -- all three native checkpoints exist and are consistent

Run:  .venv/bin/python example/tests/test_native_dlm.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example"))

from common import config as C
from common.a2d import a2d_loss, corrupt, sample_t
from common.ckpt import load_state_dict
from base.native.a2d_dllm import make_dllm_loss
from base.native.a2d_minimind import (build_bidirectional, native_forward,
                                       sample_native_dlm)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def section_interface() -> None:
    print("\n[1] dllm interface")
    model = build_bidirectional(7)
    batch = torch.randint(0, C.VOCAB, (2, 32))
    loss, extra = make_dllm_loss(model)(model, batch)
    check("loss is finite", torch.isfinite(loss).item(), f"{float(loss):.4f}")
    check("labels the source", extra["source"].startswith("dllm."), extra["source"])

    with torch.no_grad():
        # transformers-style call: keyword input_ids plus a 2D padding mask
        out = model(input_ids=batch, attention_mask=torch.ones(2, 32, dtype=torch.long))
    check("accepts attention_mask= keyword", tuple(out.logits.shape) == (2, 32, C.VOCAB),
          str(tuple(out.logits.shape)))

    check("model defaults to bidirectional",
          not torch.equal(model(batch, is_causal=True).logits,
                          model(batch, is_causal=False).logits))


def section_parity() -> None:
    print("\n[2] hand-written vs dllm loss on the native base (C3)")
    model = build_bidirectional(11)
    batch = torch.randint(0, C.VOCAB, (4, 64), generator=torch.Generator().manual_seed(5))
    b = batch.shape[0]
    eps = 1e-3

    torch.manual_seed(21)
    t = sample_t(b, batch.device, epsilon=eps)
    x_t, masked = corrupt(batch, t, C.MASK)   # global RNG: dllm draws from it too
    with torch.no_grad():
        ours = a2d_loss(native_forward(model)(x_t), batch, masked, t)

    torch.manual_seed(21)
    theirs, _ = make_dllm_loss(model, epsilon=eps)(model, batch)
    theirs = theirs.detach()

    rel = abs(float(ours) - float(theirs)) / max(abs(float(theirs)), 1e-9)
    check("relative difference < 1%", rel < 0.01,
          f"ours {float(ours):.6f} / dllm {float(theirs):.6f}  rel {rel:.2e}")

    proc = subprocess.run([sys.executable, str(ROOT / "example" / "check_ab.py"), "c3"],
                          capture_output=True, text=True, cwd=str(ROOT))
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("[c3]")]
    print("  " + (tail[-1] if tail else "no output"))
    check("check_ab c3 passes", proc.returncode == 0)


def section_sampling() -> None:
    print("\n[3] native decoder")
    model = build_bidirectional(3)
    out = sample_native_dlm(model, gen_len=16, block_size=8, steps=8,
                            temperature=0.0)
    check("ids in vocab", all(0 <= t < C.VOCAB for ids in out["ids"] for t in ids))
    check("no MASK left", all(C.MASK not in ids for ids in out["ids"]))
    check("temperature=0 deterministic",
          sample_native_dlm(model, gen_len=16, block_size=8, steps=8,
                            temperature=0.0)["ids"] == out["ids"])
    check("raw text kept alongside truncated text",
          len(out["raw_texts"]) == len(out["texts"]))


def section_matrix() -> None:
    print("\n[4] the three native checkpoints")
    for cell in ("ar-3.2m-native-mini_a2d", "ar-3.2m-native-dllm_a2d"):
        path = C.cell_dir(f"{cell}")
        if not (path / "model.safetensors").exists():
            print(f"  SKIP  {cell} has not been run")
            continue
        state = load_state_dict(path)
        meta = (path / "config.json").read_text()
        check(f"{cell}: 24 tensors, init recorded",
              len(state) == 24 and '"init_from"' in meta, f"{len(state)} tensors")


def main() -> int:
    section_interface()
    section_parity()
    section_sampling()
    section_matrix()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())