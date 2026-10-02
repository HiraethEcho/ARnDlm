#!/usr/bin/env python
"""test_dllm_a2d -- the dllm path, before it is trusted with a checkpoint.

  1 convert -- does A2DQwen3LMHeadModel accept the row-1 checkpoint with no
     missing, unexpected or altered tensor? (acceptance C5)
  2 logits  -- is the converted model actually bidirectional, and does it agree
     with row 1's 4D-zero-mask path?
  3 loss    -- MDLMTrainer.compute_loss vs the hand-written loss on an
     identical corruption (acceptance C2)
  4 sample  -- does MDLMSampler return ids in vocab with the prompt intact?

Run:  .venv/bin/python example/tests/test_dllm_a2d.py
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
from base.hf.a2d_dllm import (TIED_ALIASES, DLLMLoss, build_a2d_model, sample_dllm)
from base.hf.a2d_minimind import dlm_forward
from base.hf.model import build_model as build_hf

AR_CKPT = C.cell_dir("qwen-mini/hf-ar-example")
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def section_convert() -> None:
    print("\n[1] conversion")
    model = build_a2d_model(C.TIER0)
    check("A2DQwen3 class", type(model).__name__ == "A2DQwen3LMHeadModel",
          type(model).__name__)
    check("model_type is a2d-qwen3", model.config.model_type == "a2d-qwen3",
          model.config.model_type)
    check("tie_word_embeddings survives the config copy",
          model.config.tie_word_embeddings is True)

    if not AR_CKPT.exists():
        print("  SKIP  no AR checkpoint; run cell 1 for the weight checks")
        return
    state = load_state_dict(AR_CKPT)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in TIED_ALIASES]
    check("no missing weights", not real_missing, f"{real_missing[:3]}")
    check("no unexpected weights", not unexpected, f"{unexpected[:3]}")
    live = dict(model.state_dict())
    bad = [k for k, v in state.items() if k in live and not torch.equal(v, live[k])]
    check("every tensor transferred unchanged", not bad, f"{bad[:3]}")


def section_logits() -> None:
    print("\n[2] bidirectional + parity with row 1")
    hf = build_hf(C.TIER0, seed=3).eval()
    model = build_a2d_model(C.TIER0)
    model.load_state_dict(hf.state_dict(), strict=False)
    hf, model = hf.eval(), model.eval()

    g = torch.Generator().manual_seed(11)
    x = torch.randint(0, C.VOCAB, (2, 32), generator=g)
    with torch.no_grad():
        a = hf(x, attention_mask=torch.zeros(2, 1, 32, 32)).logits
        b = model(input_ids=x).logits
    delta = float((a - b).abs().max())
    check("converted model == row 1 with a zero mask", delta <= 1e-5, f"{delta:.3e}")

    edited = x.clone()
    edited[:, -1] = (edited[:, -1] + 1) % C.VOCAB
    with torch.no_grad():
        past = float((model(input_ids=edited).logits[:, :-1] - b[:, :-1]).abs().max())
    check("the past sees the last token", past > 1e-3, f"{past:.3e}")
    with torch.no_grad():
        causal_past = float((hf(edited).logits[:, :-1] - hf(x).logits[:, :-1]).abs().max())
    check("row 1 is still causal", causal_past == 0.0, f"{causal_past:.3e}")


def section_loss() -> None:
    print("\n[3] loss parity (C2)")
    hf = build_hf(C.TIER0, seed=4).eval()
    model = build_a2d_model(C.TIER0)
    model.load_state_dict(hf.state_dict(), strict=False)
    hf, model = hf.eval(), model.eval()

    g = torch.Generator().manual_seed(2)
    batch = torch.randint(0, C.VOCAB, (4, 64), generator=g)
    b = batch.shape[0]
    eps = 1e-3

    torch.manual_seed(9)
    t = sample_t(b, batch.device, epsilon=eps)
    x_t, masked = corrupt(batch, t, C.MASK)
    with torch.no_grad():
        ours = a2d_loss(dlm_forward(hf)(x_t), batch, masked, t)

    torch.manual_seed(9)                      # dllm draws t then the mask, same order
    theirs = DLLMLoss()(model, batch).detach()

    rel = abs(float(ours) - float(theirs)) / max(abs(float(theirs)), 1e-9)
    check("relative loss difference < 1%", rel < 0.01,
          f"ours {float(ours):.6f} / dllm {float(theirs):.6f}  rel {rel:.2e}")

    # The same comparison through the CLI, which also runs C5.
    proc = subprocess.run([sys.executable, str(ROOT / "example" / "check_ab.py"), "all"],
                          capture_output=True, text=True, cwd=str(ROOT))
    for line in proc.stdout.strip().splitlines():
        if line.startswith("[c"):
            print("  " + line)
    check("check_ab all passes", proc.returncode == 0)


def section_sample() -> None:
    print("\n[4] MDLMSampler")
    model = build_a2d_model(C.TIER0)
    out = sample_dllm(model, max_new_tokens=16, block_size=8, steps=8)
    check("two prompts sampled", len(out["ids"]) == 2)
    check("ids in vocab", all(0 <= t < C.VOCAB for ids in out["ids"] for t in ids))
    check("no MASK left", all(C.MASK not in ids for ids in out["ids"]))
    check("text decodes", all(isinstance(t, str) for t in out["texts"]),
          repr(out["texts"][0][:60]))
    check("sampler recorded", out["params"]["sampler"] == "dllm.MDLMSampler")


def main() -> int:
    section_convert()
    section_logits()
    section_loss()
    section_sample()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())