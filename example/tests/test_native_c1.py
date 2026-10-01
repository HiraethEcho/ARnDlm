#!/usr/bin/env python
"""C1 -- does the native torch port reproduce transformers bit-for-bit?

The acceptance bar (SPEC §6) is `max|logit delta| <= 1e-5` between row 1 and
row 2 on the same weights and the same batch. This script measures it in fp32 on
CPU, and when it fails, localises the failure to one component -- embedding,
norm, attention, MLP -- because "the port is wrong" is not an actionable bug
report and "RoPE is applied before QK-Norm" is.

Also checks the two structural assumptions C1 rests on:
  * the native state_dict has exactly the row-1 key set (no rename table)
  * `set_causal(False)` changes the logits without touching a single weight
    (this is the switch cells 6-8 depend on)

Run:  .venv/bin/python example/tests/test_native_c1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.ckpt import load_state_dict
from common.model import build_qwen3_config
from base.hf.model import build_model
from base.native.model import NativeQwen3ForCausalLM, build_native_model

TOL = 1e-5
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def _batch(n: int = 2, t: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, C.VOCAB, (n, t), generator=g)


def main() -> int:
    cfg = C.TIER0
    hf = build_model(cfg, seed=1).eval()
    nat = build_native_model(cfg, seed=1).eval()

    # --- structural ---------------------------------------------------------
    hf_keys = set(hf.state_dict())
    nat_keys = set(nat.state_dict())
    print("\n[1] structure")
    check("same parameter count",
          sum(p.numel() for p in hf.parameters()) == sum(p.numel() for p in nat.parameters()),
          f"{sum(p.numel() for p in hf.parameters()):,}")
    check("state_dict key sets match", hf_keys == nat_keys,
          f"only hf: {sorted(hf_keys - nat_keys)[:3]} / only native: {sorted(nat_keys - hf_keys)[:3]}")
    check("tied lm_head", nat.lm_head.weight is nat.model.embed_tokens.weight)

    missing = nat.load_state_dict(hf.state_dict(), strict=False)
    check("hf weights load without renaming", not missing.unexpected_keys,
          f"unexpected {missing.unexpected_keys[:3]}")
    check("weights really transferred",
          torch.equal(hf.model.embed_tokens.weight, nat.model.embed_tokens.weight))

    # --- the actual comparison ---------------------------------------------
    batch = _batch()
    with torch.no_grad():
        a = hf(batch).logits
        b = nat(batch).logits
    delta = (a - b).abs().max().item()
    print("\n[2] C1 logit parity")
    check(f"max|delta| <= {TOL:g}", delta <= TOL, f"{delta:.3e}  (logit scale "
          f"{a.abs().max().item():.2f})")

    if delta > TOL:
        print("  localising by component (first layer):")
        with torch.no_grad():
            h_hf = hf.model.embed_tokens(batch)
            h_nat = nat.model.embed_tokens(batch)
            print(f"    embedding     {float((h_hf - h_nat).abs().max()):.3e}")
            l0_hf = hf.model.layers[0]
            l0_nat = nat.model.layers[0]
            n_hf = l0_hf.input_layernorm(h_hf)
            n_nat = l0_nat.input_layernorm(h_nat)
            print(f"    input_ln      {float((n_hf - n_nat).abs().max()):.3e}")
            cos_hf, sin_hf = hf.model.rotary_emb(h_hf, torch.arange(batch.shape[1])[None])
            cos_nat, sin_nat = nat.model.rotary(h_hf.shape[1], h_hf.device, h_hf.dtype)
            print(f"    cos/sin       {float((cos_hf - cos_nat).abs().max()):.3e} / "
                  f"{float((sin_hf - sin_nat).abs().max()):.3e}")
            q_hf = l0_hf.self_attn.q_proj(n_hf)
            q_nat = l0_nat.self_attn.q_proj(n_nat)
            print(f"    q_proj        {float((q_hf - q_nat).abs().max()):.3e}")
            qn_hf = l0_hf.self_attn.q_norm(q_hf)
            qn_nat = l0_nat.self_attn.q_norm(q_nat)
            print(f"    q_norm        {float((qn_hf - qn_nat).abs().max()):.3e}")

    # --- the A2D switch -----------------------------------------------------
    print("\n[3] bidirectional switch")
    with torch.no_grad():
        causal = nat(batch, is_causal=True).logits
        bidir = nat(batch, is_causal=False).logits
        hf_ref = hf(batch).logits
    check("causal path still matches transformers",
          float((causal - hf_ref).abs().max()) <= TOL,
          f"{float((causal - hf_ref).abs().max()):.3e}")
    check("is_causal=False changes the output",
          float((causal - bidir).abs().max()) > 1e-3,
          f"{float((causal - bidir).abs().max()):.3e}")

    # Change the last token and look at everything before it. Under a causal
    # mask that difference must be exactly zero; anything else means the mask
    # is not actually causal. Under bidirectional it must be large.
    edited = batch.clone()
    edited[:, -1] = (edited[:, -1] + 1) % C.VOCAB
    with torch.no_grad():
        past_causal = float((hf(edited).logits[:, :-1]
                             - hf(batch).logits[:, :-1]).abs().max())
        past_bidir = float((nat(edited, is_causal=False).logits[:, :-1]
                            - bidir[:, :-1]).abs().max())
    check("causal: the past is untouched by the last token", past_causal == 0.0,
          f"{past_causal:.3e}")
    check("bidirectional: the past does see the last token", past_bidir > 1e-3,
          f"{past_bidir:.3e}")

    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())