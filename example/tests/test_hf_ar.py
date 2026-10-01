#!/usr/bin/env python
"""test_hf_ar -- cell 1 (Qwen-mini-HF, AR) before it is trusted with a checkpoint.

Four questions:

  1 build   -- does Tier 0 come out at the analytic parameter count, and does
               seeding actually reach the initialisation?
  2 loss    -- is the shift-by-one cross entropy finite, differentiable, and
               able to overfit one batch?
  3 ckpt    -- does a saved checkpoint reproduce the same logits in a fresh
               model? (PLAN §6 acceptance item 3)
  4 sample  -- does generate() return ids, and is repetition rate computable?

CPU only and under a minute: this is a correctness gate, not a benchmark.
Real GPU numbers come from example/base/hf/train_ar.py.

Run:  .venv/bin/python example/tests/test_hf_ar.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.ckpt import describe, load_state_dict, save_model
from common.metrics import repetition_rates
from common.model import build_qwen3_config
from base.hf.model import ar_loss, build_model, sample_ar

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def analytic_params(cfg: C.ModelCfg) -> int:
    """Parameter count from the shapes, before anyone builds a model."""
    h, kv, v, i, l = cfg.hidden_size, cfg.num_kv_heads, cfg.vocab_size, \
        cfg.intermediate_size, cfg.num_layers
    hd = cfg.head_dim
    attn = h * h + 2 * (h * kv * hd) + h * h + 2 * hd      # q,k,v,o + q_norm,k_norm
    mlp = 3 * h * i                                        # gate, up, down (SwiGLU)
    norm = 2 * h                                          # two RMSNorms per layer
    return v * h + l * (attn + mlp + norm) + h            # +tied embedding +final norm


def section_build() -> None:
    print("\n[1] build")
    cfg = C.TIER0
    model = build_model(cfg, seed=42)
    info = describe(model)
    want = analytic_params(cfg)
    check("parameter count matches the analytic formula", info["params_unique"] == want,
          f"{info['params_unique']:,} vs {want:,}")
    check("tied embedding detected", info["tied"],
          f"{info['state_dict_entries']} entries -> {info['stored_tensors']} stored")
    check("lm_head shares storage",
          model.lm_head.weight is model.model.embed_tokens.weight)

    again = build_model(cfg, seed=42)
    check("same seed -> same weights",
          torch.equal(model.model.embed_tokens.weight,
                      again.model.embed_tokens.weight))
    other = build_model(cfg, seed=43)
    check("different seed -> different weights",
          not torch.equal(model.model.embed_tokens.weight,
                          other.model.embed_tokens.weight))

    batch = torch.randint(0, C.VOCAB, (2, cfg.context_length))
    with torch.no_grad():
        out = model(batch[:, :-1])
    check("logits shape", tuple(out.logits.shape) == (2, cfg.context_length - 1, C.VOCAB),
          str(tuple(out.logits.shape)))


def section_loss() -> None:
    print("\n[2] loss")
    model = build_model(C.TIER0, seed=1)
    batch = torch.randint(0, C.VOCAB, (4, C.BLOCK_SIZE))
    loss, extra = ar_loss(model, batch)
    check("finite", torch.isfinite(loss).item(), f"loss {loss.item():.4f}")
    check("shift-by-one leaves one target per position",
          extra["target_tokens"] == 4 * (C.BLOCK_SIZE - 1))

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("every parameter receives a gradient",
          len(grads) == len(list(model.parameters())),
          f"{len(grads)} grads")
    check("no NaN in gradients", all(torch.isfinite(g).all() for g in grads))

    # Can it overfit one batch? 30 steps of a 3.2M model on 512 tokens is cheap
    # on CPU and catches a loss that is silently detached from the weights.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = float(loss)
    for _ in range(30):
        opt.zero_grad()
        l, _ = ar_loss(model, batch)
        l.backward()
        opt.step()
    print(f"  note  one batch: {first:.4f} -> {float(l):.4f}")
    check("loss drops when overfitting a single batch", float(l) < first - 0.5)


def section_ckpt() -> None:
    print("\n[3] ckpt")
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = C.TIER0
        model = build_model(cfg, seed=7)
        save_model(model, tmp, {"cell": "test"})

        fresh = Qwen3ForCausalLM(build_qwen3_config(cfg))
        fresh.load_state_dict(load_state_dict(tmp), strict=False)
        batch = torch.randint(0, C.VOCAB, (2, cfg.context_length))
        model.eval()
        fresh.eval()
        with torch.no_grad():
            delta = (model(batch).logits - fresh(batch).logits).abs().max().item()
        check("saved weights reproduce identical logits", delta == 0.0, f"max|d| {delta:.2e}")

        # A fresh model built from the config alone must NOT already match:
        # otherwise the check above would pass for the wrong reason.
        untouched = Qwen3ForCausalLM(build_qwen3_config(cfg))
        with torch.no_grad():
            delta2 = (model(batch).logits - untouched(batch).logits).abs().max().item()
        check("untrained baseline differs", delta2 > 1e-3, f"max|d| {delta2:.2e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def section_sample() -> None:
    print("\n[4] sample")
    model = build_model(C.TIER0, seed=3)
    out = sample_ar(model, max_new_tokens=16)
    check("two prompts sampled", len(out["ids"]) == 2, f"{out['prompts']}")
    check("length matches max_new_tokens",
          all(len(i) == 16 for i in out["ids"]), f"{[len(i) for i in out['ids']]}")
    check("ids in vocab", all(0 <= t < C.VOCAB for ids in out["ids"] for t in ids))
    check("text decodes", all(isinstance(t, str) and t for t in out["texts"]),
          repr(out["texts"][0][:60]))
    rates = repetition_rates(out["ids"], ns=(4,))
    check("rep-4 computable", 0.0 <= rates[4] <= 1.0, f"rep4 {rates[4]:.4f}")

    greedy = sample_ar(model, prompts=("Once upon a time",), temperature=0.0,
                       max_new_tokens=16)
    check("temperature=0 still samples", len(greedy["ids"][0]) == 16)


def main() -> int:
    section_build()
    section_loss()
    section_ckpt()
    section_sample()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())