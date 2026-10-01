#!/usr/bin/env python
"""Cell 2 of 10 -- Qwen-mini-MiniMind, the native torch AR baseline.

Same recipe as cell 1 (identical data, batch order, schedule, loss function),
the only difference being that the model is `example/base/native/model.py`
instead of `transformers`. That is what makes the C1 comparison meaningful:
two implementations, one loop.

After training it checks the checkpoint against the row-1 checkpoint --
`max|logit delta|` between the two models on the same weights must be <= 1e-5
(SPEC §6).

    .venv/bin/python example/base/native/train_ar.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import config as C
from common.cell import finish_cell
from common.cli import add_common_args, resolve
from common.ckpt import load_into, load_state_dict
from common.loop import Loop
from common.losses import ar_loss
from common.verify import verify_checkpoint
from base.hf.model import CELL as HF_CELL, build_model as build_hf
from base.native.model import (CELL, METHOD, build_native_model, sample_native)
from base.native.model import BASE

CKPT = C.cell_dir("qwen-mini/minimind-ar-example")
HF_CKPT = C.cell_dir("qwen-mini/hf-ar-example")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.AR_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--compare-hf", action="store_true",
                    help="also compare against the row-1 checkpoint (needs cell 1)")
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.AR_TRAIN)

    loop = Loop(build_native_model(C.TIER0, seed=train_cfg.seed), train_cfg, ar_loss,
                out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir)}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_native(lp.model, C.PROMPTS),
        eval_batches=eval_batches,
        notes=f"{data.n_tokens:,} tok, Tier 0, native torch")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(CKPT, lambda: build_native_model(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    if args.compare_hf:
        if not (HF_CKPT / "model.safetensors").exists():
            print(f"  C1  SKIP  {HF_CKPT} not found -- run cell 1 first")
        else:
            native = build_native_model(C.TIER0, seed=0).to(loop.device).float().eval()
            load_into(native, load_state_dict(HF_CKPT))
            hf = build_hf(C.TIER0, seed=0).to(loop.device).float().eval()
            load_into(hf, load_state_dict(HF_CKPT))
            worst = 0.0
            for idx in eval_batches:
                batch = data.batch(idx).to(loop.device)
                with torch.no_grad():
                    worst = max(worst, float((native(batch).logits
                                              - hf(batch).logits).abs().max()))
            passed = worst <= 1e-5
            print(f"  C1  {'OK' if passed else 'FAIL'}  hf-ar-example vs minimind-ar-example"
                  f"  max|dlogit| {worst:.2e}  (tol 1e-5)")
            ok &= passed

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())