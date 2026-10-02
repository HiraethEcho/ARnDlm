#!/usr/bin/env python
"""Cell 5 of 10 -- Qwen-mini-HF, Direct-DLM (random init, no AR checkpoint).

Identical to cell 3 except that the weights start random, so the pair isolates
one variable: what the AR checkpoint buys. 200 steps at lr 3e-4 -- a from-scratch
model needs the larger step, an A2D fine-tune would be destroyed by it.

    .venv/bin/python example/base/hf/train_direct_dlm.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.cli import add_common_args, resolve
from common.loop import Loop
from common.verify import verify_checkpoint
from base.hf.direct_dlm import (BASE, CELL, METHOD, make_direct_loss, sample_direct,
                                trainable_fingerprint)
from base.hf.model import build_model

CKPT = C.cell_dir("dlm-3.2m-hf")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.AR_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.AR_TRAIN)

    model = build_model(C.TIER0, seed=train_cfg.seed)
    fingerprint = trainable_fingerprint(model)
    print(f"  init fingerprint (first 8 values of each tensor): "
          f"{[round(float(v), 4) for v in fingerprint[:8]]}")

    loop = Loop(model, train_cfg, make_direct_loss(model), out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir), "init_from": "random",
              "mask_id": C.MASK, "attention": "bidirectional via 4D zero mask",
              "init_fingerprint": [round(float(v), 6) for v in fingerprint.tolist()]}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_direct(lp.model, block_size=args.block_size,
                                           steps=args.dlm_steps,
                                           temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"random init, lr {train_cfg.lr:.0e}, block {args.block_size}, "
              f"steps {args.dlm_steps}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(CKPT, lambda: build_model(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    for i, row in enumerate(summary["repetition"]["prompts"]):
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())