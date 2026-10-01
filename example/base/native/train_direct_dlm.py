#!/usr/bin/env python
"""Cell 8 of 10 -- Qwen-mini-MiniMind, Direct-DLM.

The native counterpart of cell 5: same objective as cell 7 (imported, not
copied), no AR checkpoint, lr 3e-4. Model =
example/base/native/model.py switched to bidirectional.

    .venv/bin/python example/base/native/train_direct_dlm.py
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
from base.hf.direct_dlm import make_direct_loss
from base.native.a2d_minimind import (BASE, DIRECT_CELL, DIRECT_METHOD,
                                       build_bidirectional, native_forward,
                                       sample_native_dlm)

CKPT = C.cell_dir("qwen-mini/native-direct-dlm-example")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.AR_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.AR_TRAIN)

    model = build_bidirectional(seed=train_cfg.seed)

    loss_fn = make_direct_loss(model, logits_fn=native_forward(model))
    loop = Loop(model, train_cfg, loss_fn, out_dir=C.run_dir(DIRECT_CELL))

    config = {"cell": DIRECT_CELL, "base": BASE, "method": DIRECT_METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir), "init_from": "random",
              "mask_id": C.MASK, "attention": "bidirectional (is_causal=False)"}

    summary = finish_cell(
        loop, data, cell=DIRECT_CELL, base=BASE, method=DIRECT_METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(DIRECT_CELL), config=config,
        sample_fn=lambda lp: sample_native_dlm(lp.model, block_size=args.block_size,
                                                steps=args.dlm_steps,
                                                temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"random init, lr {train_cfg.lr:.0e}, block {args.block_size}, "
              f"steps {args.dlm_steps}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(CKPT, lambda: build_bidirectional(0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
