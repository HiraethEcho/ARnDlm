#!/usr/bin/env python
"""Cell 1 of 10 -- Qwen-mini-HF, autoregressive baseline.

Trains Tier 0 for 200 steps on tinystories-6400, scores it on frozen batches,
samples both fixed prompts, and writes checkpoints/example/qwen-mini/hf-ar-example/.
This checkpoint is the input to cells 3-5 (dLLM-A2D, MiniMind-A2D, Direct-DLM
from scratch) and the logit reference for cell 2 (the native torch port).

    .venv/bin/python example/base/hf/train_ar.py
    .venv/bin/python example/base/hf/train_ar.py --steps 20 --no-verify   # quick
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
from base.hf.model import BASE, CELL, METHOD, ar_loss, build_model, sample_ar


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.AR_TRAIN)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip reloading the checkpoint and comparing logits")
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.AR_TRAIN)

    model = build_model(C.TIER0, seed=train_cfg.seed)
    loop = Loop(model, train_cfg, ar_loss, out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir)}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=C.cell_dir("qwen-mini/hf-ar-example"),
        run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_ar(lp.model),
        eval_batches=eval_batches,
        notes=f"{data.n_tokens:,} tok, Tier 0")

    if not args.no_verify:
        report = verify_checkpoint(C.cell_dir("qwen-mini/hf-ar-example"),
                                   lambda: build_model(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"reloaded {report.get('tensors')} tensors, "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        if not report["ok"]:
            print(f"  reason  {report.get('reason', '')}")
            return 1

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())