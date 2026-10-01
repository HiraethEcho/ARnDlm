#!/usr/bin/env python
"""Cell 1 of 10 -- Qwen-mini-HF, autoregressive baseline.

Trains Tier 0 for 200 steps on tinystories-6400, scores it on frozen batches,
samples both fixed prompts, and writes checkpoints/example/qwen-mini/hf-ar-example/.
This checkpoint is the input to cells 3-5 (dLLM-A2D, MiniMind-A2D, Direct-DLM
from scratch) and the logit reference for cell 2 (the native torch port).

    .venv/bin/python example/base/hf/train_ar.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.data import BlockStream, build_packed, fixed_batches
from common.loop import Loop
from base.hf.model import BASE, CELL, METHOD, ar_loss, build_model, sample_ar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=C.AR_TRAIN.steps)
    ap.add_argument("--batch-size", type=int, default=C.AR_TRAIN.batch_size)
    ap.add_argument("--max-blocks", type=int, default=4096,
                    help="cap corpus blocks so the smoke run reads a slice, not 24k")
    args = ap.parse_args()

    train_cfg = dataclasses.replace(C.AR_TRAIN, steps=args.steps,
                                    batch_size=args.batch_size)

    data = build_packed(C.DATA_DIR, block_size=C.BLOCK_SIZE, max_blocks=args.max_blocks)
    eval_batches = fixed_batches(4, args.batch_size, len(data))

    model = build_model(C.TIER0, seed=train_cfg.seed)
    loop = Loop(model, train_cfg, ar_loss, out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg)}

    def sample_fn(loop: Loop) -> dict:
        return sample_ar(loop.model, C.TIER0)

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=C.cell_dir("qwen-mini/hf-ar-example"),
        run_dir=C.run_dir(CELL), config=config,
        sample_fn=sample_fn, eval_batches=eval_batches,
        notes=f"{data.n_tokens:,} tok, Tier 0")

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())