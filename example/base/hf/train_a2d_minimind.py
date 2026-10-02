#!/usr/bin/env python
"""Cell 3 of 10 -- Qwen-mini-HF, MiniMind-DLM A2D.

Starts from the cell-1 AR checkpoint (weights unchanged), flips attention to
bidirectional, and fine-tunes with the weighted masked-diffusion loss of
notes/03-minimind-dlm.md §4. 200 steps at lr 1e-4.

    .venv/bin/python example/base/hf/a2d_minimind.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.cli import add_common_args, resolve
from common.ckpt import load_into, load_state_dict
from common.loop import Loop
from common.verify import verify_checkpoint
from base.hf.a2d_minimind import BASE, CELL, METHOD, make_a2d_loss, sample_dlm_row
from base.hf.model import build_model

AR_CKPT = C.cell_dir("ar-3.2m-hf")
CKPT = C.cell_dir("ar-3.2m-hf-mini_a2d")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--init-random", action="store_true",
                    help="skip the AR checkpoint (ablation: A2D without pretraining)")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8,
                    help="denoising steps per block (--steps is the training count)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", dest="top_k", type=int, default=50)
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.DLM_TRAIN)

    model = build_model(C.TIER0, seed=train_cfg.seed)
    source = "random init"
    if not args.init_random:
        if not (AR_CKPT / "model.safetensors").exists():
            print(f"FAIL  no AR checkpoint at {AR_CKPT}; run cell 1 first "
                  f"or pass --init-random")
            return 1
        load_into(model, load_state_dict(AR_CKPT))
        source = AR_CKPT.name

    loop = Loop(model, train_cfg, make_a2d_loss(model), out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir), "init_from": source,
              "mask_id": C.MASK, "attention": "bidirectional via 4D zero mask"}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_dlm_row(lp.model, block_size=args.block_size,
                                            steps=args.dlm_steps,
                                            temperature=args.temperature,
                                            top_k=args.top_k),
        eval_batches=eval_batches,
        notes=f"from {source}, A2D lr {train_cfg.lr:.0e}, block {args.block_size}, "
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
        if not row["text"]:
            print(f"    (EOS-truncated to empty; raw: {summary['sample']['raw_texts'][i][:120]!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())