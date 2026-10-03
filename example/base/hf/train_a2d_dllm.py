#!/usr/bin/env python
"""Cell 4 of 10 -- Qwen-mini-HF, dLLM-A2D through the dllm package.

Same starting weights as cell 3 (the cell-1 AR checkpoint), but the conversion
uses dllm's A2DQwen3 classes, the loss is MDLMTrainer.compute_loss, and
sampling is MDLMSampler. Nothing from cell 3 is imported on purpose -- the
point of this cell is to test the upstream path, not to re-test our own.

    .venv/bin/python example/base/hf/train_a2d_dllm.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.cli import add_common_args, resolve, dlm_steps
from common.loop import Loop
from common.verify import verify_checkpoint
from base.hf.a2d_dllm import (BASE, CELL, METHOD, DLLMLoss, build_a2d_model,
                              make_a2d_loss, sample_dllm)

AR_CKPT = C.cell_dir("ar-3.2m-hf")
CKPT = C.cell_dir("ar-3.2m-hf-dllm_a2d")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.DLM_TRAIN)

    if not (AR_CKPT / "model.safetensors").exists():
        print(f"FAIL  no AR checkpoint at {AR_CKPT}; run cell 1 first")
        return 1
    model = build_a2d_model(C.TIER0, ar_ckpt=AR_CKPT, seed=train_cfg.seed)

    loop = Loop(model, train_cfg, make_a2d_loss(DLLMLoss()), out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir), "init_from": AR_CKPT.name,
              "mask_id": C.MASK, "conversion": "dllm.pipelines.a2d.A2DQwen3LMHeadModel"}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_dllm(lp.model, block_size=args.block_size,
                                         steps=dlm_steps(args, args.dlm_steps),
                                         temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"from {AR_CKPT.name}, dllm MDLMTrainer, block {args.block_size}, "
              f"steps {args.dlm_steps}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(CKPT, lambda: build_a2d_model(C.TIER0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    for i, row in enumerate(summary["repetition"]["prompts"]):
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())