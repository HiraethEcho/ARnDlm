#!/usr/bin/env python
"""`ar-3.2m-hf-sft` -- G3 control: the AR baseline, then SFT.

Loads the cell-1 AR checkpoint unchanged and fine-tunes it on the assistant spans
of data/prepared/sft-mini-6400. This is the **control** for G3: without it there
is no way to tell whether a DLM's output got worse or was never good, because
nothing would say what the same AR model does when merely fine-tuned.

Loss is on assistant tokens only (common/losses.py::sft_ar_loss); 200 steps at
the A2D fine-tuning learning rate.

    .venv/bin/python example/base/hf/train_sft.py
    .venv/bin/python example/base/hf/train_sft.py --steps 20 --no-verify  # quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.ckpt import load_into, load_state_dict
from common.cli import add_common_args, resolve_sft
from common.cli import resume as resume_from
from common.loop import Loop
from common.losses import sft_ar_loss
from common.verify import verify_checkpoint
from base.hf.model import BASE, build_model, sample_sft

METHOD = "AR-SFT"
CELL = "ar-3.2m-hf-sft"
AR_CKPT = C.cell_dir("ar-3.2m-hf")
CKPT = C.cell_dir(CELL)


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--init-random", action="store_true",
                    help="skip the AR checkpoint (ablation: SFT from scratch)")
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve_sft(args, C.DLM_TRAIN)

    model = build_model(C.TIER0, seed=train_cfg.seed)
    source = "random init"
    if not args.init_random:
        if not (AR_CKPT / "model.safetensors").exists():
            print(f"FAIL  no AR checkpoint at {AR_CKPT}; run cell 1 first "
                  f"or pass --init-random")
            return 1
        load_into(model, load_state_dict(AR_CKPT))
        source = AR_CKPT.name

    loop = Loop(model, train_cfg, sft_ar_loss, out_dir=C.run_dir(CELL))
    if args.resume:
        resume_from(loop, CKPT)

    config = {
        "cell": CELL, "base": BASE, "method": METHOD,
        "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
        "data": str(data.prepared_dir), "init_from": source,
        "loss": "assistant-only CE", "supervised_tokens_per_block":
            data.labels.ne(-100).sum(dim=1).float().mean().item(),
    }

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_sft(lp.model),
        eval_batches=eval_batches,
        notes=f"from {source}, assistant-only loss, lr {train_cfg.lr:.0e}")

    totals = summary.get("totals", {})
    print(f"\n  supervised tokens  {totals.get('target_tokens', 0):,}  "
          f"skipped steps {totals.get('skipped_steps', 0)}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(CKPT, lambda: build_model(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- {row['prompt']!r}  rep4 {row['rep4']}\n{row['text'][:200]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())