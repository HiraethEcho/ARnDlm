#!/usr/bin/env python
"""Cell 6 of 10 -- Qwen-mini-MiniMind, dLLM-A2D.

dllm's MDLMTrainer.compute_loss driving the native torch model. The upstream
loss is used unmodified; the only accommodation is that the native forward
accepts transformers' keyword signature and ignores the padding mask (there is
no padding -- packing drops the ragged tail).

    .venv/bin/python example/base/native/train_a2d_dllm.py
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
from base.native.a2d_dllm import METHOD, make_dllm_loss
from base.native.model import BASE
from base.native.a2d_minimind import (DLLM_CELL, build_bidirectional,
                                       sample_native_dlm)

AR_CKPT = C.cell_dir("qwen-mini/minimind-ar-example")
CKPT = C.cell_dir("qwen-mini/native-dllm-a2d-example")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    train_cfg, data, eval_batches = resolve(args, C.DLM_TRAIN)

    if not (AR_CKPT / "model.safetensors").exists():
        print(f"FAIL  no AR checkpoint at {AR_CKPT}; run cell 2 first")
        return 1
    model = build_bidirectional(seed=train_cfg.seed)
    load_into(model, load_state_dict(AR_CKPT))

    loop = Loop(model, train_cfg, make_dllm_loss(model), out_dir=C.run_dir(DLLM_CELL))

    config = {"cell": DLLM_CELL, "base": BASE, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "data": str(data.prepared_dir), "init_from": AR_CKPT.name,
              "mask_id": C.MASK,
              "conversion": "native torch model, bidirectional via set_causal(False)",
              "loss": "dllm MDLMTrainer.compute_loss"}

    summary = finish_cell(
        loop, data, cell=DLLM_CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(DLLM_CELL), config=config,
        sample_fn=lambda lp: sample_native_dlm(lp.model, block_size=args.block_size,
                                                steps=args.dlm_steps,
                                                temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"from {AR_CKPT.name}, dllm loss on native torch, block {args.block_size}")

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
