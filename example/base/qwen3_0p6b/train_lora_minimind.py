#!/usr/bin/env python
"""Cell 10 of 10 -- Qwen3-0.6B, MiniMind-DLM A2D on LoRA.

The 6400-vocab cells taught the mechanism; this cell asks whether it transfers
to a real 600M model without touching its weights. LoRA r=8 on q/k/v/o only:
2,293,760 trainable parameters, 0.384% of the base.

    .venv/bin/python example/base/qwen3_0p6b/train_lora_minimind.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import config as C
from common.cell import finish_cell
from common.cli import add_common_args, resolve
from common.loop import Loop
from base.qwen3_0p6b.common_qwen import (BASE, CELL_MINIMIND, LORA_R,
                                          build_a2d_lora, build_blocks,
                                          data_dir_note, special_ids,
                                          trainable_report)
from base.qwen3_0p6b.lora_common import (make_minimind_loss, packed, prompts_for,
                                         sample_lora_dlm)

METHOD = "MiniMind-DLM-A2D"
CKPT = C.cell_dir("qwen3-0.6b/minimind-a2d-lora-example")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN,
                         default_max_blocks=0)
    ap.add_argument("--n-docs", type=int, default=2000)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    model, tokenizer = build_a2d_lora(seed=C.DLM_TRAIN.seed)
    ids = special_ids(tokenizer)
    print(f"  tokenizer: vocab {ids['vocab']} mask {ids['mask']} eos {ids['eos']}")
    print(f"  LoRA r={LORA_R}: {trainable_report(model)}")

    blocks = build_blocks(tokenizer, n_docs=args.n_docs, block_size=128)
    data = packed(blocks)
    train_cfg = dataclasses.replace(C.DLM_TRAIN, steps=C.DLM_TRAIN.steps,
                                    batch_size=C.DLM_TRAIN.batch_size)
    eval_batches = [torch.arange(train_cfg.batch_size)]

    loop = Loop(model, train_cfg, make_minimind_loss(model, ids["mask"]),
                out_dir=C.run_dir(CELL_MINIMIND))
    print(f"  data: {blocks.shape[0]} blocks x {blocks.shape[1]} tokens "
          f"({data.n_tokens:,} tok, {data_dir_note()})")

    config = {"cell": CELL_MINIMIND, "base": BASE, "method": METHOD,
              "model_id": "Qwen/Qwen3-0.6B", "train": C.as_dict(train_cfg),
              "mask_id": ids["mask"], "special_ids": ids, "lora_r": LORA_R,
              "lora_alpha": 16, "trainable": trainable_report(model),
              "attention": "bidirectional (dllm A2DQwen3 + 4D zero mask)"}

    summary = finish_cell(
        loop, data, cell=CELL_MINIMIND, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL_MINIMIND), config=config,
        sample_fn=lambda lp: sample_lora_dlm(lp.model, tokenizer,
                                             prompts_for(tokenizer), 32,
                                             mask_id=ids["mask"],
                                             block_size=args.block_size,
                                             steps=args.dlm_steps,
                                             temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"LoRA r{LORA_R} q/k/v/o, {trainable_report(model)['ratio_pct']}% trainable, "
              f"{data.n_tokens:,} tok")

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- rep4 {row['rep4']}\n{row['text'][:300]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
