#!/usr/bin/env python
"""Cell 9 of 10 -- Qwen3-0.6B, dLLM-A2D on LoRA.

Same base, same LoRA scope and the same batch as cell 10; the only difference
is which implementation of the objective drives it -- dllm's
MDLMTrainer.compute_loss here, the hand-written one there. C2's logic carries
over: the harness is told the Qwen track's mask id (151669) rather than the
project's 27.

    .venv/bin/python example/base/qwen3_0p6b/train_lora_dllm.py
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
from base.hf.a2d_dllm import DLLMLoss
from base.qwen3_0p6b.common_qwen import (BASE, CELL_DLLM, LORA_R, build_a2d_lora,
                                          build_blocks, data_dir_note,
                                          special_ids, trainable_report)
from base.qwen3_0p6b.lora_common import packed, prompts_for, sample_lora_dlm

METHOD = "dLLM-A2D"
CKPT = C.cell_dir("qwen3-0.6b/dllm-a2d-lora-example")


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
    print(f"  tokenizer: vocab {ids['vocab']} mask {ids['mask']} eos {ids['eos']} "
          f"bos {ids['bos']}")
    print(f"  LoRA r={LORA_R}: {trainable_report(model)}")

    blocks = build_blocks(tokenizer, n_docs=args.n_docs, block_size=128)
    data = packed(blocks)
    train_cfg = dataclasses.replace(C.DLM_TRAIN)
    eval_batches = [torch.arange(train_cfg.batch_size)]

    harness = DLLMLoss(special_ids={"mask_token_id": ids["mask"],
                                    "eos_token_id": ids["eos"],
                                    "bos_token_id": ids["bos"] or ids["eos"]})

    def loss_fn(model, batch):
        return harness(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}

    loop = Loop(model, train_cfg, loss_fn, out_dir=C.run_dir(CELL_DLLM))
    print(f"  data: {blocks.shape[0]} blocks x {blocks.shape[1]} tokens "
          f"({data.n_tokens:,} tok, {data_dir_note()})")

    config = {"cell": CELL_DLLM, "base": BASE, "method": METHOD,
              "model_id": "Qwen/Qwen3-0.6B", "train": C.as_dict(train_cfg),
              "mask_id": ids["mask"], "special_ids": ids, "lora_r": LORA_R,
              "lora_alpha": 16, "trainable": trainable_report(model),
              "loss": "dllm MDLMTrainer.compute_loss",
              "attention": "bidirectional (dllm A2DQwen3 + 4D zero mask)"}

    summary = finish_cell(
        loop, data, cell=CELL_DLLM, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL_DLLM), config=config,
        sample_fn=lambda lp: sample_lora_dlm(lp.model, tokenizer,
                                             prompts_for(tokenizer), 32,
                                             mask_id=ids["mask"],
                                             block_size=args.block_size,
                                             steps=args.dlm_steps,
                                             temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"LoRA r{LORA_R} q/k/v/o, {trainable_report(model)['ratio_pct']}% trainable, "
              f"{data.n_tokens:,} tok, dllm loss")

    for row in summary["repetition"]["prompts"]:
        print(f"\n--- rep4 {row['rep4']}\n{row['text'][:300]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
