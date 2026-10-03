#!/usr/bin/env python
"""`qwen3-0.6b-mini_a2d-sft` -- G4's SFT arm on the real 600M model.

Cells 9 and 10 showed that A2D transfers to Qwen3-0.6B under LoRA. This cell asks
whether the model can then *chat*: it takes the `qwen3-0.6b-mini_a2d` checkpoint
and fine-tunes it on the assistant spans of the SFT corpus, with the backbone
still frozen and LoRA still the only thing that moves.

Two things are specific to this row and both are traps:

* **The tokenizer track.** Row 3 speaks Qwen's 151,669-vocab BPE, so the SFT
  corpus is re-tokenised here rather than read from `prepared/sft-mini-6400`
  (SPEC 3.3). `common.sft` takes the encoder, the block opener, the separator and
  the pad id as parameters for exactly this reason: on Qwen, id 1 is
  `<|im_start|>` and id 2 is an ordinary BPE token, so the 6400 track's ids would
  be meaningful text in meaningless positions.
* **The assistant-only loss.** `lora_common.make_minimind_loss` builds its batch
  the same way the 6400 rows do, so an SFT batch (labels carrying IGNORE outside
  the spans) reaches it unchanged and the A2D restriction lands inside the
  assistant half.

    .venv/bin/python example/base/qwen3_0p6b/train_lora_sft.py
    .venv/bin/python example/base/qwen3_0p6b/train_lora_sft.py --n-docs 400 --steps 20
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
from common.ckpt import load_into, load_state_dict
from common.cli import add_common_args
from common.loop import Loop
from base.qwen3_0p6b.common_qwen import (BASE, LORA_R, build_a2d_lora,
                                          build_sft_data, special_ids,
                                          trainable_report)
from base.qwen3_0p6b.lora_common import (make_minimind_loss, sample_lora_dlm,
                                         sft_prompts_for)
from base.qwen3_0p6b.common_qwen import CELL_MINIMIND

METHOD = "MiniMind-DLM-A2D-SFT"
CELL = "qwen3-0.6b-mini_a2d-sft"
UPSTREAM = C.cell_dir("qwen3-0.6b-mini_a2d")
CKPT = C.cell_dir(CELL)
SFT_PROMPTS = ("你是一个 helpful 的助手。用户问：什么是深度学习？",
               "请用一句话解释什么是大语言模型。")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN,
                         default_max_blocks=0)
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--init-from", type=Path, default=UPSTREAM)
    ap.add_argument("--init-random", action="store_true",
                    help="skip the A2D checkpoint (ablation: SFT straight off Qwen3)")
    args = ap.parse_args()

    model, tokenizer = build_a2d_lora(seed=C.DLM_TRAIN.seed)
    ids = special_ids(tokenizer)
    print(f"  tokenizer: vocab {ids['vocab']} mask {ids['mask']} eos {ids['eos']}")

    source = "random init (Qwen3 backbone + fresh LoRA)"
    if not args.init_random:
        if not (args.init_from / "model.safetensors").exists():
            print(f"FAIL  no upstream checkpoint at {args.init_from}; run "
                  f"{CELL_MINIMIND} first or pass --init-random")
            return 1
        load_into(model, load_state_dict(args.init_from))
        source = args.init_from.name
    print(f"  init from {source}")
    print(f"  LoRA r={LORA_R}: {trainable_report(model)}")

    data = build_sft_data(tokenizer, n_docs=args.n_docs, block_size=128)
    train_cfg = dataclasses.replace(C.DLM_TRAIN, steps=args.steps or C.DLM_TRAIN.steps,
                                    batch_size=C.DLM_TRAIN.batch_size)
    eval_batches = [torch.arange(train_cfg.batch_size)]
    n_sup = int((data.labels != -100).sum())   # IGNORE, spelled out
    print(f"  data: {len(data)} blocks x {data.block_size} tokens, "
          f"{n_sup:,} supervised ({100 * n_sup / max(data.labels.numel(), 1):.1f}%)")

    loop = Loop(model, train_cfg, make_minimind_loss(model, ids["mask"]),
                out_dir=C.run_dir(CELL))

    config = {"cell": CELL, "base": BASE, "method": METHOD,
              "model_id": "Qwen/Qwen3-0.6B", "train": C.as_dict(train_cfg),
              "mask_id": ids["mask"], "special_ids": ids, "lora_r": LORA_R,
              "lora_alpha": 16, "trainable": trainable_report(model),
              "init_from": source, "loss": "A2D restricted to assistant spans",
              "tokenizer_track": "Qwen 151,669 (SPEC 3.3)"}

    summary = finish_cell(
        loop, data, cell=CELL, base=BASE, method=METHOD,
        ckpt_dir=CKPT, run_dir=C.run_dir(CELL), config=config,
        sample_fn=lambda lp: sample_lora_dlm(lp.model, tokenizer,
                                             # sample_lora_dlm takes *ids*, not
                                             # strings -- cell 10 uses
                                             # prompts_for(tokenizer) for the same
                                             # reason.
                                             sft_prompts_for(tokenizer, SFT_PROMPTS,
                                                             ids["eos"]), 32,
                                             mask_id=ids["mask"],
                                             block_size=args.block_size,
                                             steps=args.dlm_steps,
                                             temperature=args.temperature),
        eval_batches=eval_batches,
        notes=f"from {source}, LoRA r{LORA_R} q/k/v/o, "
              f"{trainable_report(model)['ratio_pct']}% trainable, "
              f"{n_sup:,} supervised tok, Qwen tokenizer")

    print(f"\n  loss {summary['loss']['first']} -> {summary['loss']['last']}  "
          f"eval {summary.get('eval_loss')}")
    totals = summary.get("totals", {})
    print(f"  supervised tokens {totals.get('target_tokens', 0):,}  "
          f"skipped steps {totals.get('skipped_steps', 0)}")
    for row in summary["repetition"]["prompts"]:
        print(f"\n--- rep4 {row['rep4']}\n{row['text'][:300]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())