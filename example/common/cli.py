#!/usr/bin/env python
"""Shared CLI plumbing for the ten cells.

Each cell script should be readable as: build a model, define a loss, define a
sampler, call `finish_cell`. The argument parsing, corpus loading and frozen
evaluation batches are identical everywhere, and duplicating them ten times
means ten chances for cell 7 to quietly train on a different corpus than
cell 6.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

from . import config as C
from .data import PackedData, build_packed, fixed_batches
from .sft import SFTData, load_sft

SFT_DATA_DIR = C.ROOT / "data" / "prepared" / "sft-mini-6400"


def add_common_args(parser: argparse.ArgumentParser, train_cfg: C.TrainCfg,
                    default_max_blocks: int = 4096) -> argparse.ArgumentParser:
    parser.add_argument("--steps", type=int, default=train_cfg.steps)
    parser.add_argument("--batch-size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--seed", type=int, default=train_cfg.seed)
    parser.add_argument("--max-blocks", type=int, default=default_max_blocks,
                        help="cap corpus blocks: 4096 blocks x 128 = 524,288 tokens")
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--eval-batches", type=int, default=4,
                        help="frozen batches used for the eval loss in RESULTS")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the cell's checkpoint instead of starting fresh")
    # OPEN-3: the two implementations of `steps` mean different things and
    # passing one to the other silently changes the work done.
    #   --dlm-steps       per BLOCK (this project's convention, notes/03 §7)
    #   --dlm-total-steps whole SEQUENCE, divided over the blocks
    # dllm's own sampler already does the division internally
    # (dllm/core/samplers/mdlm.py: `steps = ceil(steps / num_blocks)`), so
    # --dlm-total-steps 8 and --dlm-steps 2 run the same number of forwards.
    parser.add_argument("--dlm-total-steps", type=int, default=0,
                        help="whole-sequence denoising budget; overrides --dlm-steps "
                             "when set. 0 = use --dlm-steps as-is (per block)")
    return parser


def dlm_steps(args, per_block: int, gen_len: int = 32, block_size: int = 8) -> int:
    """Resolve `--dlm-total-steps` into the per-block number the samplers take.

    One place, so the nine DLM sampling call sites cannot drift apart. Returns
    `per_block` unchanged when no total budget was given, which is what every cell
    did before this existed.
    """
    total = getattr(args, "dlm_total_steps", 0)
    if not total:
        return per_block
    n_blocks = max(1, gen_len // block_size)
    return max(1, math.ceil(total / n_blocks))


def resolve(args, train_cfg: C.TrainCfg) -> tuple[C.TrainCfg, PackedData, list]:
    """(train config, packed corpus, frozen eval batch indices)."""
    cfg = dataclasses.replace(train_cfg, steps=args.steps, batch_size=args.batch_size,
                              seed=args.seed)
    data = build_packed(args.data_dir, block_size=C.BLOCK_SIZE,
                        max_blocks=args.max_blocks)
    batches = fixed_batches(args.eval_batches, cfg.batch_size, len(data))
    if not batches:
        # An empty batch set makes eval_loss return 0.0, which reads as a
        # perfect model and makes the checkpoint verifier report OK.
        raise ValueError(
            f"no eval batches: corpus has {len(data)} blocks, batch size is "
            f"{cfg.batch_size}. Raise --max-blocks or lower --batch-size.")
    return cfg, data, batches


def resolve_sft(args, train_cfg: C.TrainCfg) -> tuple[C.TrainCfg, SFTData, list]:
    """(train config, SFT corpus, frozen eval batches).

    Identical to `resolve` except for the loader: an SFT cell needs the assistant
    spans, so `build_packed` would hand it blocks with no supervision at all and
    every loss would be 0. Reusing the same function name shape keeps the two
    paths from drifting.
    """
    cfg = dataclasses.replace(train_cfg, steps=args.steps, batch_size=args.batch_size,
                              seed=args.seed)
    prepared = Path(args.data_dir) if args.data_dir != C.DATA_DIR else SFT_DATA_DIR
    if not (prepared / "meta.json").exists():
        raise FileNotFoundError(
            f"{prepared} does not look like an SFT prepared directory. Build it "
            f"first: .venv/bin/python example/base/hf/make_sft_data.py")
    data = load_sft(prepared, block_size=C.BLOCK_SIZE, max_blocks=args.max_blocks)
    batches = fixed_batches(args.eval_batches, cfg.batch_size, len(data))
    if not batches:
        raise ValueError(
            f"no eval batches: corpus has {len(data)} blocks, batch size is "
            f"{cfg.batch_size}. Raise --max-blocks or lower --batch-size.")
    return cfg, data, batches


def resume(loop, ckpt_dir: Path):
    """`--resume`: weights + optimizer + RNG + batch position.

    The batch position is part of it. Reloading weights without it replays the
    first batches of the epoch, which looks like a normal run and is not one.
    """
    if not (ckpt_dir / "train_state.pt").exists():
        raise FileNotFoundError(f"no train_state.pt in {ckpt_dir}; nothing to resume")
    loop.load(ckpt_dir)
    return loop