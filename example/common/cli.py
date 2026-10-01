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
from pathlib import Path

from . import config as C
from .data import PackedData, build_packed, fixed_batches


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
    return parser


def resolve(args, train_cfg: C.TrainCfg) -> tuple[C.TrainCfg, PackedData, list]:
    """(train config, packed corpus, frozen eval batch indices)."""
    cfg = dataclasses.replace(train_cfg, steps=args.steps, batch_size=args.batch_size,
                              seed=args.seed)
    data = build_packed(args.data_dir, block_size=C.BLOCK_SIZE,
                        max_blocks=args.max_blocks)
    batches = fixed_batches(args.eval_batches, cfg.batch_size, len(data))
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