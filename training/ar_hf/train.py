#!/usr/bin/env python
"""training/ar_hf/train.py -- ar-hf 格子的入口。

流程: 建模型 -> 读语料 -> Loop.fit (训练 + 验证 + 采样) -> 存盘 -> 打印摘要。
对照 example/base/hf/train_ar.py。

CLI 还没写 (utils/cli.py 延后), smoke 参数用 argparse 钉在这里:
    cd ~/Projects/ARnDlm-learn && .venv/bin/python training/ar_hf/train.py
    .venv/bin/python training/ar_hf/train.py --steps 5 --no-sample --no-eval
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.hf_qwen.model import build_model, sample_ar
from training.ar_hf.loss import ar_loss
from utils import config as C
from utils.checkpoint import describe
from utils.data import build_packed, fixed_batches
from utils.loop import Loop

CELL = "ar-512-smoke"


def main() -> int:
    ap = argparse.ArgumentParser(description="ar-hf smoke 训练")
    ap.add_argument("--steps", type=int, default=20, help="训练步数 (默认 20)")
    ap.add_argument("--batch-size", type=int, default=None, help="默认用 config 的值")
    ap.add_argument("--no-eval", action="store_true", help="跳过验证 loss")
    ap.add_argument("--no-sample", action="store_true", help="跳过采样")
    ap.add_argument("--no-save", action="store_true", help="不写 checkpoint")
    args = ap.parse_args()

    # 步数覆盖 config 里的 steps; 其余字段原样
    cfg = dataclasses.replace(C.AR_TRAIN, steps=args.steps,
                              batch_size=(args.batch_size if args.batch_size
                                          else C.AR_TRAIN.batch_size))

    # 读语料 (memmap -> 定长块), 取验证用的冻结批次
    data = build_packed(C.DATA_DIR, block_size=C.BLOCK_SIZE, max_blocks=C.MAX_BLOCKS)
    eval_batches = (None if args.no_eval
                    else fixed_batches(C.EVAL_BATCHES, cfg.batch_size, len(data)))

    # 模型在 Loop 之前建, 种子在 build_model 里打
    model = build_model(C.MODEL, seed=cfg.seed)
    loop = Loop(model, cfg, ar_loss, out_dir=C.run_dir(CELL))

    # 训练主体; 验证和采样在计时区间外, 但在 wall 里
    summary = loop.fit(
        data,
        eval_batches=eval_batches,
        sample_fn=(None if args.no_sample else (lambda lp: sample_ar(lp.model))),
    )

    # 存盘: model.safetensors + config.json + meta.json + train_state.pt
    meta = {"cell": CELL, "params": describe(loop.model)["params_unique"],
            "loss": summary["loss"], "throughput": summary["throughput"],
            "eval_loss": summary.get("eval_loss")}
    if not args.no_save:
        ckpt_dir = loop.save(C.cell_dir(CELL),
                             {"cell": CELL, "model": C.as_dict(C.MODEL),
                              "train": C.as_dict(cfg)},
                             meta)
        print(f"\n[{CELL}] done  ckpt -> {ckpt_dir}")

    # 摘要
    first, last = summary["loss"].get("first"), summary["loss"].get("last")
    print(f"  loss   {first} -> {last}   eval {summary.get('eval_loss')}")
    print(f"  tok/s  train-only {summary['throughput']['train_only_tps']:,.0f}"
          f"  wall {summary['throughput']['wall_tps']:,.0f}")
    print(f"  peak   {summary['peak_gb']} GB")
    sample = summary.get("sample") or {}
    for prompt, text in zip(sample.get("prompts", []), sample.get("texts", [])):
        print(f"--- {prompt!r}\n{text[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
