#!/usr/bin/env python
"""SFT on a DLM checkpoint -- the `{mini,dllm}_a2d-sft` cells, one script.

G3 asks "SFT first or A2D first?". This is the **A2D first** arm: take a
checkpoint that is already a DLM and fine-tune it on the assistant spans of the
SFT corpus. The mirror arm (`sft-{mini,dllm}_a2d`, SPEC's original route) takes an
SFT'd AR model and converts it.

One script rather than four cells' worth of copies: the row, the objective, the
cell name and the upstream checkpoint are parameters. The long-run skill records
what happens when the second caller copies instead of importing -- the two copies
drift and the comparison the experiment existed for stops meaning anything.

**The cell name decides the row, not the other way round.** An earlier version of
this file hard-coded the transformers model and happily ran `ar-3.2m-native-*`
cells on it, producing plausible numbers for the wrong implementation.
`test_sft_cells.py` §4 now pins the mapping.

Both objectives accept an SFT batch ([2, B, L], row 1 labels carrying IGNORE
outside the assistant spans):

* the hand-written one restricts the masking *and* the denominator
  (`common/a2d.py::corrupt(valid=...)`);
* dllm's own `compute_loss` already derives `maskable_mask = labels != -100`, so
  no patch to the package is needed. `test_dlm_sft.py` §4 pins that upstream
  behaviour: ignoring every label drives the loss to exactly 0.

    .venv/bin/python example/base/hf/train_dlm_sft.py \
        --cell ar-3.2m-hf-mini_a2d-sft
    .venv/bin/python example/base/hf/train_dlm_sft.py \
        --cell ar-3.2m-native-dllm_a2d-sft
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
from common.verify import verify_checkpoint
from base.hf.model import SFT_PROMPTS

# SPEC 2.3's SFT column: cell -> (row, objective). `--cell` must be one of
# these; an unknown name is an error rather than a new row in RESULTS.md,
# because a typo would otherwise create a cell no SPEC review ever looks at.
CELLS = {
    "ar-3.2m-hf-mini_a2d-sft": ("hf", "minimind"),
    "ar-3.2m-hf-dllm_a2d-sft": ("hf", "dllm"),
    "ar-3.2m-native-mini_a2d-sft": ("native", "minimind"),
    "ar-3.2m-native-dllm_a2d-sft": ("native", "dllm"),
}


def _hf_parts(objective: str):
    from base.hf.a2d_dllm import DLLMLoss, make_a2d_loss as dllm_loss
    from base.hf.a2d_minimind import make_a2d_loss as minimind_loss
    from base.hf.model import build_model, sample_sft

    if objective == "minimind":
        return (build_model, sample_sft, lambda m: minimind_loss(m),
                "hand-written A2D, assistant spans only")
    # dllm's `time_epsilon` stays at the package default 1e-3 rather than taking
    # a flag: OPEN-2 keeps the two `t` conventions documented and aligned at
    # comparison time, and a flag here would make the convention a per-run choice
    # nobody records.
    return (build_model, sample_sft, lambda m: dllm_loss(DLLMLoss()),
            "dllm MDLMTrainer.compute_loss, labels==-100 outside spans")


def _native_parts(objective: str):
    from base.native.a2d_dllm import make_dllm_loss
    from base.native.a2d_minimind import make_a2d_loss as minimind_loss
    from base.native.model import build_native_model, sample_native

    def sample_sft(model, **kw):
        """`sample_native` with the chat shape the SFT rows trained on."""
        return sample_native(model, prompts=[f"user\n{p}\nassistant\n"
                                            for p in SFT_PROMPTS], **kw)

    if objective == "minimind":
        return (build_native_model, sample_sft, lambda m: minimind_loss(m),
                "hand-written A2D, assistant spans only")
    return (build_native_model, sample_sft, lambda m: make_dllm_loss(m),
            "dllm MDLMTrainer.compute_loss on the native model")


def build_parts(row: str, objective: str):
    """(model builder, sampler, loss factory, loss note).

    The loss factory takes the model and returns the `loss_fn` closure -- an
    earlier version returned a bare `lambda: make_a2d_loss`, which evaluates to
    the *function object*, so Loop called the factory with (model, batch) and got
    a closure back. TypeError: cannot unpack non-iterable function object.

    Row decides the *model*, not just the loss. Nothing here is inferred from
    anything but the cell name.
    """
    parts = _hf_parts(objective) if row == "hf" else _native_parts(objective)
    builder, sampler, factory, note = parts
    base_label = "Qwen-mini-HF" if row == "hf" else "Qwen-mini-MiniMind"
    return builder, sampler, factory, note, base_label


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS),
                    help="SPEC 2.3 cell name; decides the row, the objective and the upstream")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--init-from", type=Path,
                    help="upstream DLM checkpoint (default: the cell's A2D cell)")
    ap.add_argument("--init-random", action="store_true",
                    help="skip the upstream checkpoint (ablation)")
    args = ap.parse_args()

    row, objective = CELLS[args.cell]
    builder, sampler, loss_factory, loss_note, base_label = build_parts(row, objective)
    method = f"{objective.upper()}-A2D-SFT"

    upstream = args.init_from or C.cell_dir(args.cell.replace("-sft", ""))
    train_cfg, data, eval_batches = resolve_sft(args, C.DLM_TRAIN)

    model = builder(C.TIER0, seed=args.seed)
    source = "random init"
    if not args.init_random:
        if not (upstream / "model.safetensors").exists():
            print(f"FAIL  no upstream checkpoint at {upstream}; run "
                  f"{upstream.name} first or pass --init-random")
            return 1
        load_into(model, load_state_dict(upstream))
        source = upstream.name

    loop = Loop(model, train_cfg, loss_factory(model), out_dir=C.run_dir(args.cell))
    if args.resume:
        resume_from(loop, C.cell_dir(args.cell))

    config = {
        "cell": args.cell, "base": base_label, "method": method,
        "row": row, "objective": objective,
        "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
        "data": str(data.prepared_dir), "init_from": source,
        "loss": loss_note, "mask_id": C.MASK,
    }

    summary = finish_cell(
        loop, data, cell=args.cell, base=base_label, method=method,
        ckpt_dir=C.cell_dir(args.cell), run_dir=C.run_dir(args.cell), config=config,
        sample_fn=lambda lp: sampler(lp.model),
        eval_batches=eval_batches,
        notes=f"from {source}, {loss_note}, lr {train_cfg.lr:.0e}")

    print(f"\n  loss {summary['loss']['first']} -> {summary['loss']['last']}  "
          f"eval {summary.get('eval_loss')}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(C.cell_dir(args.cell),
                                   lambda: builder(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]

    for srow in summary["repetition"]["prompts"]:
        print(f"\n--- {srow['prompt']!r}  rep4 {srow['rep4']}\n{srow['text'][:200]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())