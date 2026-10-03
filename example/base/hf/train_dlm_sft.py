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
from base.hf.model import SFT_PROMPTS
from common.cell import finish_cell
from common.ckpt import load_into_any
from common.cli import add_common_args, resolve_sft
from common.cli import resume as resume_from
from common.loop import Loop
from common.sft import encode_prompt
from common.verify import verify_checkpoint
# SPEC 2.3's SFT column: cell -> (row, objective). `--cell` must be one of
# these; an unknown name is an error rather than a new row in RESULTS.md,
# because a typo would otherwise create a cell no SPEC review ever looks at.
# cell -> (row, objective, upstream cell name). `--cell` must be one of these; an
# unknown name is an error rather than a new row in RESULTS.md, because a typo
# would otherwise create a cell no SPEC review ever looks at.
#
# The eight entries are the two directions of G3. `{mini,dllm}_a2d-sft` is the
# **A2D first** arm: pretrain, A2D, then SFT on chat. `sft-{mini,dllm}_a2d-sft` is
# the **original** arm: pretrain, SFT, A2D, then SFT again -- the trailing stage is
# what G3's cell row asks for ("ar-sft -> A2D -> DLM-SFT") and it is what puts both
# routes on the same evaluation (assistant spans, same objective), so that the
# difference between them is the order and nothing else. ISSUES.md DOEN-8.
CELLS = {
    "ar-3.2m-hf-mini_a2d-sft": ("hf", "minimind", "ar-3.2m-hf-mini_a2d"),
    "ar-3.2m-hf-dllm_a2d-sft": ("hf", "dllm", "ar-3.2m-hf-dllm_a2d"),
    "ar-3.2m-native-mini_a2d-sft": ("native", "minimind", "ar-3.2m-native-mini_a2d"),
    "ar-3.2m-native-dllm_a2d-sft": ("native", "dllm", "ar-3.2m-native-dllm_a2d"),
    "ar-3.2m-hf-sft-mini_a2d-sft": ("hf", "minimind", "ar-3.2m-hf-sft-mini_a2d"),
    "ar-3.2m-hf-sft-dllm_a2d-sft": ("hf", "dllm", "ar-3.2m-hf-sft-dllm_a2d"),
    "ar-3.2m-native-sft-mini_a2d-sft": ("native", "minimind", "ar-3.2m-native-sft-mini_a2d"),
    "ar-3.2m-native-sft-dllm_a2d-sft": ("native", "dllm", "ar-3.2m-native-sft-dllm_a2d"),
    # G2: a DLM pretrained from scratch, then SFT'd. Same script, no special
    # case -- the upstream is just a Direct-DLM checkpoint instead of an A2D one.
    "dlm-3.2m-hf-sft": ("hf", "minimind", "dlm-3.2m-hf"),
    "dlm-3.2m-dllm-sft": ("hf", "dllm", "dlm-3.2m-dllm"),
}

def _hf_parts(objective: str):
    from base.hf.a2d_dllm import (DLLMLoss, build_a2d_model, make_a2d_loss as dllm_loss,
                                  sample_dllm)
    from base.hf.a2d_minimind import make_a2d_loss as minimind_loss
    from base.hf.model import build_model, sample_sft

    if objective == "minimind":
        return (build_model, sample_sft, lambda m: minimind_loss(m),
                "hand-written A2D, assistant spans only")

    def sample_sft_dllm(model, **kw):
        """dllm's sampler on the exact prefix an SFT block starts with.

        The sampler cannot be `sample_sft`: that goes through `generate`, which
        assumes causal decoding, and the A2D model is bidirectional. Passing it
        one blew up in `create_causal_mask` with "expanded size of the tensor (1)
        must match the existing size (22)".
        """
        return sample_dllm(model,
                           prompts=[encode_prompt(p) for p in SFT_PROMPTS],
                           prompt_labels=list(SFT_PROMPTS),
                           block_size=kw.get("block_size", 8),
                           steps=kw.get("steps", 8),
                           temperature=kw.get("temperature", 0.3))

    # dllm's `time_epsilon` stays at the package default 1e-3 rather than taking
    # a flag: OPEN-2 keeps the two `t` conventions documented and aligned at
    # comparison time, and a flag here would make the convention a per-run choice
    # nobody records.
    #
    # `build_a2d_model`, not `build_model`: dllm's loss forwards with
    # `attention_mask=None`, so the model itself has to be bidirectional.
    # Same bug as DONE-13, second copy -- the trailing `*-a2d-sft` cells were
    # causal for the same reason the `*-sft-*_a2d` ones were. The minimind
    # objective is safe either way: its loss passes an explicit 4D zero mask.
    return (build_a2d_model, sample_sft_dllm, lambda m: dllm_loss(DLLMLoss()),
            "dllm MDLMTrainer.compute_loss, labels==-100 outside spans")

def _native_parts(objective: str):
    from base.native.a2d_dllm import make_dllm_loss
    from base.native.a2d_minimind import make_a2d_loss as minimind_loss
    from base.native.a2d_minimind import sample_native_dlm
    from base.native.model import build_native_model, sample_native

    def sample_sft(model, **kw):
        """`sample_native` with the exact prefix an SFT block starts with."""
        return sample_native(model, prompts=list(SFT_PROMPTS),
                             prompt_ids=[encode_prompt(p) for p in SFT_PROMPTS], **kw)

    def builder(cfg, seed):
        # Bidirectionality has to come from the model here too -- see the hf
        # branch above. The native port keeps it in `_last_is_causal`.
        m = build_native_model(cfg, seed=seed)
        if objective == "dllm":
            m.set_causal(False)
        return m

    if objective == "minimind":
        return (build_native_model, sample_sft, lambda m: minimind_loss(m),
                "hand-written A2D, assistant spans only")
    def sample_sft_dllm(model, **kw):
        """The block sampler on the exact prefix an SFT block starts with.

        Not `sample_sft`: the AR sampler forces `is_causal=True` on every step,
        which is wrong for a model switched to bidirectional for A2D.
        `sample_native_dlm` reads logits through `native_forward`, so it is
        implementation-agnostic -- dllm supplies the objective here, not the
        sampler. Same shape as `train_a2d_from_sft.py`.
        """
        return sample_native_dlm(model, prompt_ids=[encode_prompt(p) for p in SFT_PROMPTS],
                                 prompt_labels=list(SFT_PROMPTS),
                                 block_size=kw.get("block_size", 8),
                                 steps=kw.get("steps", 8),
                                 temperature=kw.get("temperature", 0.3))

    return (builder, sample_sft_dllm, lambda m: make_dllm_loss(m),
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
    base_label = C.BASE_OF[row]
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

    row, objective, upstream_cell = CELLS[args.cell]
    builder, sampler, loss_factory, loss_note, base_label = build_parts(row, objective)
    method = f"{objective.upper()}-A2D-SFT"

    upstream = args.init_from or C.cell_dir(upstream_cell)
    train_cfg, data, eval_batches = resolve_sft(args, C.DLM_TRAIN)

    model = builder(C.TIER0, seed=args.seed)
    source = "random init"
    if not args.init_random:
        if not (upstream / "model.safetensors").exists():
            print(f"FAIL  no upstream checkpoint at {upstream}; run "
                  f"{upstream.name} first or pass --init-random")
            return 1
        load_into_any(model, upstream)
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
