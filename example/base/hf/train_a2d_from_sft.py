#!/usr/bin/env python
"""SFT first, then A2D -- the `sft-{mini,dllm}_a2d` cells (G3's original route).

The mirror of `train_dlm_sft.py`. That one takes a DLM and fine-tunes it on chat;
this one takes a model that has already been SFT'd and converts it to a DLM.

Which corpus the A2D stage trains on is the whole point of G3. If the original
route's A2D stage used the chat corpus, the two routes would differ in both the
order *and* the data, and "the variable is order" (SPEC G3) would be false. So
this stage trains on the **pretraining** corpus, exactly like
`train_a2d_minimind.py` / `train_a2d_dllm.py`, and the only difference from those
is which checkpoint it starts from.

SPEC is inconsistent about whether a trailing DLM-SFT belongs here: §2.3's
pipeline column stops at A2D while G3's cell row says `ar-sft -> A2D -> DLM-SFT`.
That gap is recorded in ISSUES.md DONE-8. This script implements the four cells
the table lists, each ending after the A2D stage.

    .venv/bin/python example/base/hf/train_a2d_from_sft.py \
        --cell ar-3.2m-hf-sft-mini_a2d
    .venv/bin/python example/base/hf/train_a2d_from_sft.py \
        --cell ar-3.2m-native-sft-dllm_a2d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.cell import finish_cell
from common.ckpt import load_into, load_state_dict
from common.cli import add_common_args, resolve, dlm_steps
from common.cli import resume as resume_from
from common.loop import Loop
from common.verify import verify_checkpoint

CELLS = {
    "ar-3.2m-hf-sft-mini_a2d": ("hf", "minimind"),
    "ar-3.2m-hf-sft-dllm_a2d": ("hf", "dllm"),
    "ar-3.2m-native-sft-mini_a2d": ("native", "minimind"),
    "ar-3.2m-native-sft-dllm_a2d": ("native", "dllm"),
}


def build_parts(row: str, objective: str):
    """(model builder, dlm sampler, loss factory, loss note).

    Sampling uses the DLM decoder, not `sample_sft`: a cell that ends in a DLM
    must be judged by decoding it. Comparing an A2D cell against an AR cell on
    the AR decoder would compare two different things (OPEN-3: the two decoders
    also have different `steps` semantics, so their speeds are not comparable
    without the conversion `check_ab decode-bench` performs).

    Each factory takes the model and returns the closure: `lambda m:
    minimind_loss(m)`. A bare `lambda: minimind_loss` evaluates to the function
    object, so Loop calls the *factory* with (model, batch) and gets a closure
    back -- "cannot unpack non-iterable function object". That mistake has been
    made twice in this session, in two different files, so the shape is now
    fixed in one place rather than at each call site.
    """
    if row == "hf":
        from base.hf.a2d_dllm import DLLMLoss, build_a2d_model
        from base.hf.a2d_dllm import make_a2d_loss as dllm_loss
        from base.hf.a2d_dllm import sample_dllm
        from base.hf.a2d_minimind import make_a2d_loss as minimind_loss
        from base.hf.a2d_minimind import sample_dlm_row
        from base.hf.model import build_model
        factory = (lambda m: minimind_loss(m)) if objective == "minimind" \
            else (lambda m: dllm_loss(DLLMLoss()))
        # dllm's sampler has no `top_k` (it decodes with its own defaults), so it
        # is called through a shim rather than being given a kwarg it cannot
        # accept. Same reason `check_ab decode-bench` has to normalise work.
        sampler = sample_dlm_row if objective == "minimind" else (
            lambda model, **kw: sample_dllm(model, block_size=kw.get("block_size", 8),
                                            steps=kw.get("steps", 8),
                                            temperature=kw.get("temperature", 0.3)))
        # The dllm objective calls MDLMTrainer.compute_loss, which forwards with
        # `attention_mask=None` -- so the *model* has to be bidirectional. A
        # plain Qwen3ForCausalLM answers that with a causal mask, and the cell
        # trains a causal model under a name that says A2D. Measured: editing the
        # last token left every earlier logit bit-identical (delta 0.0) for
        # `build_model` and moved them by 8.3e-01 for `build_a2d_model`.
        #
        # The minimind objective is safe either way: its loss passes an explicit
        # 4D all-zero mask at forward time, so it never depends on the builder.
        builder = build_model if objective == "minimind" else build_a2d_model
        return (builder, sampler, factory,
                "hand-written A2D" if objective == "minimind"
                else "dllm MDLMTrainer.compute_loss")

    from base.native.a2d_dllm import make_dllm_loss
    from base.native.a2d_minimind import make_a2d_loss as minimind_loss
    from base.native.a2d_minimind import sample_native_dlm
    from base.native.model import build_native_model
    factory = (lambda m: minimind_loss(m)) if objective == "minimind" \
        else (lambda m: make_dllm_loss(m))

    def builder(cfg, seed):
        # Same reason as the transformers row above: dllm's loss forwards with no
        # mask, so bidirectionality has to come from the model. The native port
        # keeps it in `self._last_is_causal`, which `set_causal` flips.
        m = build_native_model(cfg, seed=seed)
        if objective == "dllm":
            m.set_causal(False)
        return m

    return (builder, sample_native_dlm, factory,
            "hand-written A2D on native torch" if objective == "minimind"
            else "dllm MDLMTrainer.compute_loss on native torch")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--init-from", type=Path,
                    help="upstream SFT checkpoint (default: the cell's -sft cell)")
    ap.add_argument("--init-random", action="store_true",
                    help="skip the upstream checkpoint (ablation: A2D without SFT)")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8,
                    help="denoising steps per block (OPEN-3: per block, not total)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", dest="top_k", type=int, default=50)
    args = ap.parse_args()

    row, objective = CELLS[args.cell]
    builder, sampler, loss_factory, loss_note = build_parts(row, objective)
    model = builder(C.TIER0, seed=args.seed)
    base_label = C.BASE_OF[row]
    method = f"{objective.upper()}-A2D"

    upstream = args.init_from or C.cell_dir(args.cell.replace("-mini_a2d", "")
                                                          .replace("-dllm_a2d", ""))
    train_cfg, data, eval_batches = resolve(args, C.DLM_TRAIN)

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
        "loss": loss_note, "mask_id": C.MASK, "route": "G3 original (SFT then A2D)",
        "note": "A2D stage trains on the pretraining corpus so that G3's only "
                "variable is the order",
    }

    summary = finish_cell(
        loop, data, cell=args.cell, base=base_label, method=method,
        ckpt_dir=C.cell_dir(args.cell), run_dir=C.run_dir(args.cell), config=config,
        sample_fn=lambda lp: sampler(lp.model, block_size=args.block_size,
                                    steps=dlm_steps(args, args.dlm_steps),
                                    temperature=args.temperature,
                                    top_k=args.top_k),
        eval_batches=eval_batches,
        notes=f"from {source}, {loss_note}, G3 original route, lr {train_cfg.lr:.0e}")

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
        print(f"\n--- {srow['prompt']!r}  rep4 {srow['rep4']}\n{srow['text'][:160]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())