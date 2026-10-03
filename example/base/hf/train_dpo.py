#!/usr/bin/env python
"""`ar-3.2m-hf-sft-dpo` -- G3's DPO arm on the AR row.

DPO needs two models: the policy and a frozen reference. At Tier 0 the reference
is a second copy of the same 3.2M weights (22 MB), so this cell is cheap. On the
LoRA row the reference would be the same base with the adapters disabled, which
peft does for free -- that cell is not built here.

**Why only the AR row.** SPEC's own matrix says the `dlm` row has "no sequence
logprob", and it is right: DPO's objective is a product of autoregressive
factors, while a masked-diffusion model denoises every position at once. Running
DPO on a DLM needs an ELBO formulation first, which is a research question, not a
configuration flag. ISSUES.md OPEN-10.

**Why so little of the corpus.** Measured on 800 pairs: prompt 40 tokens
(median), chosen 577, rejected 626 -- only 10.5% fit a 128-token block. Turn-atomic
dropping keeps ~15% of pairs (62 rows from 400 in the test), which is ~2,600 of
the 17,166 available. Enough for a smoke test; the drop rate is reported in the
run's config so nobody mistakes it for the full set.

    .venv/bin/python example/base/hf/train_dpo.py
    .venv/bin/python example/base/hf/train_dpo.py --steps 20 --max-pairs 200
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from common import config as C
from base.hf.model import SFT_PROMPTS, build_model, sample_sft
from common.cell import finish_cell
from common.ckpt import load_into, load_state_dict
from common.cli import add_common_args
from common.data import BlockStream
from common.dpo import IGNORE, dpo_blocks, dpo_loss
from common.loop import Loop
from common.sft import encode_prompt
from common.verify import verify_checkpoint
METHOD = "AR-SFT-DPO"
# cell -> upstream SFT checkpoint. Same script for both rows: the objective does
# not care which implementation produced the weights it is refining.
CELLS = {
    "ar-3.2m-hf-sft-dpo": "ar-3.2m-hf-sft",
    "ar-3.2m-native-sft-dpo": "ar-3.2m-native-sft",
}
CELL = "ar-3.2m-hf-sft-dpo"
BASE_LABEL = "Qwen-mini-HF"

def make_dpo_loss(policy, reference, beta: float):
    """`loss_fn(model, batch)` for common.loop.Loop.

    The batch is [B, L] ids plus a parallel label tensor; the reference is the
    second model. Row 2i/2i+1 must be one pair -- `common.dpo.dpo_loss` rejects an
    odd batch rather than dropping a row.
    """
    def loss_fn(model, batch):
        ids, labels = batch
        return dpo_loss(model, reference, ids, labels, beta=beta)
    return loss_fn

class PairData:
    """The smallest PackedData-compatible surface `Loop.fit` actually touches.

    Loop only calls `len(data)`, `data.batch(idx)`, `data.block_size` and
    `data.n_tokens`, so this wraps a tensor of pair rows instead of inheriting the
    whole class. `batch` returns [2, B, L] -- row 0 ids, row 1 labels -- which is
    the shape the DPO loss unpacks.
    """

    def __init__(self, ids: torch.Tensor, labels: torch.Tensor):
        self.ids = ids
        self.labels = labels
        self.block_size = int(ids.shape[1])
        self.n_tokens = int(ids.numel())
        self.prepared_dir = None

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.ids[idx].long(), self.labels[idx].long()])

def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(__doc__), C.DLM_TRAIN,
                         default_max_blocks=0)
    ap.add_argument("--cell", default=CELL, choices=sorted(CELLS),
                    help="SPEC cell name; decides the row and the upstream SFT checkpoint")
    ap.add_argument("--max-pairs", type=int, default=4000,
                    help="cap pairs read from dpo.jsonl; most do not fit ctx 128")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--init-from", type=Path)
    ap.add_argument("--init-random", action="store_true",
                    help="skip the SFT checkpoint (ablation)")
    args = ap.parse_args()

    cell = args.cell
    upstream = args.init_from or C.cell_dir(CELLS[cell])
    if not (args.init_random or (upstream / "model.safetensors").exists()):
        print(f"FAIL  no upstream checkpoint at {upstream}; run "
              f"{CELLS[cell]} first or pass --init-random")
        return 1

    # The row comes from the cell name: a `native` cell that ran the transformers
    # model would be a different experiment wearing the same name.
    native = "-native-" in cell
    if native:
        from base.native.model import build_native_model, sample_native
        builder = build_native_model
        base_label = "Qwen-mini-MiniMind"
        # The row decides the sampler too, not just the model. Row 2's sampler is
        # a hand-rolled loop without a KV cache and reads `model.cfg` for the
        # device; the transformers row's `generate` reads `model.device`. Swapping
        # the builder and not the sampler gives
        # "'NativeQwen3ForCausalLM' object has no attribute 'device'".
        def sampler(model, **kw):
            return sample_native(model, prompts=list(SFT_PROMPTS),
                             prompt_ids=[encode_prompt(p) for p in SFT_PROMPTS], **kw)
    else:
        from base.hf.model import build_model, sample_sft
        builder = build_model
        base_label = BASE_LABEL
        sampler = sample_sft

    policy = builder(C.TIER0, seed=args.seed)
    source = "random init"
    if not args.init_random:
        load_into(policy, load_state_dict(upstream))
        source = upstream.name
    # The reference is the SFT checkpoint, frozen: that is what DPO anchors to,
    # not the current weights.
    reference = builder(C.TIER0, seed=args.seed)
    load_into(reference, load_state_dict(upstream if not args.init_random
                                         else C.cell_dir("ar-3.2m-hf")))
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    src = C.ROOT / "data" / "raw" / "minimind_dataset" / "dpo.jsonl"
    ids, labels, stats = dpo_blocks(src, C.BLOCK_SIZE, max_pairs=args.max_pairs)
    if ids.shape[0] == 0:
        print(f"FAIL  no pair fits {C.BLOCK_SIZE} tokens", file=sys.stderr)
        return 1
    # Rows come out of dpo_blocks as [all chosen..., all rejected...]; interleave
    # them so row 2i/2i+1 is one pair, which is the layout dpo_loss requires.
    n_pair = ids.shape[0] // 2
    order = torch.arange(2 * n_pair).reshape(n_pair, 2).T.reshape(-1)
    data = PairData(ids[order], labels[order])
    n_sup = int((data.labels != IGNORE).sum())
    print(f"  pairs kept {n_pair} of {stats['n_pairs']} "
          f"({stats['n_dropped_pairs']} dropped, ctx {C.BLOCK_SIZE}), "
          f"{n_sup:,} supervised tokens")

    train_cfg = C.AR_TRAIN.__class__(**{**C.as_dict(C.DLM_TRAIN),
                                        "steps": args.steps or C.DLM_TRAIN.steps,
                                        "batch_size": args.batch_size})
    # Even batch: dpo_loss rejects an odd one.
    if train_cfg.batch_size % 2:
        train_cfg = train_cfg.__class__(**{**C.as_dict(train_cfg),
                                           "batch_size": train_cfg.batch_size + 1})
    eval_batches = [torch.arange(train_cfg.batch_size) % len(data)]

    stream = BlockStream(len(data), train_cfg.batch_size, seed=train_cfg.seed)
    loop = Loop(policy, train_cfg, make_dpo_loss(policy, reference, args.beta),
                out_dir=C.run_dir(cell))
    # Loop moves the policy to the device; the reference is ours to place, and
    # leaving it on the CPU is a device mismatch inside the embedding lookup.
    reference.to(loop.device)

    config = {"cell": cell, "base": base_label, "method": METHOD,
              "model": C.as_dict(C.TIER0), "train": C.as_dict(train_cfg),
              "init_from": source, "reference": "frozen SFT checkpoint",
              "beta": args.beta, "data": str(src), "corpus": stats,
              "n_pairs_kept": n_pair, "supervised_tokens": n_sup,
              "note": "AR row only: a DLM has no sequence logprob (OPEN-10)"}

    summary = finish_cell(
        loop, data, cell=cell, base=base_label, method=METHOD,
        ckpt_dir=C.cell_dir(cell), run_dir=C.run_dir(cell), config=config,
        sample_fn=lambda lp: sampler(lp.model),
        eval_batches=None, stream=stream,
        notes=f"from {source}, DPO beta {args.beta}, {n_pair}/{stats['n_pairs']} pairs "
              f"fit ctx {C.BLOCK_SIZE}, ref = frozen SFT")
    accs = [r["accuracy"] for r in loop.rows if "accuracy" in r]
    acc = accs[-1] if accs else None
    first_acc = accs[0] if accs else None
    print(f"\n  loss {summary['loss']['first']} -> {summary['loss']['last']}  "
          f"preference accuracy {first_acc} -> {acc}")

    ok = True
    if not args.no_verify:
        report = verify_checkpoint(C.cell_dir(cell), lambda: builder(C.TIER0, seed=0),
                                   loop.model, data, eval_batches, loop.device)
        print(f"  verify  {'OK' if report['ok'] else 'FAIL'}  "
              f"max|dlogit| {report.get('max_logit_delta', float('nan')):.2e}")
        ok &= report["ok"]
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
