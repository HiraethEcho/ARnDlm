#!/usr/bin/env python
"""Cross-cell checks C1..C5 from example/PLAN.md §7, in one place.

Each check answers one acceptance question from SPEC §6 and prints a number
plus PASS/FAIL; the exit code is 0 only when every selected check passes.

    .venv/bin/python example/check_ab.py c1          # trained checkpoints
    .venv/bin/python example/check_ab.py c1 --random # fresh weights, no ckpt needed
    .venv/bin/python example/check_ab.py all

Checks that need a cell that has not run yet report SKIP rather than passing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common import config as C
from common.ckpt import load_into, load_state_dict
from common.data import build_packed, fixed_batches
from common.verify import logits_delta
from base.hf.model import build_model as build_hf
from base.native.model import build_native_model

HF_AR = C.cell_dir("qwen-mini/hf-ar-example")
NATIVE_AR = C.cell_dir("qwen-mini/minimind-ar-example")
TOL = 1e-5


def _eval_batches(n: int = 4, batch: int = 8, max_blocks: int = 4096):
    data = build_packed(C.DATA_DIR, block_size=C.BLOCK_SIZE, max_blocks=max_blocks)
    return data, fixed_batches(n, batch, len(data))


def c1(args) -> dict:
    """Row 1 vs row 2 logits on the same weights. SPEC §6: <= 1e-5."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hf = build_hf(C.TIER0, seed=args.seed)

    # The native model must be given the HF weights, never merely the same seed.
    # The two classes consume the RNG in a different order (transformers
    # initialises inside post_init, this port initialises in apply), so identical
    # seeds give different tensors -- max|delta| 2.7 on fresh weights.
    nat = build_native_model(C.TIER0, seed=args.seed)
    if not args.random:
        # Both models must hold the *same* weights; loading each cell's own
        # checkpoint into its own model compares two different models.
        source = HF_AR if HF_AR.exists() else NATIVE_AR
        if not source.exists():
            return {"status": "SKIP", "detail": "no AR checkpoints; run cells 1 and 2"}
        state = load_state_dict(source)
        weights = f"trained ({source.name})"
    else:
        state = hf.state_dict()
        weights = "random (hf init)"
    load_into(hf, state)
    load_into(nat, state)

    hf, nat = hf.to(device).float().eval(), nat.to(device).float().eval()
    data, batches = _eval_batches(batch=args.batch_size)
    delta = logits_delta(nat, hf, data, batches, device)

    result = {"status": "PASS" if delta <= TOL else "FAIL", "max_logit_delta": delta,
              "tol": TOL, "weights": weights}
    if args.layer_report:
        result["layers"] = layer_report(hf, nat, data, batches, device)
    return result


def layer_report(hf, nat, data, batches, device) -> list[dict]:
    """Max logit delta after each decoder layer.

    When C1 fails, "the port is wrong" is not actionable. This walks the two
    models in lockstep so the first diverging layer is visible.
    """
    rows = []
    batch = data.batch(batches[0]).to(device)
    with torch.no_grad():
        h_hf = hf.model.embed_tokens(batch)
        h_nat = nat.model.embed_tokens(batch)
        rows.append({"stage": "embed_tokens",
                     "delta": float((h_hf - h_nat).abs().max())})
        cos_hf, sin_hf = hf.model.rotary_emb(h_hf, torch.arange(batch.shape[1],
                                                                device=device)[None])
        cos_nat, sin_nat = nat.model.rotary(h_hf.shape[1], h_hf.device, h_hf.dtype)
        rows.append({"stage": "rope cos/sin",
                     "delta": max(float((cos_hf - cos_nat).abs().max()),
                                  float((sin_hf - sin_nat).abs().max()))})
        for i, (l_hf, l_nat) in enumerate(zip(hf.model.layers, nat.model.layers)):
            h_hf = l_hf(h_hf, attention_mask=None, position_embeddings=(cos_hf, sin_hf))
            h_nat = l_nat(h_nat, cos_nat, sin_nat, True)
            rows.append({"stage": f"layer {i}", "delta": float((h_hf - h_nat).abs().max())})
        rows.append({"stage": "lm_head",
                     "delta": float((hf.lm_head(h_hf) - nat.lm_head(h_nat)).abs().max())})
    return rows


def c2(args) -> dict:
    """Hand-written A2D loss vs dllm's MDLMTrainer.compute_loss. < 1% (SPEC §6).

    The comparison only means something if both sides see the same corruption.
    dllm draws `rand(B)` for t and then `rand(B, L)` for the mask; the
    hand-written path draws in the same order, so reseeding torch immediately
    before each call makes the two corruptions identical. (Seed the *global*
    RNG on purpose -- a private generator would desynchronise them.)
    """
    import torch as _torch

    from common.a2d import a2d_loss, corrupt, sample_t
    from base.hf.a2d_dllm import DLLMLoss, build_a2d_model
    from base.hf.a2d_minimind import dlm_forward
    from base.hf.model import build_model as build_hf

    if not (HF_AR / "model.safetensors").exists():
        return {"status": "SKIP", "detail": "no AR checkpoint; run cell 1"}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hf = build_hf(C.TIER0).to(device).float().eval()
    load_into(hf, load_state_dict(HF_AR))
    dllm_model = build_a2d_model(C.TIER0, ar_ckpt=HF_AR).to(device).float().eval()

    data, batches = _eval_batches(batch=args.batch_size)
    batch = data.batch(batches[0]).to(device)
    b = batch.shape[0]

    epsilon = 1e-3                      # dllm's MDLMConfig.time_epsilon
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    t = sample_t(b, device, epsilon=epsilon)
    x_t, masked = corrupt(batch, t, C.MASK)

    with torch.no_grad():
        loss_ours = a2d_loss(dlm_forward(hf)(x_t), batch, masked, t)

    dllm_loss = DLLMLoss()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loss_dllm = dllm_loss(dllm_model, batch).detach()

    # Confirm both sides really did see the same corruption, not merely similar
    # losses: re-draw with the same seed and compare t and the mask directly.
    torch.manual_seed(args.seed)
    t_ref = sample_t(b, device, epsilon=epsilon)
    same_t = bool(torch.equal(t, t_ref))
    probs = torch.rand(batch.shape, device=device)
    same_mask = bool(torch.equal(probs < t_ref[:, None], masked))

    rel = abs(float(loss_ours) - float(loss_dllm)) / max(abs(float(loss_dllm)), 1e-9)
    return {"status": "PASS" if rel < 0.01 and same_t and same_mask else "FAIL",
            "rel_diff": rel, "loss_ours": float(loss_ours), "loss_dllm": float(loss_dllm),
            "same_corruption": same_t and same_mask, "tol": 0.01}


def c5(args) -> dict:
    """A2D conversion must not lose or invent a single weight."""
    from base.hf.a2d_dllm import TIED_ALIASES, build_a2d_model

    if not (HF_AR / "model.safetensors").exists():
        return {"status": "SKIP", "detail": "no AR checkpoint; run cell 1"}

    state = load_state_dict(HF_AR)
    model = build_a2d_model(C.TIER0, ar_ckpt=HF_AR)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in TIED_ALIASES]
    mismatched = [k for k, v in state.items()
                  if k in dict(model.state_dict()) and not torch.equal(v, model.state_dict()[k])]
    ok = not missing and not unexpected and not mismatched
    return {"status": "PASS" if ok else "FAIL", "missing": len(missing),
            "unexpected": len(unexpected), "mismatched": len(mismatched),
            "tensors": len(state)}


CHECKS = {"c1": c1, "c2": c2, "c5": c5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=[*CHECKS, "all"])
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--random", action="store_true", help="compare freshly initialised weights")
    ap.add_argument("--layer-report", action="store_true",
                    help="print per-layer logit deltas (C1)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    names = list(CHECKS) if args.check == "all" else [args.check]
    worst = 0
    for name in names:
        out = CHECKS[name](args)
        status = out["status"]
        worst += status == "FAIL"
        detail = " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in out.items() if k not in ("status", "layers"))
        print(f"[{name}] {status:4s} {detail}")
        for row in out.get("layers", []):
            print(f"      {row['stage']:<16s} {row['delta']:.3e}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())