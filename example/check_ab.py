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
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common import config as C
from common.ckpt import load_into, load_state_dict
from common.data import build_packed, fixed_batches
from common.verify import logits_delta
from base.hf.model import build_model as build_hf
from base.native.model import build_native_model

HF_AR = C.cell_dir("ar-3.2m-hf")
NATIVE_AR = C.cell_dir("ar-3.2m-native")
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

    result = {"status": "PASS" if delta <= args.tol else "FAIL", "max_logit_delta": delta,
              "tol": args.tol, "weights": weights}
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


def bench_losses(ours_fn, dllm_fn, repeats: int = 20) -> dict:
    """ms/step of one full loss computation, per implementation.

    Cell 3 runs at 46k tok/s and cell 4 at 29k, and "the library is slower" is
    not an actionable statement. What is timed here is *everything the loss
    function does*, which is not the same as the loss alone: our side re-uses a
    frozen corruption while dllm redraws t and the mask on every call, so this
    is an upper bound on dllm's overhead, not an attribution of it.
    """
    import time

    def timed(fn):
        for _ in range(3):                       # warmup
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats * 1e3

    return {"ours_ms": timed(ours_fn), "dllm_ms": timed(dllm_fn)}


def c2(args) -> dict:
    """Hand-written A2D loss vs dllm's MDLMTrainer.compute_loss. < 1% (SPEC §6).

    The comparison only means something if both sides see the same corruption.
    dllm draws `rand(B)` for t and then `rand(B, L)` for the mask; the
    hand-written path draws in the same order, so reseeding torch immediately
    before each call makes the two corruptions identical. (Seed the *global*
    RNG on purpose -- a private generator would desynchronise them.)
    """
    from common.a2d import a2d_loss, corrupt, sample_t
    from base.hf.a2d_dllm import DLLMLoss, build_a2d_model
    from base.hf.a2d_minimind import dlm_forward

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
    out = {"status": "PASS" if rel < 0.01 and same_t and same_mask else "FAIL",
           "rel_diff": rel, "loss_ours": float(loss_ours), "loss_dllm": float(loss_dllm),
           "same_corruption": same_t and same_mask, "tol": 0.01}

    if args.bench:
        with torch.no_grad():
            ours_ms = bench_losses(
                lambda: a2d_loss(dlm_forward(hf)(x_t), batch, masked, t),
                lambda: dllm_loss(dllm_model, batch))
        out.update({k: round(v, 3) for k, v in ours_ms.items()})
        out["dllm_overhead_x"] = round(ours_ms["dllm_ms"] / max(ours_ms["ours_ms"], 1e-9), 2)
    return out


def c5(args) -> dict:
    """A2D conversion must not lose, invent, or reshape a single weight.

    Compares a *freshly built* A2D model against the AR checkpoint before any
    load. An earlier version compared after the load, i.e. compared the values
    it had just written with themselves, which can never fail.

    `args` is unused: every other check reads at least `--tol` or `--seed` and
    this is a pure weight-census check, so a separate `c5()` would break the
    `CHECKS[name](args)` dispatch. Keeping the parameter keeps the signature
    uniform, and a future `--ckpt`-like knob lands here without touching main.
    """
    from base.hf.a2d_dllm import TIED_ALIASES, build_a2d_model

    if not (HF_AR / "model.safetensors").exists():
        return {"status": "SKIP", "detail": "no AR checkpoint; run cell 1"}

    state = load_state_dict(HF_AR)
    fresh = build_a2d_model(C.TIER0)                 # no AR weights loaded yet
    fresh_state = dict(fresh.state_dict())

    # lm_head.weight is a tied alias: the A2D model has the entry, the
    # checkpoint does not store it (safetensors keeps one tensor per storage).
    extra = sorted(set(fresh_state) - set(state) - set(TIED_ALIASES))
    absent = sorted(set(state) - set(fresh_state))
    reshaped = [k for k in set(state) & set(fresh_state)
                if tuple(state[k].shape) != tuple(fresh_state[k].shape)]

    loaded_missing, loaded_unexpected = fresh.load_state_dict(state, strict=False)
    loaded_missing = [k for k in loaded_missing if k not in TIED_ALIASES]
    drifted = [k for k in state
               if k in fresh_state and not torch.equal(state[k], fresh_state[k])]

    ok = not (extra or absent or reshaped or loaded_missing or loaded_unexpected
              or drifted)
    return {"status": "PASS" if ok else "FAIL", "extra": len(extra),
            "absent": len(absent), "reshaped": len(reshaped),
            "missing": len(loaded_missing), "unexpected": len(loaded_unexpected),
            "drifted": len(drifted), "tensors": len(state)}


def decode_bench(args) -> dict:
    """Time both decoders on the same model, prompt and block/step settings.

    DLM decoding re-runs the whole sequence every step (notes/03 §5), so the
    cost is steps x full_forward. Comparing the hand-written sampler against
    MDLMSampler says whether that loop is worth rewriting; comparing against a
    single forward says how expensive denoising is in absolute terms.
    """
    import time

    from common.a2d import sample_dlm as sample_ours
    from base.hf.a2d_dllm import build_a2d_model, sample_dllm
    from base.hf.a2d_minimind import dlm_forward

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_a2d_model(C.TIER0, ar_ckpt=HF_AR if HF_AR.exists() else None).to(device).eval()
    prompt = torch.tensor([[C.BOS, 100, 200, 300]], device=device)

    def timed(fn, warmup: int = 2, runs: int = 3):
        # Warmup is not optional here: the first timed call pays CUDA context
        # and kernel-selection costs, which for a 33-forward decode loop was
        # larger than the measurement itself (notes/05-measurement.md Bug 3).
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        return min(samples)

    gen = args.gen_len
    num_blocks = math.ceil(gen / args.block_size)
    # dllm divides its `steps` budget across blocks
    # (samplers/mdlm.py:131: `steps = math.ceil(steps / num_blocks)`), while
    # `steps` here means steps *per block* (notes/03 §7: steps >= block_size).
    # At the same "steps=8" label dllm therefore runs 8 forwards and this sampler
    # runs 32. Multiply dllm's budget so the benchmark compares equal work.
    ours = timed(lambda: sample_ours(dlm_forward(model), prompt, gen,
                                     block_size=args.block_size, steps=args.dlm_steps,
                                     temperature=args.temperature))
    theirs = timed(lambda: sample_dllm(model, prompts=("x",), max_new_tokens=gen,
                                       block_size=args.block_size,
                                       steps=args.dlm_steps * num_blocks,
                                       temperature=args.temperature))
    forwards = 1 + (gen + args.block_size - 1) // args.block_size * args.dlm_steps
    return {"status": "INFO", "ours_ms": round(ours, 1), "dllm_ms": round(theirs, 1),
            "gen_len": gen, "block_size": args.block_size, "dlm_steps": args.dlm_steps,
            "full_forwards": forwards,
            "ms_per_forward_ours": round(ours / forwards, 2)}


def c3(args) -> dict:
    """Same as C2, on the native base. C2 and C3 together show the loss does not
    care which implementation of the model it drives."""
    from common.a2d import a2d_loss, corrupt, sample_t
    from base.hf.a2d_dllm import DLLMLoss
    from base.native.a2d_dllm import make_dllm_loss
    from base.native.a2d_minimind import build_bidirectional, native_forward

    if not (NATIVE_AR / "model.safetensors").exists():
        return {"status": "SKIP", "detail": "no native AR checkpoint; run cell 2"}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    nat = build_bidirectional(0).to(device).float().eval()
    load_into(nat, load_state_dict(NATIVE_AR))

    data, batches = _eval_batches(batch=args.batch_size)
    batch = data.batch(batches[0]).to(device)
    b = batch.shape[0]
    eps = 1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    t = sample_t(b, device, epsilon=eps)
    x_t, masked = corrupt(batch, t, C.MASK)
    with torch.no_grad():
        ours = a2d_loss(native_forward(nat)(x_t), batch, masked, t)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    theirs = make_dllm_loss(nat, epsilon=eps)(nat, batch)[0].detach()

    # Verify both sides saw the same corruption rather than trusting two similar
    # loss numbers: two different masks can still agree inside 1%.
    torch.manual_seed(args.seed)
    t_ref = sample_t(b, device, epsilon=eps)
    same_t = bool(torch.equal(t, t_ref))
    probs = torch.rand(batch.shape, device=device)
    same_mask = bool(torch.equal(probs < t_ref[:, None], masked))

    rel = abs(float(ours) - float(theirs)) / max(abs(float(theirs)), 1e-9)
    return {"status": "PASS" if rel < 0.01 and same_t and same_mask else "FAIL",
            "rel_diff": rel, "loss_ours": float(ours), "loss_dllm": float(theirs),
            "same_corruption": same_t and same_mask, "tol": 0.01}


def c4(args) -> dict:
    """`N steps` must equal `M + resume + (N-M)`, step for step.

    Runs on the real Tier 0 model, not a toy: the toy in test_common.py passes
    and still missed this once, because the resumed run restarted the epoch's
    batch order and nothing raised.

    Runs twice, once per RNG consumption, because there are two independent
    things to restore and `ar_loss` exercises only the first:

      * `ar_loss` draws nothing from the RNG, so this arm tests the batch
        stream and the optimizer.
      * `_rng_loss` masks with `torch.rand`, so this arm also tests the RNG
        snapshot. It is the arm that matters for every DLM cell: the loss
        itself consumes randomness, so a resumed DLM run would train on a
        different corruption sequence than a straight one. The snapshot was
        once taken before the training loop instead of after it, and this arm
        is what catches that -- the `ar_loss` arm passed the whole time
        (worst_step_delta 5.2e-01 on the `_rng_loss` arm).

    Tolerance is 1e-6 rather than 0: SDPA backward kernels on CUDA are not
    bit-reproducible across runs, so an exact comparison fails intermittently
    on noise and hides a real regression.
    """
    import dataclasses

    from common.data import BlockStream
    from common.loop import Loop
    from common.losses import ar_loss

    def _rng_loss(model, batch):
        """A2D-shaped objective: random mask, then CE on the visible tokens."""
        mask = torch.rand(batch.shape, device=batch.device) < 0.5
        noisy = batch.clone()
        noisy[mask] = C.MASK
        loss, extra = ar_loss(model, noisy)
        return loss, extra

    if args.steps < 20:
        # With steps - 10 <= 0 the "partial" run trains nothing and the
        # continuation replays from scratch: the check passes without ever
        # resuming anything.
        return {"status": "FAIL", "steps": args.steps,
                "detail": "--steps must be >= 20 so the partial run has steps"}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = dataclasses.replace(C.AR_TRAIN, steps=args.steps, log_every=10 ** 9,
                              batch_size=args.batch_size, seed=args.seed)
    data = build_packed(C.DATA_DIR, block_size=C.BLOCK_SIZE, max_blocks=4096)

    straight = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, ar_loss, device=device)
    straight.fit(data)

    tmp = Path(tempfile.mkdtemp())
    try:
        part = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, ar_loss, device=device)
        # max_steps, not a smaller cfg.steps: changing steps changes the cosine
        # schedule and the two runs stop being comparable (this bit me twice).
        part.fit(data, max_steps=args.steps - 10)
        if part.step != args.steps - 10:
            return {"status": "FAIL", "detail": f"partial run reached step {part.step}"}
        part.save(tmp, {})

        cont = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, ar_loss, device=device)
        stream = BlockStream(len(data), cfg.batch_size, cfg.seed)
        cont.load(tmp, stream=stream)
        cont.fit(data, stream=stream)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    worst = max((abs(a - b) for a, b in zip(straight.losses, cont.losses)), default=0.0)
    same_len = len(straight.losses) == len(cont.losses) == args.steps

    rng_worst, rng_len = _c4_rng_arm(args, cfg, data, build_hf, _rng_loss, device)

    return {"status": "PASS" if worst <= 1e-6 and same_len and rng_worst <= 1e-6 else "FAIL",
            "worst_step_delta": worst, "tol": 1e-6, "steps": args.steps,
            "matched": same_len,
            "rng_worst_step_delta": rng_worst, "rng_matched": rng_len}


def _c4_rng_arm(args, cfg, data, build_hf, loss_fn, device) -> tuple[float, bool]:
    """The same straight-vs-resume comparison with a loss that consumes the RNG."""
    import dataclasses
    import shutil
    import tempfile

    from common.data import BlockStream
    from common.loop import Loop

    straight = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, loss_fn, device=device)
    straight.fit(data)

    tmp = Path(tempfile.mkdtemp())
    try:
        part = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, loss_fn, device=device)
        part.fit(data, max_steps=args.steps - 10)
        part.save(tmp, {})
        cont = Loop(build_hf(C.TIER0, seed=cfg.seed), cfg, loss_fn, device=device)
        stream = BlockStream(len(data), cfg.batch_size, cfg.seed)
        cont.load(tmp, stream=stream)
        cont.fit(data, stream=stream)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    worst = max((abs(a - b) for a, b in zip(straight.losses, cont.losses)), default=0.0)
    return worst, len(straight.losses) == len(cont.losses) == args.steps


def c6(args) -> dict:
    """dlm-3.2m-hf vs dlm-3.2m-dllm: same corruption, two implementations.

    The from-scratch arm of G5. Same shape as C2 -- reseed before each side so
    both draw the same t and the same mask, then verify they really did -- but
    starting from random weights instead of an AR checkpoint, which is the only
    difference between the two Direct-DLM cells.
    """
    from common.a2d import a2d_loss, corrupt, sample_t
    from base.hf.direct_dlm import make_direct_loss
    from base.hf.direct_dlm_dllm import Bidirectional, make_dllm_loss

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    eps = 1e-3
    hf_model = build_hf(C.TIER0, seed=args.seed).to(device).float().eval()
    dllm_model = Bidirectional(build_hf(C.TIER0, seed=args.seed)).to(device).float().eval()

    data, batches = _eval_batches(batch=args.batch_size)
    batch = data.batch(batches[0]).to(device)
    b = batch.shape[0]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    t = sample_t(b, device, epsilon=eps)
    x_t, masked = corrupt(batch, t, C.MASK)
    with torch.no_grad():
        ours = make_direct_loss(hf_model, epsilon=eps)(hf_model, batch)[0]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    theirs = make_dllm_loss(epsilon=eps)(dllm_model, batch)[0].detach()

    torch.manual_seed(args.seed)
    t_ref = sample_t(b, device, epsilon=eps)
    same_t = bool(torch.equal(t, t_ref))
    probs = torch.rand(batch.shape, device=device)
    same_mask = bool(torch.equal(probs < t_ref[:, None], masked))

    rel = abs(float(ours) - float(theirs)) / max(abs(float(theirs)), 1e-9)
    return {"status": "PASS" if rel < 0.01 and same_t and same_mask else "FAIL",
            "rel_diff": rel, "loss_hf": float(ours), "loss_dllm": float(theirs),
            "same_corruption": same_t and same_mask, "tol": 0.01}


def c7(args) -> dict:
    """SFT rows: same weights, same SFT batch, two implementations, one loss.

    The SFT cells start from *their own* AR checkpoints, and those two were
    trained independently, so `ar-3.2m-hf-sft` and `ar-3.2m-native-sft` are
    expected to end at slightly different losses (measured: 6.4502 vs 6.4344,
    0.24% apart) -- and that difference is upstream, not the SFT step.

    So this check does what C1 does for the AR rows: load **one** checkpoint into
    both implementations and run the same assistant-only loss over the same
    batches. Anything above 1e-5 is a real divergence in the SFT path.
    """
    from common.losses import sft_ar_loss
    from common.sft import load_sft

    prepared = C.ROOT / "data" / "prepared" / "sft-mini-6400"
    if not (prepared / "meta.json").exists():
        return {"status": "SKIP", "reason": "sft-mini-6400 not built"}
    src = Path(args.ckpt or HF_AR)
    if not (src / "model.safetensors").exists():
        return {"status": "SKIP", "reason": f"no checkpoint at {src}"}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hf_model = build_hf(C.TIER0, seed=args.seed).to(device).float().eval()
    native_model = build_native_model(C.TIER0, seed=args.seed).to(device).float().eval()
    load_into(hf_model, load_state_dict(src))
    load_into(native_model, load_state_dict(src))

    data = load_sft(prepared, block_size=C.BLOCK_SIZE, max_blocks=256)
    batches = fixed_batches(4, args.batch_size, len(data))
    worst, n = 0.0, 0
    for idx in batches:
        batch = data.batch(idx).to(device)
        with torch.no_grad():
            ours = float(sft_ar_loss(hf_model, batch)[0])
            theirs = float(sft_ar_loss(native_model, batch)[0])
        worst = max(worst, abs(ours - theirs))
        n += 1
    return {"status": "PASS" if worst <= args.tol else "FAIL",
            "max_abs_loss_delta": worst, "batches": n, "tol": args.tol,
            "source": src.name}


CHECKS = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6, "c7": c7}
EXTRAS = {"decode-bench": decode_bench}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=[*CHECKS, *EXTRAS, "all"])
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40,
                    help="training steps for c4 (must be >= 10)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--random", action="store_true", help="compare freshly initialised weights")
    ap.add_argument("--layer-report", action="store_true",
                    help="print per-layer logit deltas (C1)")
    ap.add_argument("--bench", action="store_true",
                    help="time each loss implementation on the same batch (C2)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ckpt", help="checkpoint C7 loads into both rows (default: ar-3.2m-hf)")
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dlm-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    if args.check in EXTRAS:
        print(f"[{args.check}] " + " ".join(f"{k}={v}" for k, v in EXTRAS[args.check](args).items()))
        return 0

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