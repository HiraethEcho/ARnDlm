#!/usr/bin/env python
"""Measure training throughput for a Qwen-style AR model on this machine.

Why this exists: PLAN.md gates the model size on a smoke measurement. Hidden
size, batch and seq_len are coupled -- picking them before measuring is backwards.

This script only *drives* transformers; it implements no model of its own. It is
measurement tooling, not the T1a training loop (training/ar_hf/).

Measurement rules come from notes/05-measurement.md and are enforced here:
  - Bug 1: the timer wraps one optimizer step only, so eval/sampling never
    lands in the denominator. train-only and wall-clock are reported separately.
  - Bug 2: numerator and denominator cover the same range (post-warmup deltas).
  - Bug 3: --steps defaults to 400 and --warmup steps are discarded outright.
  - Bug 4: stdout is line-buffered and the output is meant for `tee`, not `tail`.

Example:
    .venv/bin/python scripts/bench_throughput.py --seq-len 512 --batch 16
    .venv/bin/python scripts/bench_throughput.py --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ROCm exposes the GPU through the torch.cuda namespace, same as CUDA.
DEV = "cuda"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/prepared/minimind3-6400"))
    p.add_argument("--vocab", type=int, default=6400)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--inter", type=int, default=1536)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=400, help="Timed steps, excluding warmup.")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-ckpt", action="store_true", help="Slower on this machine; off by default.")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--sweep", action="store_true", help="Run the seq_len x batch grid instead of one point.")
    p.add_argument("--out", type=Path, help="Write the result as JSON here.")
    p.add_argument("--skip-peak", action="store_true", help="Skip the matmul peak measurement.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(a: argparse.Namespace):
    """A Qwen3 config built purely from library defaults -- no custom modules."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    head_dim = a.hidden // a.heads
    cfg = Qwen3Config(
        vocab_size=a.vocab,
        hidden_size=a.hidden,
        intermediate_size=a.inter,
        num_hidden_layers=a.layers,
        num_attention_heads=a.heads,
        num_key_value_heads=a.kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max(4096, a.seq_len),
        tie_word_embeddings=True,
        attn_implementation="sdpa",
        use_cache=False,
    )
    model = Qwen3ForCausalLM(cfg)
    return model.to(device=DEV, dtype=torch_dtype(a.dtype))


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def make_batches(a: argparse.Namespace, rng: np.random.Generator):
    """Yield fixed-length token blocks. Packing, not padding: 1.71x on this box."""
    tokens = np.memmap(a.data / "tokens.bin", dtype="<u2", mode="r")
    n_blocks = tokens.shape[0] // a.seq_len
    if n_blocks < a.batch * (a.steps + a.warmup):
        print(f"warning: only {n_blocks:,} blocks available, reusing the corpus")
    order = np.concatenate([rng.permutation(n_blocks) for _ in range(
        int(np.ceil(a.batch * (a.steps + a.warmup) / n_blocks)) + 1)])
    pos = 0

    def next_batch():
        nonlocal pos
        idx = order[pos:pos + a.batch]
        pos += a.batch
        raw = np.stack([tokens[i * a.seq_len:(i + 1) * a.seq_len] for i in idx])
        import torch

        return torch.from_numpy(raw.astype(np.int64)).to(DEV, non_blocking=True)

    return next_batch


def measure_peak(dtype) -> float:
    """Peak device matmul TFLOPS, so effective/peak is reportable (notes/05)."""
    import torch

    n = 4096
    best = 0.0
    for _ in range(3):
        a = torch.randn(n, n, device=DEV, dtype=dtype)
        b = torch.randn(n, n, device=DEV, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        best = max(best, 2 * n**3 / dt)
        del a, b
    torch.cuda.empty_cache()
    return best / 1e12


def run(a: argparse.Namespace) -> dict:
    import torch

    torch.manual_seed(a.seed)
    # Sweep runs every config in one process. Without this reset the reported
    # peak memory is the max over ALL configs so far, not this one.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(a.seed)
    model = make_model(a)
    model.gradient_checkpointing_enable() if a.grad_ckpt else None
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=a.lr,
        fused=True if a.fused_adam else False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    next_batch = make_batches(a, rng)
    tokens_per_step = a.batch * a.seq_len

    print(f"model   {n_params:,} params ({n_train:,} trainable)")
    print(f"shape   batch={a.batch} seq={a.seq_len} tokens/step={tokens_per_step:,}")
    print(f"dtype   {a.dtype}  grad_ckpt={a.grad_ckpt}  fused_adam={a.fused_adam}")
    print(f"run     warmup={a.warmup} timed={a.steps}", flush=True)

    total_step_time = 0.0
    wall0 = time.perf_counter()
    for step in range(a.warmup + a.steps):
        batch = next_batch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step < a.warmup:
            continue
        total_step_time += dt
        loss_val = float(loss.detach())
        if (step - a.warmup + 1) % max(1, a.steps // 10) == 0:
            n = step - a.warmup + 1
            print(f"  step {n:>4}/{a.steps}  {tokens_per_step / dt:>9,.0f} tok/s"
                  f"  {dt * 1000:>7.1f} ms/step  loss {loss_val:.3f}", flush=True)
    wall = time.perf_counter() - wall0

    train_tps = tokens_per_step * a.steps / total_step_time
    wall_tps = tokens_per_step * a.steps / wall
    # 6 * N * tokens: forward 1x, backward 2x.
    eff_tflops = 6 * n_train * tokens_per_step * a.steps / total_step_time / 1e12

    result = {
        "batch": a.batch, "seq_len": a.seq_len, "params": n_params,
        "trainable": n_train, "dtype": a.dtype, "grad_ckpt": a.grad_ckpt,
        "steps": a.steps, "warmup": a.warmup,
        "train_only_tps": round(train_tps),
        "wall_tps": round(wall_tps),
        "ms_per_step": round(total_step_time / a.steps * 1000, 1),
        "eff_tflops": round(eff_tflops, 3),
        "final_loss": round(loss_val, 4),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "pred_hours_for_332M_tokens": round(332_495_324 / train_tps / 3600, 2),
    }
    return result


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    a = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    print(f"device  {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    if not a.skip_peak:
        peak = measure_peak(torch_dtype(a.dtype))
        print(f"peak    {peak:.2f} TFLOPS matmul ({a.dtype})", flush=True)
    else:
        peak = 0.0

    grid = [(s, b) for s in (128, 256, 512) for b in (8, 16, 32)] if a.sweep else [(a.seq_len, a.batch)]
    results = []
    for seq_len, batch in grid:
        a.seq_len, a.batch = seq_len, batch
        print(f"\n=== seq_len={seq_len} batch={batch} ===", flush=True)
        r = run(a)
        r["peak_tflops"] = round(peak, 2)
        if peak:
            r["eff_vs_peak"] = f"{100 * r['eff_tflops'] / peak:.1f}%"
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())