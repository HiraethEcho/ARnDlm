"""One training loop for every row.

Both base models (transformers Qwen3 and the native torch port) and both
hand-written objectives (AR cross-entropy, A2D masked-diffusion CE) go through
this class. Keeping a single loop is what makes the A/B comparison meaningful:
if the loop differed between rows, a loss delta could never be attributed to
the model.

The dllm row does not use this class -- that cell exists precisely to exercise
the upstream `MDLMTrainer` path (T2b).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch

from . import ckpt as ckpt_io
from .config import TrainCfg, as_dict
from .data import BlockStream, PackedData
from .metrics import TpsMeter, loss_windows
from .seeds import autocast, load_rng_state, pick_device, rng_state, seed_all


def _optimizer(model: torch.nn.Module, cfg: TrainCfg) -> torch.optim.Optimizer:
    """Weight decay on matrices only, and tied weights deduplicated by id().

    SPEC 3.1 calls the dedup out explicitly: with tied embeddings, `parameters()`
    yields the same tensor twice, and passing it twice to the optimizer makes
    AdamW apply two updates to one tensor.
    """
    seen, decay, no_decay = set(), [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=1e-8)


def _lr_at(step: int, cfg: TrainCfg) -> float:
    """Linear warmup, then cosine to `min_lr_ratio * lr`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


class Loop:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: TrainCfg,
        loss_fn,
        device=None,
        out_dir: Path | None = None,
        amp: str = "bfloat16",
        require_cuda: bool = True,
    ):
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.device = device or pick_device(require_cuda)
        self.amp = amp
        self.out_dir = Path(out_dir) if out_dir else None
        self.model.to(self.device)
        self.opt = _optimizer(model, cfg)
        self.step = 0
        self.losses: list[float] = []
        self.rows: list[dict] = []
        self.stream: BlockStream | None = None
        seed_all(cfg.seed)

    # ---------------------------------------------------------------- training

    def fit(self, data: PackedData, stream: BlockStream | None = None, sample_fn=None,
            max_steps: int | None = None) -> dict:
        """Run until `max_steps` (or cfg.steps) updates have been done.

        `max_steps` stops a run early *without* touching the LR schedule, which
        a smaller `cfg.steps` would do -- acceptance C4 compares `N + resume`
        against one long run, and the two halves must follow the same schedule.
        """
        stream = stream or BlockStream(len(data), self.cfg.batch_size, self.cfg.seed)
        self.stream = stream          # remembered so save() cannot save a foreign one
        meter = TpsMeter()
        self.model.train()
        n_tokens = self.cfg.batch_size * data.block_size
        limit = self.cfg.steps if max_steps is None else min(self.cfg.steps, self.step + max_steps)

        while self.step < limit:
            idx = stream.next()
            batch = data.batch(idx).to(self.device, non_blocking=True)

            lr = _lr_at(self.step, self.cfg)
            for group in self.opt.param_groups:
                group["lr"] = lr

            if self.device.type == "cuda":
                torch.cuda.synchronize()          # exclude queue time from the step
            t0 = time.perf_counter()

            with autocast(self.device, self.amp):
                loss, extra = self.loss_fn(self.model, batch)
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - t0

            self.step += 1
            value = float(loss.detach())
            self.losses.append(value)
            meter.add(n_tokens, step_seconds)      # only the update, never validation
            self.rows.append({
                "step": self.step,
                "loss": round(value, 6),
                "lr": round(lr, 8),
                "tokens": meter.tokens,
                "train_only_tps": round(meter.train_tps, 1),
                "wall_tps": round(meter.wall_tps, 1),
            })
            if self.step % self.cfg.log_every == 0 or self.step == 1:
                print(f"step {self.step:>4}/{limit}  loss {value:.4f}  "
                      f"lr {lr:.2e}  train {meter.train_tps:,.0f} tok/s  "
                      f"wall {meter.wall_tps:,.0f} tok/s", flush=True)

        summary = {
            "steps": self.step,
            "tokens": meter.tokens,
            "loss": loss_windows(self.losses, self.cfg.loss_window),
            "throughput": meter.as_dict(),
            "config": as_dict(self.cfg),
            "extra": extra if isinstance(extra, dict) else {},
        }
        if sample_fn is not None:
            summary["sample"] = sample_fn(self)
        if self.out_dir:
            self._write_logs(summary)
        return summary

    def _write_logs(self, summary: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        with (self.out_dir / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    # ------------------------------------------------------------- checkpoints

    def save(self, out_dir: Path, config: dict, meta: dict | None = None,
             stream: BlockStream | None = None) -> Path:
        stream = stream or self.stream
        out_dir = ckpt_io.save_model(self.model, out_dir, config, meta)
        ckpt_io.save_train_state(
            out_dir,
            step=self.step,
            optimizer=self.opt.state_dict(),
            losses=self.losses,
            rng=rng_state(),
            stream=stream.state() if stream is not None else None,
        )
        return out_dir

    def load(self, out_dir: Path, stream: BlockStream | None = None,
             load_optimizer: bool = True) -> None:
        """Restore model + optimizer + RNG + batch position.

        Restoring the stream matters as much as the optimizer: acceptance C4
        requires `save at N + continue` to be bit-identical to one long run.
        """
        out_dir = Path(out_dir)
        state = ckpt_io.load_state_dict(out_dir)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
        train_state = ckpt_io.load_train_state(out_dir)
        self.step = int(train_state["step"])
        self.losses = list(train_state.get("losses", []))
        if load_optimizer:
            self.opt.load_state_dict(train_state["optimizer"])
            load_rng_state(train_state["rng"])
        if stream is not None and train_state.get("stream") is not None:
            stream.load_state(train_state["stream"])