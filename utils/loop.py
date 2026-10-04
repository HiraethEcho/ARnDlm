"""utils/loop.py -- 训练循环: 优化器、LR 调度、单步计时、验证、存盘/恢复。

所有路线共用同一份循环, A/B 差值才能归因到模型。
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch

from . import checkpoint
from .config import TrainCfg, as_dict
from .data import BlockStream, PackedData
from .device import autocast, pick_device
from .metrics import TpsMeter, loss_windows
from .seed import load_rng_state, rng_state, seed_all


def _optimizer(model: torch.nn.Module, cfg: TrainCfg) -> torch.optim.Optimizer:
    # 参数分两组: dim>=2 的矩阵用 weight_decay, 其余 (norm/bias) 不衰减
    seen, decay, no_decay = set(), [], []
    for name, p in model.named_parameters():
        # 已经收过的参数 (tied 权重) 跳过
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
    # 前 warmup_steps 步线性升温, 之后按 cosine 降到 min_lr_ratio * lr
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


class Loop:
    """训练循环本体。模型在这里初始化之前就该有种子 (seed_all 在 build_model 里调)。"""

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
        # 设备选择会打印选了什么
        self.device = device or pick_device(require_cuda)
        self.amp = amp
        self.out_dir = Path(out_dir) if out_dir else None
        # 权重搬到 GPU
        self.model.to(self.device)
        self.opt = _optimizer(model, cfg)
        self.step = 0
        self.losses: list[float] = []
        self.rows: list[dict] = []
        self.stream: BlockStream | None = None
        self._stream_state: dict | None = None
        self._rng_after_train: dict | None = None
        seed_all(cfg.seed)

    # ------------------------------------------------------------- training

    def fit(self, data: PackedData, stream: BlockStream | None = None, sample_fn=None,
            max_steps: int | None = None, eval_batches=None) -> dict:
        """跑到 max_steps (默认 cfg.steps); 返回 summary。"""
        stream = stream or self._resume_stream(data)
        self.stream = stream
        meter = TpsMeter()
        self.model.train()
        # 每步的 token 数 = batch × 块长
        n_tokens = self.cfg.batch_size * data.block_size
        limit = self.cfg.steps if max_steps is None else min(self.cfg.steps, self.step + max_steps)
        extra: dict = {}
        totals: dict = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        while self.step < limit:
            # 取下一批索引, 把块搬到 GPU
            idx = stream.next()
            batch = data.batch(idx).to(self.device, non_blocking=True)
            value, step_seconds, extra = self._train_step(batch)
            # loss 函数报告的标量折进本行
            if isinstance(extra, dict) and "target_tokens" in extra:
                totals["target_tokens"] = totals.get("target_tokens", 0) + extra["target_tokens"]
                totals["skipped_steps"] = totals.get("skipped_steps", 0) + bool(
                    extra.get("skipped"))

            self.step += 1
            self.losses.append(value)
            # 只有参数更新这段进 train-only 分母
            meter.add(n_tokens, step_seconds)
            lr = self.opt.param_groups[0]["lr"]
            row = {
                "step": self.step,
                "loss": round(value, 6),
                "lr": round(lr, 8),
                "tokens": meter.tokens,
                "train_only_tps": round(meter.train_tps, 1),
                "wall_tps": round(meter.wall_tps, 1),
            }
            for k, v in (extra if isinstance(extra, dict) else {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = round(float(v), 6)
            self.rows.append(row)
            if self.step % self.cfg.log_every == 0 or self.step == 1:
                print(f"step {self.step:>4}/{limit}  loss {value:.4f}  "
                      f"lr {lr:.2e}  train {meter.train_tps:,.0f} tok/s  "
                      f"wall {meter.wall_tps:,.0f} tok/s", flush=True)

        # 训练一停就快照 RNG; eval 和采样会消耗随机数
        self._rng_after_train = rng_state()

        # 验证和采样在 summary 组装之前跑, wall 里、train-only 外
        eval_loss = self.eval_loss(data, eval_batches) if eval_batches is not None else None
        sample = sample_fn(self) if sample_fn is not None else None

        summary = {
            "steps": self.step,
            "tokens": meter.tokens,
            "loss": loss_windows(self.losses, self.cfg.loss_window),
            "throughput": meter.as_dict(),
            "eval_loss": eval_loss,
            "sample": sample,
            "peak_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if self.device.type == "cuda" else None),
            "config": as_dict(self.cfg),
            "extra": extra if isinstance(extra, dict) else {},
            "totals": totals,
        }
        if self.out_dir:
            self._write_logs(summary)
        return summary

    def _resume_stream(self, data: PackedData) -> BlockStream:
        # 有存档就从存档位置继续, 否则从头
        stream = BlockStream(len(data), self.cfg.batch_size, self.cfg.seed)
        if self._stream_state is not None:
            stream.load_state(self._stream_state)
            print(f"[resume] batch stream at epoch {stream.epoch} pos {stream.pos}")
        return stream

    def _train_step(self, batch: torch.Tensor) -> tuple[float, float, dict]:
        """一次参数更新。返回 (loss, 本步秒数, loss_fn 的附加标量)。"""
        lr = _lr_at(self.step, self.cfg)
        for group in self.opt.param_groups:
            group["lr"] = lr

        # 计时区间用两次 synchronize 夹住, 量到的才是本步
        if self.device.type == "cuda":
            torch.cuda.synchronize()
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
        return float(loss.detach()), time.perf_counter() - t0, extra

    @torch.no_grad()
    def eval_loss(self, data: PackedData, batches) -> float:
        """冻结批次上的平均 loss, 恒 fp32。"""
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        for idx in batches:
            batch = data.batch(idx).to(self.device)
            with autocast(self.device, "float32"):
                loss, _ = self.loss_fn(self.model, batch)
            total += float(loss)
            count += 1
        if was_training:
            self.model.train()
        return total / max(count, 1)

    def _write_logs(self, summary: dict) -> None:
        # summary.json 一次写全, metrics.csv 一行一步
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        with (self.out_dir / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()) if self.rows
                                    else ["step", "loss", "lr", "tokens",
                                          "train_only_tps", "wall_tps"])
            writer.writeheader()
            writer.writerows(self.rows)

    # ---------------------------------------------------------- checkpoints

    def save(self, out_dir: Path, config: dict, meta: dict | None = None,
             stream: BlockStream | None = None) -> Path:
        # 权重 + 训练状态 (step/optimizer/流位置/RNG) 一起存
        stream = stream or self.stream
        out_dir = checkpoint.save_model(self.model, out_dir, config, meta)
        checkpoint.save_train_state(
            out_dir,
            step=self.step,
            optimizer=self.opt.state_dict(),
            losses=self.losses,
            stream=stream.state() if stream is not None else None,
            config=as_dict(self.cfg),
            rows=self.rows,
            # 存训练结束时的 RNG 快照, 不是当前状态
            rng=(self._rng_after_train if self._rng_after_train is not None
                 else rng_state()),
        )
        return out_dir

    def load(self, out_dir: Path, stream: BlockStream | None = None,
             load_optimizer: bool = True) -> None:
        """恢复权重 + optimizer + RNG + 批次位置。"""
        out_dir = Path(out_dir)
        state = checkpoint.load_state_dict(out_dir)
        checkpoint.load_into(self.model, state)
        train_state = checkpoint.load_train_state(out_dir)
        self.step = int(train_state["step"])
        self.losses = list(train_state.get("losses", []))
        self.rows = list(train_state.get("rows", []))
        saved_cfg = train_state.get("config")
        # 配置和存档不一致就不是"续训", 打印差异
        if saved_cfg and saved_cfg != as_dict(self.cfg):
            changed = {k: (saved_cfg.get(k), v) for k, v in as_dict(self.cfg).items()
                       if saved_cfg.get(k) != v}
            print(f"[resume] config differs from the checkpoint: {changed}")
        if load_optimizer:
            self.opt.load_state_dict(train_state["optimizer"])
            load_rng_state(train_state["rng"])
        if stream is not None and train_state.get("stream") is not None:
            stream.load_state(train_state["stream"])
        self._stream_state = train_state.get("stream")
