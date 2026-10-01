"""Throughput and repetition metrics.

Two rules from notes/05-measurement.md are encoded here rather than left to
each script:

* Bug 1 -- validation and sampling must stay out of the train-only
  denominator. `add()` therefore takes the seconds measured around one
  parameter update, while wall time is accumulated separately.
* Bug 2 -- numerator and denominator must cover the same scope. Both counters
  start at zero in `__init__`, so a resumed run reports its own numbers.

Repetition rate, not loss, is the metric that decides whether a DLM worked:
notes/03-minimind-dlm.md §11 shows a model whose loss fell while the output
looped.
"""
from __future__ import annotations

import time


class TpsMeter:
    """Token throughput, reported both ways (train-only and wall-clock)."""

    def __init__(self) -> None:
        self.tokens = 0
        self.train_seconds = 0.0
        self._wall_start = time.perf_counter()

    def add(self, tokens: int, train_seconds: float) -> None:
        self.tokens += tokens
        self.train_seconds += train_seconds

    @property
    def wall_seconds(self) -> float:
        return time.perf_counter() - self._wall_start

    @property
    def train_tps(self) -> float:
        return self.tokens / self.train_seconds if self.train_seconds else 0.0

    @property
    def wall_tps(self) -> float:
        return self.tokens / self.wall_seconds if self.wall_seconds else 0.0

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "train_seconds": round(self.train_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "train_only_tps": round(self.train_tps, 1),
            "wall_tps": round(self.wall_tps, 1),
        }


def repetition_rates(seqs, ns=(1, 2, 3, 4)) -> dict[int, float]:
    """rep-n = 1 - unique_ngrams / total_ngrams, pooled over sequences.

    Sequences shorter than n contribute nothing. Reporting them as 0.0 would
    make a degenerate 3-token sample look perfect, which is exactly how a
    broken sampler hides.
    """
    out = {}
    for n in ns:
        total, unique = 0, set()
        for seq in seqs:
            ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
            if len(ids) < n:
                continue
            grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
            total += len(grams)
            unique.update(grams)
        out[n] = 1.0 - (len(unique) / total) if total else float("nan")
    return out


def loss_windows(losses, k: int = 20) -> dict:
    """Mean of the first and last k steps. The smoke runs only *record* this;
    they never assert on it (pipeline validity, not model quality)."""
    losses = [float(x) for x in losses]
    if not losses:
        return {"first": None, "last": None, "min": None, "mean": None}
    head = losses[:k]
    tail = losses[-k:]
    return {
        "first": sum(head) / len(head),
        "last": sum(tail) / len(tail),
        "min": min(losses),
        "mean": sum(losses) / len(losses),
    }