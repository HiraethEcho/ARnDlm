"""utils/metrics.py -- 吞吐（train-only 和 wall 两个口径）与重复率。

只被 utils/loop.py 调用。不用 torch。
"""

from __future__ import annotations

import time


class TpsMeter:
    """每次参数更新记一次 token 数和/train 秒数；wall 从 __init__ 起算。"""

    def __init__(self) -> None:
        self.tokens = 0
        self.train_seconds = 0.0
        self._wall_start = time.perf_counter()

    def add(self, tokens: int, train_seconds: float) -> None:
        # 累计本段训练的 token 数和秒数
        self.tokens += tokens
        self.train_seconds += train_seconds

    @property
    def wall_seconds(self) -> float:
        # 从 meter 创建到现在的真实时间
        return time.perf_counter() - self._wall_start

    @property
    def train_tps(self) -> float:
        # train-only 口径：分母只有参数更新的时间
        return self.tokens / self.train_seconds if self.train_seconds else 0.0

    @property
    def wall_tps(self) -> float:
        # wall 口径：分母是全部经过时间
        return self.tokens / self.wall_seconds if self.wall_seconds else 0.0

    def as_dict(self) -> dict:
        # 两个口径一起落进 summary
        return {
            "tokens": self.tokens,
            "train_seconds": round(self.train_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "train_only_tps": round(self.train_tps, 1),
            "wall_tps": round(self.wall_tps, 1),
        }


def repetition_rates(seqs, ns=(1, 2, 3, 4)) -> dict:
    # rep-n = 1 - 不同的 n-gram 数 / 总 n-gram 数，跨所有序列汇总
    out = {}
    for n in ns:
        total, unique = 0, set()
        for seq in seqs:
            ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
            # 短于 n 的序列跳过, 不当 0.0 计
            if len(ids) < n:
                continue
            grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
            total += len(grams)
            unique.update(grams)
        out[n] = 1.0 - (len(unique) / total) if total else float("nan")
    return out


def loss_windows(losses, k: int = 20) -> dict:
    # 首 k 步和尾 k 步的均值；样本不够时把 k 夹小, 头尾不重叠
    losses = [float(x) for x in losses]
    if not losses:
        return {"first": None, "last": None, "min": None, "mean": None}
    k = max(1, min(k, len(losses) // 2 or 1))
    head = losses[:k]
    tail = losses[-k:]
    return {
        "first": sum(head) / len(head),
        "last": sum(tail) / len(tail),
        "min": min(losses),
        "mean": sum(losses) / len(losses),
    }
