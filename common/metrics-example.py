"""common/metrics-example.py -- common/metrics.py 的参考实现。

SPEC 6.4 的口径逻辑, 纯 python/numpy, 不用 torch:
- TpsMeter: train-only 与 wall 两个吞吐口径（验证/采样不进 train-only 分母）
- rep4: 4-gram 重复率（DLM 结论必报; loss 降但复读是 DLM 最典型失败）
- LossHistory: 首尾窗口均值（smoke 判读丢预热段的前置工具）

时钟可注入（clock 参数）, 测试不 sleep。
"""
from __future__ import annotations

import time
from collections import Counter


class TpsMeter:
    """train-only: 只累计 add_train 传入的秒数与 token; wall: 从构造时刻到 report。"""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._t0 = clock()
        self.train_seconds = 0.0
        self.train_tokens = 0

    def add_train(self, tokens: int, seconds: float) -> None:
        """每个训练步调一次。验证 / 采样 / 存盘不走这里。"""
        self.train_tokens += tokens
        self.train_seconds += seconds

    def report(self) -> dict:
        wall = self._clock() - self._t0
        return {
            "train_tokens": self.train_tokens,
            "train_seconds": round(self.train_seconds, 4),
            "wall_seconds": round(wall, 4),
            "train_tps": self.train_tokens / self.train_seconds if self.train_seconds > 0 else 0.0,
            "wall_tps": self.train_tokens / wall if wall > 0 else 0.0,
        }


def rep4(ids) -> float:
    """4-gram 重复率 = 1 - unique/total。无重复 -> 0.0, 全复读 -> 1.0; 不足 4 个 token -> 0.0。"""
    ids = list(ids)
    if len(ids) < 4:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def rep4_worst_window(ids, window: int = 64) -> float:
    """按 window 切段取最差段 (SPEC 6.4: rep-4 在短样本上会跳, 长生成按窗口看)。"""
    ids = list(ids)
    if len(ids) < window:
        return rep4(ids)
    return max(rep4(ids[i:i + window]) for i in range(0, len(ids) - window + 1, window))


class LossHistory:
    """逐步 loss 记录 + 首尾窗口摘要（window 取 config 的 train.loss_window）。"""

    def __init__(self):
        self.steps: list[int] = []
        self.losses: list[float] = []

    def add(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(float(loss))

    def summary(self, window: int) -> dict:
        if not self.losses:
            return {"n": 0}
        w = min(window, len(self.losses))
        return {
            "n": len(self.losses),
            "first_window_mean": sum(self.losses[:w]) / w,
            "last_window_mean": sum(self.losses[-w:]) / w,
            "min": min(self.losses),
            "last": self.losses[-1],
        }


def as_dict(meter: TpsMeter, history: LossHistory, window: int, extra: dict | None = None) -> dict:
    """summary.json 的 metrics 部分（唯一真相的骨架; 调用方再合并 sample/config）。"""
    out = {"throughput": meter.report(), "loss": history.summary(window)}
    if extra:
        out.update(extra)
    return out


if __name__ == "__main__":
    t = 0.0
    m = TpsMeter(clock=lambda: t)
    for _ in range(10):
        t += 0.5          # 每步 0.5s 训练
        m.add_train(1000, 0.5)
        t += 0.5          # 每步之间 0.5s 非训练（验证/采样）—— 只进 wall
    r = m.report()
    print(r)  # train_tps=2000, wall_tps=1000 —— 两个口径在这里分开
    print("rep4 全复读:", rep4([5] * 32), " 无重复:", rep4(list(range(32))))
