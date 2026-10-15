"""test/05-metrics.py -- prepare/metrics.py 的验收（双口径分开 / rep4 两端 / 窗口均值）。

跑法: .venv/bin/python test/05-metrics.py
"""
import sys
from pathlib import Path

from _impl import load, MODE


mod = load("mx", "common/metrics-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


print("[1] TpsMeter 双口径: 非训练时间只进 wall")
t = 0.0
m = mod.TpsMeter(clock=lambda: t)
for _ in range(10):
    t += 0.5
    m.add_train(1000, 0.5)
    t += 0.5
r = m.report()
check("train_seconds = 5.0", r["train_seconds"] == 5.0, r)
check("wall_seconds = 10.0", r["wall_seconds"] == 10.0, r)
check("train_tps = 2000", abs(r["train_tps"] - 2000) < 1e-9, r)
check("wall_tps = 1000", abs(r["wall_tps"] - 1000) < 1e-9, r)
check("train_tokens = 10000", r["train_tokens"] == 10000)

print("[2] 零步 / 零秒不除零")
m0 = mod.TpsMeter(clock=lambda: 0.0)
r0 = m0.report()
check("空 report 不炸", r0["train_tps"] == 0.0 and r0["wall_tps"] == 0.0, r0)

print("[3] rep4 两端与中间")
check("无重复 = 0", mod.rep4(list(range(64))) == 0.0)
# 全复读: 61 个 4-gram 全同, unique=1 -> 1 - 1/61 (有限长度到不了 1.0)
check("全复读 = 1-1/61", abs(mod.rep4([7] * 64) - (1 - 1 / 61)) < 1e-12, mod.rep4([7] * 64))
# 周期 3 循环: 63 个 4-gram 只有 3 种 -> 1 - 3/63
check("周期 3 循环 = 1-3/63", abs(mod.rep4([1, 2, 3] * 22) - (1 - 3 / 63)) < 1e-12, mod.rep4([1, 2, 3] * 22))
# list(range(32))*2: 前 29 个 + 跨边界 3 个 = unique 32, total 61
half = list(range(32)) * 2
check("重复一遍 = 1-32/61", abs(mod.rep4(half) - (1 - 32 / 61)) < 1e-12, mod.rep4(half))
check("不足 4 token = 0", mod.rep4([1, 2, 3]) == 0.0)

print("[4] rep4_worst_window: 一段复读不被平均掩盖")
clean = list(range(64))
noisy = list(range(64)) + [9] * 64
check("64+64 中 64 token 全复读 -> worst=1-1/61", abs(mod.rep4_worst_window(noisy, 64) - (1 - 1 / 61)) < 1e-12)
check("干净序列 worst=0", mod.rep4_worst_window(clean, 64) == 0.0)

print("[5] LossHistory 首尾窗口")
h = mod.LossHistory()
for i in range(200):
    h.add(i, 10.0 - i / 20)  # 10.0 -> 0.05
s = h.summary(100)
check("first_window_mean = 7.525", abs(s["first_window_mean"] - 7.525) < 1e-9)
check("last_window_mean = 2.525", abs(s["last_window_mean"] - 2.525) < 1e-9)
check("min = 0.05", abs(s["min"] - 0.05) < 1e-9)
check("空 history", mod.LossHistory().summary(100) == {"n": 0})

print("[6] as_dict 组装")
out = mod.as_dict(m, h, 100, extra={"sample": {"texts": ["x"]}})
check("结构", "throughput" in out and "loss" in out and "sample" in out)
check("吞吐数字来自 meter", out["throughput"]["train_tps"] == 2000.0)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/05-metrics [IMPL={MODE}]")
