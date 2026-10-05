"""test/02-seed-device.py -- seed-example / device-example 的验收（PLAN P1: 同 seed 三路一致）。

跑法: .venv/bin/python test/02-seed-device.py
"""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seed_mod = load("seed_example", "common/seed-example.py")
dev_mod = load("device_example", "common/device-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


print("[1] 同 seed 三路同序列")
def draw():
    return (random.random(), np.random.rand(3), torch.randn(3))


seed_mod.seed_all(42)
a = draw()
seed_mod.seed_all(42)
b = draw()
check("python random 一致", a[0] == b[0])
check("numpy 一致", np.array_equal(a[1], b[1]))
check("torch 一致", torch.equal(a[2], b[2]))

print("[2] snapshot/restore 逐位恢复")
seed_mod.seed_all(7)
first = draw()
snap = seed_mod.snapshot()
second = draw()  # 消耗随机数
seed_mod.restore(snap)
third = draw()
check("torch 恢复", torch.equal(second[2], third[2]))
check("numpy 恢复", np.array_equal(second[1], third[1]))
check("python 恢复", second[0] == third[0])
check("快照包含 cuda 状态", "torch_cuda" in snap)

print("[3] restore 后 continue 与一次性跑完等价")
seed_mod.seed_all(7)
first2 = draw()
mid = draw()
snap2 = seed_mod.snapshot()
tail = [torch.randn(2) for _ in range(3)]
seed_mod.seed_all(7)
draw(); draw()
seed_mod.restore(snap2)
tail2 = [torch.randn(2) for _ in range(3)]
check("后续序列逐位一致", all(torch.equal(x, y) for x, y in zip(tail, tail2)))

print("[4] device: cuda 存在时返回 cuda")
dev = dev_mod.pick_device()
check("device 是 cuda", dev.type == "cuda", repr(dev))
check("本机 GPU", "Radeon" in torch.cuda.get_device_name(0) or "AMD" in torch.cuda.get_device_name(0),
      torch.cuda.get_device_name(0))

print("[5] dtype 转换与 autocast")
check("bfloat16", dev_mod.to_dtype("bfloat16") is torch.bfloat16)
check("float32", dev_mod.to_dtype("float32") is torch.float32)
try:
    dev_mod.to_dtype("fp16")
    check("未知 dtype 报错", False)
except ValueError:
    check("未知 dtype 报错", True)

x = torch.randn(8, 8, device=dev)
with dev_mod.autocast("bfloat16"):
    y = x @ x
check("bf16 autocast 生效", y.dtype == torch.bfloat16, repr(y.dtype))
with dev_mod.autocast("float32"):
    z = x @ x
check("fp32 eval 不进 autocast", z.dtype == torch.float32, repr(z.dtype))

print("[6] 没有 cuda 的机器要报错而不是退 CPU（用假环境测不了，只验证报错路径存在）")
import inspect
src = inspect.getsource(dev_mod.pick_device)
check("pick_device 无静默回退", "RuntimeError" in src)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/02-seed-device")
