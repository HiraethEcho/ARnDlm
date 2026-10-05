"""test/07-c4-resume.py -- SPEC C4: 连续 30 步 vs 20 + resume + 10, 逐步差 <= 1e-6（native 侧）。

跑法: .venv/bin/python test/07-c4-resume.py
（trans 侧的 Trainer resume 是 03-call-flow 留下的开放点, 不在此测。）
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("config_example", "common/config-example.py")
main_mod = load("main_example", "src/training/main-example.py")
native = load("ar_native_example", "src/training/ar_native-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


cfg = cfg_mod.load("smoketest")
cfg.TRAIN.steps  # noqa -- 未用; 批次顺序由 seed 推回
data_dir = ROOT / "data/prepared/tinystories-6400"

tmp = Path(tempfile.mkdtemp(prefix="c4-"))
STEPS_A, STEPS_B = 30, 20


def fresh_ctx(tag: str):
    run_dir = tmp / f"run-{tag}"
    run_dir.mkdir(parents=True)
    return main_mod.Ctx(cfg, f"ar-3.2m-native-smoke-{tag}", data_dir, run_dir,
                        run_dir / "ckpt")


try:
    print("[1] 连续 30 步")
    ctxA = fresh_ctx("A")
    A = native.run(cfg, STEPS_A, ctxA)
    lossesA = list(A["history"].losses)
    weightsA = {k: v.detach().clone() for k, v in A["model"].state_dict().items()}

    print("[2] 20 步 + resume + 10 步")
    ctxB1 = fresh_ctx("B1")
    B = native.run(cfg, STEPS_B, ctxB1)
    lossesB1 = list(B["history"].losses)
    state = load("state_example_test", "src/training/state-example.py")
    state.save_state(ctxB1.run_dir / "train_state.pt", step=B["step"],
                     optimizer=B["optimizer"], data_position=B["data_position"])

    # 复制 B1 的 checkpoint 作为 resume 源（模拟真实续训: 用存盘的, 不用内存对象）
    ckpt = tmp / "ckpt-resume"
    shutil.copytree(ctxB1.ckpt_dir, ckpt)
    shutil.copy(ctxB1.run_dir / "train_state.pt", ckpt / "train_state.pt")

    ctxB2 = fresh_ctx("B2")
    B2 = native.run(cfg, STEPS_A, ctxB2, resume_path=ckpt)
    lossesB = lossesB1 + list(B2["history"].losses)

    print("[3] 逐步 loss 等价 (C4 <= 1e-6)")
    check("步数一致", len(lossesA) == len(lossesB) == STEPS_A, (len(lossesA), len(lossesB)))
    check("前 20 步逐位一致", all(a == b for a, b in zip(lossesA[:STEPS_B], lossesB[:STEPS_B])))
    max_diff = max(abs(a - b) for a, b in zip(lossesA[STEPS_B:], lossesB[STEPS_B:]))
    check(f"resume 段 max|Δloss| = {max_diff:.3e} <= 1e-6", max_diff <= 1e-6, max_diff)

    print("[4] 终点权重逐位一致")
    weightsB = B2["model"].state_dict()
    wdiff = max((weightsA[k] - weightsB[k]).abs().max().item() for k in weightsA)
    check(f"权重 max|Δ| = {wdiff:.3e} == 0", wdiff == 0.0, wdiff)

    print("[5] data_position 自洽")
    check("position(29) 一致", A["data_position"] == B2["data_position"],
          (A["data_position"], B2["data_position"]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/07-c4-resume  (C4 max|Δloss| = {max_diff:.1e}, 权重差 {wdiff:.1e})")
