"""test/_impl.py -- 骨架 / 参考实现 双加载开关（ROADMAP「对拍怎么做」）。

    IMPL=example   # 缺省：加载 `-example.py` 参考实现
    IMPL=x         # 加载新布局里的 `x.py` 骨架

用法：第二个参数永远写参考实现路径，开关负责换成骨架。

    from _impl import load, MODE
    cfg_mod = load("config_example", "common/config-example.py")

骨架还是空的（只有 docstring）时直接报错说清楚，不让 AttributeError 埋在调用栈里。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 参考实现路径 -> 骨架路径（两列都是仓库相对路径）
SKELETON_OF = {
    "common/a2d-example.py": "a2d/a2d.py",
    "common/config-example.py": "models/config.py",
    "common/data-example.py": "prepare/data.py",
    "common/device-example.py": "prepare/device.py",
    "common/metrics-example.py": "prepare/metrics.py",
    "common/prepare-example.py": "prepare/prepare.py",
    "common/seed-example.py": "prepare/seed.py",
    "common/sft-prepare-example.py": "prepare/sft-prepare.py",
    "common/tokenizer-example.py": "models/tokenizer.py",
    "src/eval/chat-example.py": "scripts/chat.py",
    "src/eval/repl-example.py": "scripts/repl.py",
    "src/models/native/model-example.py": "models/model.py",
    "src/models/trans-example.py": "models/model-trans.py",
    "src/training/a2d_dllm-example.py": "a2d/a2d_dllm.py",
    "src/training/a2d_mini-example.py": "a2d/a2d_mini.py",
    "src/training/ar_native-example.py": "trainer/ar_native.py",
    "src/training/ar_trans-example.py": "trainer/ar_trans.py",
    "src/training/dlm-example.py": "trainer/dlm.py",
    "src/training/dpo-example.py": "trainer/dpo.py",
    "src/training/main-example.py": "trainer/main.py",
    "src/training/sft-example.py": "trainer/sft.py",
    "src/training/state-example.py": "trainer/state.py",
}

MODE = os.environ.get("IMPL", "example").strip() or "example"
if MODE not in ("example", "x"):
    raise SystemExit(f"IMPL 只认 example（缺省）或 x，收到 {MODE!r}")


def resolve(example_path: str) -> str:
    """参考实现路径 -> 本次 IMPL 实际要加载的路径（仓库相对）。"""
    if MODE == "example":
        return example_path
    try:
        return SKELETON_OF[example_path]
    except KeyError:
        raise SystemExit(f"IMPL=x 没有登记 {example_path} 的骨架，补 test/_impl.py 的 SKELETON_OF") from None


def load(name: str, example_path: str):
    """按 IMPL 加载模块；name 是唯一模块名（避免同名冲突）。"""
    rel = resolve(example_path)
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"IMPL={MODE}: 要加载 {rel}，文件不存在")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not [n for n in vars(mod) if not n.startswith("__")]:
        raise SystemExit(f"IMPL={MODE}: {rel} 还是空骨架（只有 docstring），先填它再对拍")
    return mod
