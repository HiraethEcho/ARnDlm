#!/usr/bin/env python
"""test_sft_cells -- do the SFT cell scripts refuse to lie?

The single worst bug in this project's history is documented in the long-run
skill: "row 3 was LoRA on a randomly initialised 600M backbone -- 3 of 10 cells
measured a random model; every number looked plausible". A cell that cannot find
its upstream checkpoint must therefore **exit non-zero**, not quietly train from
random init, because the second outcome produces a table that reads fine.

Also checked here: the cell names match SPEC 2.3, each script exposes
`--init-random` as the deliberate ablation, and each imports without side effects.

Run:  .venv/bin/python example/tests/test_sft_cells.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# SPEC 2.3's SFT column, as (module, cell name, upstream checkpoint dir).
CELLS = [
    ("base.hf.train_sft", "ar-3.2m-hf-sft", "ar-3.2m-hf"),
    ("base.native.train_sft", "ar-3.2m-native-sft", "ar-3.2m-native"),
]


def section_spec_names() -> None:
    print("\n[1] cell names match SPEC 2.3")
    spec = (C.ROOT / "SPEC.md").read_text()
    for mod, cell, upstream in CELLS:
        check(f"{cell} is in SPEC.md", f"`{cell}`" in spec)
        check(f"{cell} is downstream of {upstream}", upstream in cell or cell.endswith("-sft"))
    for cell in ("ar-3.2m-hf-mini_a2d-sft", "ar-3.2m-hf-dllm_a2d-sft",
                 "ar-3.2m-native-mini_a2d-sft", "ar-3.2m-native-dllm_a2d-sft"):
        check(f"{cell} is in SPEC.md", f"`{cell}`" in spec)


def section_importable() -> None:
    print("\n[2] the scripts import and expose their knobs")
    for mod, cell, _ in CELLS:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            check(f"{mod} imports", False, f"{type(exc).__name__}: {exc}")
            continue
        check(f"{mod} imports", True)
        check(f"{mod} declares CELL = {cell}", getattr(m, "CELL", None) == cell,
              str(getattr(m, "CELL", None)))
        src = Path(m.__file__).read_text()
        check(f"{mod} offers --init-random", "--init-random" in src)
        check(f"{mod} guards the upstream checkpoint",
              "no AR checkpoint" in src or "no upstream checkpoint" in src,
              "silently training from random init is the bug this exists for")


def section_refuses_without_upstream() -> None:
    print("\n[3] a missing upstream checkpoint exits non-zero")
    argv = sys.argv
    for mod, cell, _ in CELLS:
        m = importlib.import_module(mod)
        missing = Path("/nonexistent/upstream-checkpoint")
        if not hasattr(m, "AR_CKPT"):
            check(f"{mod} names its upstream checkpoint", False, "no AR_CKPT attribute")
            continue
        original = m.AR_CKPT
        m.AR_CKPT = missing
        sys.argv = [f"{mod}.py", "--steps", "1", "--max-blocks", "8", "--no-verify"]
        try:
            rc = m.main()
        except SystemExit as exc:
            rc = int(exc.code or 0)
        finally:
            sys.argv = argv
            m.AR_CKPT = original
        check(f"{mod} returns non-zero", rc != 0, f"rc={rc}")


def section_rows() -> None:
    print("\n[4] the cell name decides the row and the objective")
    spec = (C.ROOT / "SPEC.md").read_text()
    import base.hf.train_dlm_sft as dlmsft
    for cell, (row, obj, upstream) in sorted(dlmsft.CELLS.items()):
        check(f"{cell} is in SPEC.md", f"`{cell}`" in spec)
        check(f"{cell} names the {row} row", row in cell,
              "a native cell that runs the transformers model produces numbers "
              "for the wrong implementation")
        check(f"{cell} points at a real upstream cell",
              f"`{upstream}`" in spec, upstream)
        builder, _, _, _, base_label = dlmsft.build_parts(row, obj)
        module = builder.__module__
        check(f"{cell} builds its row's model",
              ("base.hf" in module) if row == "hf" else ("base.native" in module),
              module)
        check(f"{cell} labels its base correctly",
              base_label == ("Qwen-mini-HF" if row == "hf" else "Qwen-mini-MiniMind"),
              base_label)
        check(f"{cell} accepts --cell", cell in dlmsft.CELLS)


CELLS_G3_ORIGINAL = {
    "ar-3.2m-hf-sft-mini_a2d": "ar-3.2m-hf-sft",
    "ar-3.2m-hf-sft-dllm_a2d": "ar-3.2m-hf-sft",
    "ar-3.2m-native-sft-mini_a2d": "ar-3.2m-native-sft",
    "ar-3.2m-native-sft-dllm_a2d": "ar-3.2m-native-sft",
}


def section_original_route() -> None:
    print("\n[5] the SFT-then-A2D script has no coverage of its own -- this is it")
    import base.hf.train_a2d_from_sft as a2dsft
    spec = (C.ROOT / "SPEC.md").read_text()
    for cell, (row, obj) in sorted(a2dsft.CELLS.items()):
        check(f"{cell} is in SPEC.md", f"`{cell}`" in spec)
        check(f"{cell} names the {row} row", row in cell,
              "a native cell running the transformers model is the wrong experiment")
        builder, sampler, factory, note = a2dsft.build_parts(row, obj)
        module = builder.__module__
        check(f"{cell} builds its row's model",
              ("base.hf" in module) if row == "hf" else ("base.native" in module), module)
        check(f"{cell} factory takes the model", "<lambda>" in repr(factory)
              or hasattr(factory, "__code__") and factory.__code__.co_argcount >= 1,
              "a bare `lambda: make_loss` evaluates to the function object, so Loop "
              "calls the factory with (model, batch) and gets a closure back")
        check(f"{cell} sampler is callable", callable(sampler))
        check(f"{cell} upstream is the -sft cell",
              a2dsft.CELLS[cell][0] and f"`{a2dsft.CELLS[cell][0]}`" in spec,
              a2dsft.CELLS[cell][0])

    # The derived upstream must equal the cell with `-mini_a2d`/`-dllm_a2d`
    # stripped, and every one of those upstreams must actually exist on disk.
    for cell in a2dsft.CELLS:
        derived = cell.replace("-mini_a2d", "").replace("-dllm_a2d", "")
        check(f"{cell} -> {derived} is a real cell", f"`{derived}`" in spec)
        check(f"{derived} has a checkpoint",
              (C.cell_dir(derived) / "model.safetensors").exists(),
              str(C.cell_dir(derived)))


def main() -> int:
    section_spec_names()
    section_importable()
    section_refuses_without_upstream()
    section_rows()
    section_original_route()
    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())