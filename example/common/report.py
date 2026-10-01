"""Append one row per cell to example/RESULTS.md.

Kept in code rather than typed by hand because ten hand-copied rows is ten
chances to report train-only throughput where wall-clock was measured, or to
skip the repetition rate. `rep-4` is a required argument, not an optional
extra -- notes/03-minimind-dlm.md §11 is the case this project exists to
avoid: loss falling while the output loops.
"""
from __future__ import annotations

from pathlib import Path

from .config import ROOT

RESULTS = ROOT / "example" / "RESULTS.md"

COLUMNS = ("cell", "base", "method", "params", "steps", "loss first->last",
           "train-only tok/s", "wall tok/s", "eval loss", "rep-4", "notes")


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # Two decimals on a throughput is noise; one on a loss is the point.
        return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.4f}"
    return str(value)


def append_result(cell: str, base: str, method: str, params: int, steps: int,
                  loss: dict | None = None, train_tps: float | None = None,
                  wall_tps: float | None = None, eval_loss: float | None = None,
                  rep4: float | None = None, notes: str = "") -> Path:
    """Upsert one row per cell.

    Replace rather than append: a rerun of the same cell (a quick smoke, a
    resumed run) must not leave two rows with different step counts and no way
    to tell which one RESULTS.md is quoting.
    """
    loss = loss or {}
    row = [
        cell, base, method, _fmt(params), _fmt(steps),
        f"{_fmt(loss.get('first'))} -> {_fmt(loss.get('last'))}",
        _fmt(train_tps), _fmt(wall_tps), _fmt(eval_loss), _fmt(rep4), notes,
    ]
    line = "| " + " | ".join(row) + " |"
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    if not RESULTS.exists():
        RESULTS.write_text(
            "# RESULTS — A 版 smoke 实测\n\n"
            "由 `example/common/report.py` 自动追加，不要手改。\n\n"
            "每格一行，重跑会覆盖旧行。Tier 0 · 只验 pipeline，不看模型能力。\n\n"
            "| " + " | ".join(COLUMNS) + " |\n"
            + "|" + "|".join(["---"] * len(COLUMNS)) + "|\n"
        )

    lines = RESULTS.read_text().splitlines()
    prefix = f"| {cell} |"
    for i, existing in enumerate(lines):
        if existing.startswith(prefix):
            lines[i] = line
            break
    else:
        lines.append(line)
    RESULTS.write_text("\n".join(lines) + "\n")
    return RESULTS