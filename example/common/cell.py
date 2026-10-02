"""The run-one-cell ritual, shared by all ten cells.

Each cell differs in exactly three things: how the model is built, what the
loss is, and how text is sampled. Everything after that -- evaluate on frozen
batches, sample, measure repetition, save the checkpoint, write one RESULTS row
-- is identical, and duplicating it ten times is how a cell ends up reporting
train-only throughput where wall-clock was measured, or skipping repetition
rate because that argument was optional in its own file.
"""
from __future__ import annotations

from pathlib import Path

from .data import PackedData
from .loop import Loop
from .metrics import repetition_rates
from .report import append_result


def repetition_from_sample(sample: dict | None) -> tuple[float | None, list[dict]]:
    """rep-4 over generated ids. Returns (rep4, per-prompt rows)."""
    if not sample or not sample.get("ids"):
        return None, []
    rates = repetition_rates(sample["ids"], ns=(4,))
    rows = []
    for prompt, ids, text in zip(sample.get("prompts", []), sample["ids"],
                                 sample.get("texts", [])):
        rows.append({
            "prompt": prompt,
            "ids": [int(i) for i in ids],
            "text": text,
            "rep4": round(repetition_rates([ids], ns=(4,))[4], 4),
        })
    return rates[4], rows


def finish_cell(loop: Loop, data: PackedData, *, cell: str, base: str, method: str,
                ckpt_dir: Path, run_dir: Path, config: dict, sample_fn=None,
                eval_batches=None, notes: str = "") -> dict:
    """Train, score, sample, checkpoint, report. Returns the run summary."""
    summary = loop.fit(data, sample_fn=sample_fn, eval_batches=eval_batches)

    rep4, sample_rows = repetition_from_sample(summary.get("sample"))
    summary["repetition"] = {"rep4": rep4, "prompts": sample_rows}

    meta = {
        "cell": cell, "base": base, "method": method,
        "params": _params(loop),
        "loss": summary["loss"], "throughput": summary["throughput"],
        "eval_loss": summary.get("eval_loss"), "repetition": summary["repetition"],
    }
    ckpt_dir = loop.save(ckpt_dir, config, meta=meta)
    summary["ckpt"] = str(ckpt_dir)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples.txt").write_text(
        "\n\n".join(f"--- {r['prompt']}\n{r['text']}" for r in sample_rows))
    # Rewrite summary.json: fit() wrote it before repetition and the checkpoint
    # path existed, and logs/run is the only record of a run.
    import json

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    append_result(cell, base, method, meta["params"], summary["steps"],
                  loss=summary["loss"],
                  train_tps=summary["throughput"]["train_only_tps"],
                  wall_tps=summary["throughput"]["wall_tps"],
                  eval_loss=summary.get("eval_loss"), rep4=rep4, notes=notes)

    print(f"\n[{cell}] done  ckpt -> {ckpt_dir}")
    first, last = summary["loss"].get("first"), summary["loss"].get("last")
    loss_str = (f"{first:.4f} -> {last:.4f}" if first is not None else "no steps")
    eval_str = (f"  eval {summary['eval_loss']:.4f}"
                if summary.get("eval_loss") is not None else "")
    print(f"  loss   {loss_str}{eval_str}")
    print(f"  tok/s  train-only {summary['throughput']['train_only_tps']:,.0f}"
          f"  wall {summary['throughput']['wall_tps']:,.0f}")
    print(f"  peak   {summary['peak_gb']} GB   rep-4 "
          + ("n/a" if rep4 is None else f"{rep4:.4f}"))
    return summary


def _params(loop: Loop) -> int:
    from .ckpt import describe

    return describe(loop.model)["params_unique"]