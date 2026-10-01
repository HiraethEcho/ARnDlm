#!/usr/bin/env python
"""Stream a bounded text sample from Hugging Face into data/raw/.

Examples:
    python scripts/download_data.py tinystories --max-docs 10000
    python scripts/download_data.py openwebtext --max-docs 10000
    python scripts/download_data.py wikipedia-zh --max-docs 10000

The script never downloads a full corpus by default. data/ is the repository
symlink to /mnt/extra/data, so raw files stay outside the Git worktree.

Two backends:
  datasets -- `datasets.load_dataset(streaming=True)`; the documented path.
  parquet  -- hf_hub_download + pyarrow, no `datasets` import at all.

`datasets` 4.2.0 is unusable in this .venv: it calls the dill Pickler with the
pre-0.3.9 `_batch_setitems(obj.items(), obj)` signature, which dill 0.4.0 on
Python 3.14 rejects with "takes 2 positional arguments but 3 were given".
Pinning dill back would fight the locked versions, so `auto` falls back to the
parquet backend instead. The parquet backend reads the same columns and caps
itself at --max-parquet-files shards, since one shard already holds far more
documents than --max-docs.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASETS: dict[str, dict[str, Any]] = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text": "text",
        "output": "tinystories.jsonl",
        "parquet_pattern": "data/train-*.parquet",
    },
    "openwebtext": {
        "path": "openwebtext",
        "config": None,
        "split": "train",
        "text": "text",
        "output": "openwebtext.jsonl",
        "parquet_pattern": "plain_text/train-*.parquet",
    },
    "wikipedia-zh": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "text": "text",
        "output": "wikipedia-zh.jsonl",
        "parquet_pattern": "20231101.zh/train-*.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=sorted(DATASETS))
    p.add_argument("--max-docs", type=int, default=10000)
    p.add_argument("--max-chars", type=int, default=0, help="Stop after this many characters; 0 means unlimited.")
    p.add_argument("--output", type=Path)
    p.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache directory.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backend", choices=("auto", "datasets", "parquet"), default="auto")
    p.add_argument("--max-parquet-files", type=int, default=1, help="Shards to read in the parquet backend.")
    return p.parse_args()


def stream_parquet(spec: dict[str, Any], max_files: int) -> Iterator[str]:
    """Yield the text column of the first matching parquet shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(
        s.rfilename
        for s in HfApi().repo_info(spec["path"], repo_type="dataset").siblings
        if fnmatch.fnmatch(s.rfilename, spec["parquet_pattern"])
    )
    if not names:
        raise SystemExit(f"no file matches {spec['parquet_pattern']} in {spec['path']}")
    for name in names[:max_files]:
        local = hf_hub_download(spec["path"], name, repo_type="dataset")
        print(f"reading {name}", flush=True)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1024, columns=[spec["text"]]):
            yield from batch.column(0).to_pylist()


def stream_rows(spec: dict[str, Any], backend: str, max_files: int = 1) -> Iterable[dict[str, Any]]:
    if backend == "parquet":
        return ({"text": t} for t in stream_parquet(spec, max_files))
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": spec["path"],
        "split": spec["split"],
        "streaming": True,
    }
    if spec["config"]:
        kwargs["name"] = spec["config"]
    return load_dataset(**kwargs)


def main() -> int:
    args = parse_args()
    spec = DATASETS[args.dataset]
    if args.max_docs <= 0:
        raise SystemExit("--max-docs must be positive")

    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir)

    backend = args.backend
    if backend == "auto":
        try:
            stream = stream_rows(spec, "datasets")
            next(iter(stream), None)
        except Exception as exc:  # noqa: BLE001 - any import/runtime failure falls back
            print(f"datasets backend failed ({type(exc).__name__}: {exc}); using parquet")
            backend = "parquet"
    stream = stream_rows(spec, backend, args.max_parquet_files)

    output = args.output or Path("data/raw") / spec["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")

    docs = chars = 0
    with output.open("w", encoding="utf-8") as f:
        for row in stream:
            text = row.get(spec["text"])
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if args.max_chars and chars + len(text) > args.max_chars:
                break
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            docs += 1
            chars += len(text)
            if docs >= args.max_docs:
                break

    print(f"wrote {docs} documents, {chars} characters -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
