#!/usr/bin/env python
"""Tokenize JSONL documents into the project's binary training format.

Example:
    python scripts/prepare_data.py data/raw/tinystories.jsonl \
        --output data/prepared/tinystories-6400
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="JSONL or plain-text file")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    p.add_argument("--text-key", default="text")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument("--append-eos", action="store_true")
    p.add_argument("--eos-id", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_text(path: Path, key: str):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            text = row.get(key) if isinstance(row, dict) else None
            if isinstance(text, str):
                yield text


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty: {args.output}; use --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens: list[int] = []
    lengths: list[int] = []
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    for text in read_text(args.input, args.text_key):
        text = text.strip()
        if len(text) < args.min_chars:
            continue
        ids = tokenizer.encode(text).ids
        if args.append_eos:
            ids.append(args.eos_id)
        if not ids:
            continue
        tokens.extend(ids)
        lengths.append(len(ids))
        if args.max_docs and len(lengths) >= args.max_docs:
            break

    if not tokens:
        raise SystemExit("no documents produced")
    if max(tokens) > np.iinfo(np.uint16).max:
        raise SystemExit("token id exceeds uint16; use a tokenizer with vocab <= 65535")

    np.asarray(tokens, dtype=np.uint16).tofile(args.output / "tokens.bin")
    # npy format is used for compatibility with existing readers.
    arr = np.asarray(lengths, dtype=np.int32)
    np.save(args.output / "doc_lengths.npy", arr)
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "n_docs": len(lengths),
        "total_tokens": len(tokens),
        "source_file": str(args.input),
        "source_sha256": source_sha,
        "tokenizer": str(args.tokenizer),
        "append_eos": args.append_eos,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(lengths)} documents, {len(tokens)} tokens -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
