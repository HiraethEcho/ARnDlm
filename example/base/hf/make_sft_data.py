#!/usr/bin/env python
"""Build the SFT prepared directory from a MiniMind `sft_t2t_*.jsonl`.

    .venv/bin/python example/base/hf/make_sft_data.py --max-blocks 4096

Writes `data/prepared/sft-mini-6400/`: tokens.bin, assistant_spans.npy,
supervised_tokens.npy, meta.json. The numbers in meta.json are the ones to read
before trusting the corpus -- `n_dropped_turns_with_thinking` is the one that
bit us (ISSUES.md OPEN-7).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config as C
from common.sft import write_prepared

DEFAULT_SRC = C.ROOT / "data" / "raw" / "minimind_dataset" / "sft_t2t_mini.jsonl"
DEFAULT_OUT = C.ROOT / "data" / "prepared" / "sft-mini-6400"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--block-size", type=int, default=C.BLOCK_SIZE)
    ap.add_argument("--max-blocks", type=int, default=4096,
                    help="Tier 0 only needs a slice; the corpus is 905k dialogues")
    ap.add_argument("--max-docs", type=int, default=0)
    a = ap.parse_args()

    if not a.source.exists():
        print(f"missing {a.source}", file=sys.stderr)
        return 1
    meta = write_prepared(a.source, a.out, a.block_size, a.max_blocks, a.max_docs)
    print(json.dumps(meta, indent=2))
    if meta["n_blocks"] == 0:
        print("no blocks produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())