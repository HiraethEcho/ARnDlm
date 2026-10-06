"""scripts/make_wikitext_jsonl.py -- 下载 wikitext-103-raw-v1 并整理成 {"text": doc} jsonl。

文档切分: wikitext-103 的 " = 标题 = " 行起新文档; 空行丢弃; 文内行以 \\n 连接。
输出一行一个文档, 供 common/prepare-example.py tokenize。

用法: HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/make_wikitext_jsonl.py <out.jsonl>
"""
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

HEADING = re.compile(r"^ = [^=].* = $")  # 一级标题 " = 标题 = "（整行）, 节标题是 " = = X = = " 被排除

out_path = Path(sys.argv[1])
out_path.parent.mkdir(parents=True, exist_ok=True)

import glob
files = sorted(glob.glob(str(Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve() / "*.parquet")))
print("parquet:", files)
tables = [pq.read_table(f, columns=["text"]) for f in files]
rows = [t.column("text").to_pylist() for t in tables]
ds_rows = [r for chunk in rows for r in chunk]
print(f"rows: {len(ds_rows)}")

docs, cur = [], []
n_rows = 0
for text in ds_rows:
    n_rows += 1
    if HEADING.match(text.rstrip("\n")):
        if cur:
            docs.append("\n".join(cur).strip())
            cur = []
        cur.append(text.strip())
    elif text.strip():
        cur.append(text.strip())
if cur:
    docs.append("\n".join(cur).strip())
docs = [d for d in docs if d]
print(f"articles: {len(docs)} (from {n_rows} rows)")

with open(out_path, "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
print(f"-> {out_path}")
