"""test/08-prepare.py -- prepare/prepare.py 的验收（落盘格式与 data-example 对接 + 大声跳过）。

跑法: .venv/bin/python test/08-prepare.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path


import hashlib
import numpy as np

from _impl import load, MODE

ROOT = Path(__file__).resolve().parent.parent


prep = load("prepare_example", "common/prepare-example.py")
data = load("data_example", "common/data-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


tmp = Path(tempfile.mkdtemp(prefix="prep-"))
try:
    print("[1] 构造小 jsonl: 正常 zh 行 / conversations 行(无 text) / 空行 / MASK 行")
    rows = [
        {"text": "今天天气很好，我们去公园散步。"},
        {"conversations": [{"from": "human", "value": "你好"}]},   # 无 text —— 已知坑
        {"text": "   "},                                           # 空白
        {"text": "这段话里有 <|buffer1|> 预留 token。"},              # 会产出 MASK 27
        {"text": "第二条正常文本，测试往返一致性。"},
    ]
    src = tmp / "in.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    out = tmp / "out"
    meta = prep.prepare(src, out, ROOT / "common/tokenizer.json", vocab=6400)

    check("kept = 2", meta["n_docs"] == 2, meta)
    check("no_text 跳过 1", meta["skipped"]["no_text"] == 1, meta["skipped"])
    check("empty 跳过 1", meta["skipped"]["empty"] == 1, meta["skipped"])
    check("mask 行跳过 1", meta["skipped"]["mask_token"] == 1, meta["skipped"])
    check("total_tokens = sum(doc_lengths)",
          meta["total_tokens"] == int(np.load(out / "doc_lengths.npy").sum()))
    check("sha256 与源文件哈希一致",
      meta["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest())

    print("[2] 产物能被 data-example 直接加载并四道自检通过")
    c = data.Corpus(out, block_size=16, vocab=6400)
    check("n_blocks > 0", c.n_blocks > 0, c.n_blocks)
    blk = c.block(0)
    check("BOS 开头", int(blk[0]) == 1)
    check("无 MASK id", 27 not in blk.tolist())

    print("[3] decode 往返（data 层包裹后去 special 能还原原文）")
    tk = load("tok_example", "common/tokenizer-example.py").Tok()
    ids = c._gather(np.arange(int(c.doc_lengths[0]) + 2, dtype=np.int64))
    check("文档 0 包裹往返", tk.decode(ids.tolist()).strip() == "今天天气很好，我们去公园散步。",
          tk.decode(ids.tolist()))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/08-prepare [IMPL={MODE}]")
