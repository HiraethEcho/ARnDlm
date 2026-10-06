"""test/15-sft-prepare.py -- sft-prepare-example 的验收: 产物被 SFTBlocks 直接吃 + span 语义。

跑法: .venv/bin/python test/15-sft-prepare.py
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = load("p", "common/sft-prepare-example.py")
sftx = load("s", "src/training/sft-example.py")
tok = load("t", "common/tokenizer-example.py").Tok()

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


tmp = Path(tempfile.mkdtemp(prefix="sftp-"))
try:
    print("[1] 造 3 条对话 -> prepare 128 ctx / 上限 8 块")
    rows = [
        {"conversations": [
            {"role": "user", "content": "什么是深度学习？"},
            {"role": "assistant", "content": "深度学习是机器学习的一个分支，用多层神经网络学习表示。"},
            {"role": "user", "content": "和传统方法有什么不同？"},
            {"role": "assistant", "content": "传统方法依赖手工特征，深度学习端到端地学特征。"},
        ]},
        {"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
        ]},
        {"conversations": [
            {"role": "user", "content": "这个问题太长太长太长" * 30},   # 单轮超整块 -> 丢
            {"role": "assistant", "content": "好"},
        ]},
    ]
    src = tmp / "in.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp / "prepared"
    meta = prep.write_prepared(src, out, block_size=128, max_blocks=8)
    check("n_blocks = 3（第三条超长被丢）", meta["n_blocks"] == 3, meta["n_blocks"])
    check("n_long_turns = 1", meta["n_long_turns"] == 1, meta)
    check("supervised_tokens = sum(spans)",
          meta["supervised_tokens"] == int((np.load(out / "assistant_spans.npy")[:, 2]
                                            - np.load(out / "assistant_spans.npy")[:, 1]).sum()))

    print("[2] SFTBlocks 直接装载（格式互认）+ 四道对账")
    sft = sftx.SFTBlocks(out)
    check("硬对账通过（没抛错）", sft.n_blocks == 3)

    print("[3] span 语义: 区间文本 = 全文中 assistant 段（T4a 语义）")
    spans = np.load(out / "assistant_spans.npy")
    hits = 0
    for b, s_, e in spans:
        block_ids = np.asarray(sft.blocks[int(b) * sft.bs:(int(b) + 1) * sft.bs], dtype=np.int64)
        full = tok.decode(block_ids.tolist())
        span_text = tok.decode(block_ids[s_:e].tolist())
        segs = full.split("assistant\n")[1:]  # 任一 assistant 段（多轮块有多段）
        if span_text and any(span_text in a for a in segs):
            hits += 1
    check("全部 span 区间文本 ⊆ assistant 段", hits == len(spans), f"{hits}/{len(spans)}")

    print("[4] 加噪约束（T4f）: corrupt 的 masked ⊆ span")
    import torch
    import importlib.util as iu
    spec = iu.spec_from_file_location("a2dx", ROOT / "common/a2d-example.py")
    a2d = importlib.util.module_from_spec(spec); spec.loader.exec_module(a2d)
    viol = 0
    for b in range(sft.n_blocks):
        labels = sft.labels(int(b))
        ids = torch.from_numpy(np.asarray(
            sft.blocks[int(b) * sft.bs:(int(b) + 1) * sft.bs], dtype=np.int64))
        t = torch.full((1,), 0.9)
        _, masked = a2d.corrupt(ids[None], t, valid=(torch.from_numpy(labels) != -100)[None])
        if bool((masked[0] & (torch.from_numpy(labels) == -100)).any()):
            viol += 1
    check("加噪 ⊆ assistant 区间", viol == 0, viol)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/15-sft-prepare")
