#!/usr/bin/env python
"""test 02 — 数据管线读对了吗。

数据是预 tokenize 好的二进制, 不用模型, 也不用 torch 就能验:
  1. tokens.bin / doc_lengths.npy / meta.json 三个文件自洽
  2. packing 拼接后没有跨文档污染 (每块能追溯回原始文档)
  3. 切出来的块拿去 decode, 首尾 token 是不是 BOS / EOS

需要 data/ 软链接 (./scripts/setup_symlinks.sh)。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "data" / "prepared" / "minimind3-6400"
TOKENIZER = ROOT / "src" / "base" / "tokenizer.json"

BOS, EOS = 1, 2


def main():
    try:
        import numpy as np
    except ImportError:
        sys.exit("FAIL 没装 numpy")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("FAIL 没装 tokenizers")

    for f in ("tokens.bin", "doc_lengths.npy", "meta.json"):
        if not (PREPARED / f).exists():
            sys.exit(f"FAIL 找不到 {PREPARED / f}  (先跑 ./scripts/setup_symlinks.sh)")

    ok = True
    meta = json.loads((PREPARED / "meta.json").read_text())

    # --- memmap 而不是 read(): 665 MB 全读进内存没必要, 训练时也只会顺序扫 ---
    # dtype="<u2" 是显式小端 uint16。不写 "<" 的话, 换机器 (大端) 会静默读错。
    tokens = np.memmap(PREPARED / "tokens.bin", dtype="<u2", mode="r")
    lengths = np.load(PREPARED / "doc_lengths.npy")

    print(f"tokens : {tokens.shape[0]:,} ({tokens.nbytes / 1e6:.0f} MB, {tokens.dtype})")
    print(f"docs   : {len(lengths):,}  mean {lengths.mean():.1f}  max {lengths.max()}")
    print(f"meta   : vocab_size={meta['vocab_size']} n_docs={meta['n_docs']} "
          f"val_frac={meta.get('val_frac')}")

    # --- 三个文件必须互相对得上 ---
    if len(lengths) != meta["n_docs"]:
        print(f"FAIL  doc_lengths 有 {len(lengths)} 行, meta 说 {meta['n_docs']}")
        ok = False
    if int(lengths.sum()) != tokens.shape[0]:
        # 这是最常见的 prepare 脚本 bug: 写了 meta 但 tokens.bin 是旧的
        print(f"FAIL  lengths.sum()={int(lengths.sum()):,} != tokens {tokens.shape[0]:,}")
        ok = False
    if meta.get("total_tokens") != tokens.shape[0]:
        print(f"FAIL  meta.total_tokens={meta.get('total_tokens'):,} "
              f"!= tokens {tokens.shape[0]:,}")
        ok = False
    if meta["vocab_size"] != 6400:
        print(f"FAIL  meta vocab_size={meta['vocab_size']}, 期望 6400")
        ok = False

    if tokens.max() >= meta["vocab_size"]:
        print(f"FAIL  出现 id {int(tokens.max())} >= vocab {meta['vocab_size']}")
        ok = False

    # --- packing: 把多个文档拼成定长块 ---
    # 本机 packing 比 padding 快 1.71x (见 logs/dev/02-hardware-perf.md),
    # 前提是拼接正确。拼接方式: 每篇文档前后加 BOS/EOS, 然后首尾相接,
    # 不足一整块的部分用 PAD(0) 补。
    T = 512
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    print(f"\n--- packing T={T} ---")

    tok = Tokenizer.from_file(str(TOKENIZER))

    # 随机抽 3 篇短文档, 手工拼一块, 验证首尾 token 和可追溯性
    short = np.argsort(lengths)[:200]           # 取最短的 200 篇, 保证能塞进一块
    block, spans = [], []
    for i in short:
        if sum(len(b) for b in block) + lengths[i] + 2 > T:
            break
        doc = tokens[offsets[i]:offsets[i + 1]].astype(np.int32)
        start = sum(len(b) for b in block)
        block.append([BOS])
        block.append(doc.tolist())
        block.append([EOS])
        spans.append((i, start, start + 1 + len(doc) + 1))

    flat = [t for part in block for t in part]
    flat += [0] * (T - len(flat))                # PAD 补齐

    if len(flat) != T:
        print(f"FAIL  拼出来 {len(flat)} != {T}")
        ok = False

    # 每篇文档的正文首尾必须是 BOS / EOS —— 这就是「没有跨文档污染」的证据
    for doc_id, s, e in spans:
        if flat[s] != BOS:
            print(f"FAIL  doc {doc_id} 开头是 {flat[s]}, 期望 BOS {BOS}")
            ok = False
            break
        if flat[e - 1] != EOS:
            print(f"FAIL  doc {doc_id} 结尾是 {flat[e - 1]}, 期望 EOS {EOS}")
            ok = False
            break
    else:
        print(f"  拼了 {len(spans)} 篇, 每篇首尾都是 BOS/EOS — 无跨文档污染")

    # PAD 区必须全是 0, 训练时靠它算 attention mask
    pad_region = flat[sum(len(b) for b in block):]
    if any(t != 0 for t in pad_region):
        print(f"FAIL  PAD 区非零: {pad_region[:10]}")
        ok = False
    else:
        print(f"  PAD 区 {len(pad_region)} token 全为 0")

    # decode 一篇**有代表性的**文档。
    # 别用上面那堆最短文档来 decode —— 它们是 4 token 的碎片
    # ('CCertainly' / '200Hello'), 看着像数据坏了, 其实是语料里真的存在
    # 这么短的文档。拿它们当样本会误判。
    rng = np.random.default_rng(0)
    pick = int(rng.integers(len(lengths)))
    doc = tokens[offsets[pick]:offsets[pick + 1]].astype(np.int32).tolist()
    print(f"\n--- 随机 doc {pick} (len={len(doc)}) ---")
    print(tok.decode(doc)[:300])

    med = int(np.argsort(lengths)[len(lengths) // 2])
    print(f"\n--- 中位长度 doc {med} (len={int(lengths[med])}) ---")
    print(tok.decode(tokens[offsets[med]:offsets[med + 1]].astype(np.int32).tolist())[:300])

    # 语料里最短的那批文档是退化碎片。这不是 bug, 但**做数据过滤时要注意**:
    # 按长度过滤会优先留下它们, 而不是留下好样本。
    n_tiny = int((lengths < 16).sum())
    tiny = [tok.decode(tokens[offsets[i]:offsets[i + 1]].astype(np.int32).tolist())
            for i in np.argsort(lengths)[:3]]
    print("\n--- 注意 ---")
    print(f"len < 16 的文档: {n_tiny:,} ({100 * n_tiny / len(lengths):.2f}%)")
    print(f"最短 3 篇: {tiny}")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
