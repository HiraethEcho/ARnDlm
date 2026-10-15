"""test/04-data.py -- prepare/data.py 的验收（四道自检 / 包裹正确 / 批次纯函数 / epoch 重排）。

跑法: .venv/bin/python test/04-data.py
"""
import json
import sys
import tempfile
from pathlib import Path


import numpy as np

from _impl import load, MODE

mod = load("data_example", "common/data-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


print("[1] 真实语料: 四道自检通过, n_blocks 正确")
c = mod.Corpus("data/prepared/tinystories-6400", block_size=128, vocab=6400, max_blocks=4096)
stream_len = int(c.doc_lengths.sum()) + 2 * len(c.doc_lengths)
check("n_blocks = stream//bs 上限 4096", c.n_blocks == min(stream_len // 128, 4096),
      f"{c.n_blocks} vs {min(stream_len // 128, 4096)}")
check("自检 4 条全过(没抛错)", True)

print("[2] 包裹正确: 与朴素 python 逐位对拍")
# 朴素构建前 3 篇文档的虚拟流, 与 _gather 对拍
naive = []
for i in range(3):
    naive += [mod.BOS] + c.tokens[c._tok_start[i]:c._tok_start[i] + c.doc_lengths[i]].tolist() + [mod.EOS]
got = c._gather(np.arange(len(naive), dtype=np.int64))
check("前 3 篇虚拟流逐位一致", got.tolist() == list(map(int, naive)))
check("block0 以 BOS 开头", int(c.block(0)[0]) == mod.BOS)

# 找一个横跨文档边界的 block: block 0 末尾在第 0 篇内(ctx128 > 文档很长?), 换朴素对拍整个 block1
naive_full = []
for i in range(len(c.doc_lengths)):
    naive_full.append(np.concatenate([[mod.BOS],
                                      np.asarray(c.tokens[c._tok_start[i]:c._tok_start[i] + c.doc_lengths[i]]),
                                      [mod.EOS]]))
stream = np.concatenate(naive_full)
check("stream_len 对账", len(stream) == c.stream_len)
for bi in (0, 1, 5, 100):
    blk = c.block(bi)
    check(f"block{bi} 逐位一致", np.array_equal(blk, stream[bi * 128:(bi + 1) * 128].astype(np.int32)))

print("[3] Batches: 纯函数性 + epoch 重排")
b = mod.Batches(c, batch_size=8, seed=42)
x0a, x0b = b.batch(0), b.batch(0)
check("同 step 两次逐位一致", np.array_equal(x0a, x0b))
check("shape/dtype", x0a.shape == (8, 128) and x0a.dtype == np.int32)
b2 = mod.Batches(c, batch_size=8, seed=42)
check("同 seed 新实例一致", np.array_equal(b2.batch(3), b.batch(3)))
b3 = mod.Batches(c, batch_size=8, seed=43)
check("不同 seed 不同", not np.array_equal(b3.batch(0), b.batch(0)))

print("[4] epoch 跨界: 中间批次 = 上轮尾 + 下轮头")
# 构造 n_blocks 恰好 = 2*B 的小语料, step 2 会跨界
with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    rng = np.random.default_rng(0)
    toks = rng.integers(28, 6400, size=512).astype(np.uint16)  # MASK=27 之上
    # 2 篇文档, 每篇 256 token -> 虚拟流 512+4, 只够 3 块 128; 改: 每篇 250 -> 流 504 -> 3 块
    dl = np.array([250, 250], dtype=np.int32)
    toks = toks[:500]
    (base / "tokens.bin").write_bytes(toks.tobytes())
    np.save(base / "doc_lengths.npy", dl)
    (base / "meta.json").write_text(json.dumps({"vocab_size": 6400, "n_docs": 2, "total_tokens": 500}))
    small = mod.Corpus(base, block_size=128, vocab=6400)
    check("小语料 n_blocks=3", small.n_blocks == 3, small.n_blocks)
    bs = mod.Batches(small, batch_size=2, seed=1)
    perm0 = bs._perm(0)
    perm1 = bs._perm(1)
    x0 = bs.batch(0)   # blocks [perm0[0], perm0[1]]
    x1 = bs.batch(1)   # 跨界: [perm0[2], perm1[0]]
    got1 = np.concatenate([small.block(perm0[2])[None], small.block(perm1[0])[None]])
    check("跨界批次 = 尾块+下轮首块", np.array_equal(x1, got1))
    # 位置推回
    check("position(0)=(0,0)", bs.position(0) == (0, 0))
    check("position(1)=(0,2)", bs.position(1) == (0, 2))
    check("position(2)=(1,1)", bs.position(2) == (1, 1), bs.position(2))

print("[5] 自检会真的失败（检查不能只会通过）")
with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    toks = np.arange(28, 92, dtype=np.uint16)  # 64 token 一篇
    np.save(base / "doc_lengths.npy", np.array([64], dtype=np.int32))
    (base / "tokens.bin").write_bytes(toks.tobytes())
    (base / "meta.json").write_text(json.dumps({"vocab_size": 6400, "n_docs": 1, "total_tokens": 64}))
    # 5a 长度对账失败
    np.save(base / "doc_lengths.npy", np.array([63], dtype=np.int32))
    try:
        mod.Corpus(base, 32, 6400)
        check("长度对账报错", False)
    except ValueError as e:
        check("长度对账报错", "长度对账" in str(e))
    # 5b MASK 污染
    toks2 = toks.copy(); toks2[3] = 27
    np.save(base / "doc_lengths.npy", np.array([64], dtype=np.int32))
    (base / "tokens.bin").write_bytes(toks2.tobytes())
    try:
        mod.Corpus(base, 32, 6400)
        check("MASK 污染报错", False)
    except ValueError as e:
        check("MASK 污染报错", "MASK" in str(e))
    # 5c vocab 对账失败
    (base / "tokens.bin").write_bytes(toks.tobytes())
    try:
        mod.Corpus(base, 32, 9999)
        check("vocab 对账报错", False)
    except ValueError as e:
        check("vocab 对账报错", "vocab" in str(e))
    # 5d id 越界
    toks3 = toks.copy(); toks3[0] = 6400
    (base / "tokens.bin").write_bytes(toks3.tobytes())
    try:
        mod.Corpus(base, 32, 6400)
        check("id 越界报错", False)
    except ValueError as e:
        check("id 越界报错", "越界" in str(e))

print("[6] minimind3-6400 同格式(只对账, 不扫 MASK)")
mbase = "data/prepared/minimind3-6400"
if Path(mbase).exists():
    mm = np.load(mbase + "/doc_lengths.npy", mmap_mode="r")
    meta = json.load(open(mbase + "/meta.json"))
    check("minimind3 sum(doc_lengths) == total", int(mm[:].sum()) == meta["total_tokens"])
else:
    print("  skip: minimind3-6400 不在盘上")

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/04-data [IMPL={MODE}]")
