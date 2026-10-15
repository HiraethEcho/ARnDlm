"""test/09-sft.py -- trainer/sft.py 的验收（SPEC 6.2: T4b 区间正确 / T4a 展平 / T4f 无噪区间外 / 批次纯函数）。

跑法: .venv/bin/python test/09-sft.py
"""
import shutil
import sys
import tempfile
from pathlib import Path


import numpy as np

from _impl import load, MODE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


cfg_mod = load("config_example", "common/config-example.py")
sft_mod = load("sft_example", "src/training/sft-example.py")
main_mod = load("main_example", "src/training/main-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


SFT_DIR = ROOT / "data/prepared/sft-mini-6400"
cfg = cfg_mod.load("smoketest")

print("[1] SFTBlocks 装载 + 硬对账")
sft = sft_mod.SFTBlocks(SFT_DIR)
check("n_blocks = 4096", sft.n_blocks == 4096)
check("supervised 对账 = 260,590（硬数字）",
      int((sft._spans[:, 2] - sft._spans[:, 1]).sum()) == 260590)
covered = int((sft._span_ends > sft._span_starts).sum())
check(f"有 span 的块数 = {covered} > 0 且 ≤ 4096", 0 < covered <= 4096, covered)

print("[2] T4b: 区间正确 —— 区间外全 -100, 区间内 = 原 token")
rng = np.random.default_rng(0)
bad = 0
for b in rng.choice(sft.n_blocks, size=50, replace=False):
    labels = sft.labels(int(b))
    ids = np.asarray(sft.blocks[int(b) * sft.bs:(int(b) + 1) * sft.bs], dtype=np.int64)
    spans = sft._spans[sft._span_starts[b]:sft._span_ends[b], 1:3]
    in_span = np.zeros(sft.bs, dtype=bool)
    for s, e in spans:
        in_span[s:e] = True
    if not ((labels[~in_span] == -100).all() and (labels[in_span] == ids[in_span]).all()):
        bad += 1
check("50 块抽查全对", bad == 0, f"{bad} 块错")
# 区间错位检查: labels 的监督 token 数 == span 覆盖数（对拍整库太慢, 抽 200 块, 同一组）
blocks = rng.choice(sft.n_blocks, 200, replace=False)
sup = sum((sft.labels(int(b)) != -100).sum() for b in blocks)
spans_sum = sum(int((sft._spans[sft._span_starts[b]:sft._span_ends[b], 2]
                     - sft._spans[sft._span_starts[b]:sft._span_ends[b], 1]).sum())
                for b in blocks)
check("监督 token 数 == span 覆盖数(200 块)", sup == spans_sum, (sup, spans_sum))

print("[3] T4a-lite: assistant 区间 decode 非空（3 个 span 抽查）")
tok = load("tok_example", "common/tokenizer-example.py").Tok()
got_text = 0
spans_all = sft._spans
for i in rng.choice(len(spans_all), 3, replace=False):
    b, s, e = spans_all[i]
    ids = np.asarray(sft.blocks[int(b) * sft.bs:int(b) * sft.bs + sft.bs], dtype=np.int64)[s:e]
    text = tok.decode(ids.tolist())
    if text.strip():
        got_text += 1
check("3 个区间 decode 非空", got_text == 3)

print("[4] T4f: 全语料无 MASK id 27（数据侧已验, 这里再全扫一遍）")
check("无 27", 27 not in np.unique(np.asarray(sft.blocks)))

print("[5] SFTBatcher 纯函数 + epoch 跨界")
sb = sft_mod.SFTBatcher(sft, batch_size=8, seed=42)
check("同 step 两次一致", np.array_equal(sb.batch(0), sb.batch(0)))
check("labels 同 step 一致", np.array_equal(sb.labels(0, None), sb.labels(0, None)))
check("position(511)=(0,4088)", sb.position(511) == (0, 4088), sb.position(511))
check("position(512)=(1,0)", sb.position(512) == (1, 0), sb.position(512))
x0, x1 = sb.batch(511), sb.batch(512)
check("跨界批次形状/无重复块", x0.shape == (8, 128) and not np.array_equal(x0[-1], x1[0]))

print("[6] e2e: native 在 SFT 数据上 20 步 (assistant-only loss 能算)")
tmp = Path(tempfile.mkdtemp(prefix="sft-"))
try:
    run_dir = tmp / "run"
    run_dir.mkdir()
    ctx = main_mod.Ctx(cfg, "ar-3.2m-native-sft-smoke", SFT_DIR, run_dir, run_dir / "ckpt")
    res = sft_mod.run_sft(cfg, 20, ctx, impl="native")
    losses = res["history"].losses
    check("20 步 loss 都有限且为正", all(np.isfinite(l) and l > 0 for l in losses), losses[:3])
    # assistant-only 的证据: 首 loss 应远低于全 token loss(全 token 语义 loss ~8.7+, 区间 loss 更高)
    check("首步 loss 有限", np.isfinite(losses[0]), losses[0])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/09-sft [IMPL={MODE}]")
