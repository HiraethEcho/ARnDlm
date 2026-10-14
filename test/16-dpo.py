"""test/16-dpo.py -- dpo-example 的专项验收（此前只有 smoke run 的 margin acc 背书）。

钉住的语义:
  ① 数据格式: 右 pad 定长、response 区间 mask 精确（prompt 段永不全监督）
  ② shift 语义: seq_logp 与手算 CE 逐位一致; 全 False 的 mask -> logp == 0
  ③ 方向性: policy == ref -> loss == ln2、margin == 0（数学恒等式, 不是「差不多」）
  ④ margin 正负: 用桩替换 seq_logp, 构造「policy 更爱 chosen」-> acc=1, 反向 -> acc=0
  ⑤ lr 缩放: main 的 DPO 路径 lr == ctx.lr_at(step) * 0.1
跑法: .venv/bin/python test/16-dpo.py
"""
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("c", "common/config-example.py")
dx = load("dx", "src/training/dpo-example.py")
te = load("te", "src/models/trans-example.py")
main_mod = load("m", "src/training/main-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


DPO_JSONL = "/mnt/extra/data/raw/minimind_dataset/dpo.jsonl"
dev = torch.device("cuda")

print("[1] 数据格式: 右 pad 定长 + response mask 精确")
data = dx.load_dpo_pairs(DPO_JSONL, max_pairs=64, total_len=120)
L = data["seq"].shape[1]
check("n_pairs = 64", data["n_pairs"] == 64, data["n_pairs"])
check("seq 形状 [2N, 120]", data["seq"].shape == (128, 120), data["seq"].shape)
check("mask dtype bool 且同形", data["resp_mask"].dtype == bool
      and data["resp_mask"].shape == data["seq"].shape)
# chosen/rejected 相邻成对: 同 prompt 前缀（前 8 个 token 应相同）
same_prefix = all(np.array_equal(data["seq"][2 * i][:8], data["seq"][2 * i + 1][:8])
                  for i in range(data["n_pairs"]))
check("成对行 shared prompt 前缀", same_prefix)
# mask 是单段连续区间（无空洞）: 非零下标的差分全为 1
contig = 0
for i in range(len(data["seq"])):
    nz = np.flatnonzero(data["resp_mask"][i])
    if len(nz) and np.array_equal(np.diff(nz), np.ones(len(nz) - 1, dtype=nz.dtype)):
        contig += 1
check("mask 全是单段连续区间", contig == len(data["seq"]), f"{contig}/{len(data['seq'])}")
starts = [int(np.flatnonzero(data["resp_mask"][i])[0]) for i in range(len(data["seq"]))]
check("监督区从 prompt 之后开始（>0）", min(starts) > 0, min(starts))

print("[2] shift 语义: seq_logp == 手算 CE; 空 mask -> 0")
cfg = cfg_mod.load("smoketest")
mt = te.build(cfg.MODEL, seed=42).to(dev).eval()
ids = torch.from_numpy(data["seq"][:4]).long().to(dev)
mask = torch.from_numpy(data["resp_mask"][:4]).to(dev)
with torch.no_grad():
    lp = dx.seq_logp(mt, ids, mask)
    logits = mt(ids).logits[:, :-1].float()
    tgt = ids[:, 1:]
    manual = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
        reduction="none").view_as(tgt)
    manual = (manual * mask[:, 1:]).sum(-1)
check("seq_logp 与手算逐位一致", torch.equal(lp, manual), (lp.tolist(), manual.tolist()))
with torch.no_grad():
    lp0 = dx.seq_logp(mt, ids, torch.zeros_like(mask))
check("全 False mask -> 0", torch.equal(lp0, torch.zeros_like(lp0)))

print("[3] 方向性: policy == ref -> loss == ln2, margin == 0")
ref = mt  # 同一个模型
seq_c, seq_r, mask_c, mask_r = dx.pair_batch(data, np.arange(8), dev)
with torch.no_grad():
    loss0, aux0 = dx.dpo_loss(mt, ref, seq_c, seq_r, mask_c, mask_r)
check(f"loss == ln2 ({math.log(2):.6f})", abs(loss0.item() - math.log(2)) < 1e-6, loss0.item())
check("margin == 0", abs(aux0["margin"]) < 1e-9, aux0["margin"])

print("[4] margin 正负: 用桩替换 seq_logp（不依赖训练动态, 无 flakiness）")
orig = dx.seq_logp


def stub(c, r):
    def f(model, ids, mask):
        # policy: chosen 0 / rejected -1；ref: 两侧皆 0
        val = 0.0 if mask is mask_c or ids is seq_c else -1.0
        return torch.full((ids.shape[0],), val, device=ids.device)
    return f


# 直接构造四个 logprob 张量更直白: policy 更爱 chosen -> margin 正
class _Fake:
    pass


real_seq_logp = dx.seq_logp
calls = {"n": 0}


def fake_seq_logp(model, ids, mask):
    """policy/ref 由调用顺序区分: dpo_loss 先 ref 两次, 再 policy 两次。"""
    calls["n"] += 1
    same_ids = torch.equal(ids, seq_c)
    if calls["n"] <= 2:      # ref: 两侧都 0
        return torch.zeros(ids.shape[0], device=ids.device)
    return torch.full((ids.shape[0],), 0.0 if same_ids else -1.0, device=ids.device)


dx.seq_logp = fake_seq_logp
loss1, aux1 = dx.dpo_loss(mt, ref, seq_c, seq_r, mask_c, mask_r)
dx.seq_logp = real_seq_logp
beta = 0.1
expect = -math.log(1 / (1 + math.exp(-beta)))  # -logsigmoid(beta)
check(f"policy 更爱 chosen -> margin=+{beta}, acc=1", abs(aux1["margin"] - beta) < 1e-6
      and aux1["acc"] == 1.0 and abs(loss1.item() - expect) < 1e-6,
      (aux1, loss1.item(), expect))


def fake_rev(model, ids, mask):
    calls["n"] += 1
    same_ids = torch.equal(ids, seq_c)
    if calls["n"] <= 2:
        return torch.zeros(ids.shape[0], device=ids.device)
    return torch.full((ids.shape[0],), -1.0 if same_ids else 0.0, device=ids.device)


calls["n"] = 0
dx.seq_logp = fake_rev
loss2, aux2 = dx.dpo_loss(mt, ref, seq_c, seq_r, mask_c, mask_r)
dx.seq_logp = real_seq_logp
check("反向（更爱 rejected）-> margin=-beta, acc=0", abs(aux2["margin"] + beta) < 1e-6
      and aux2["acc"] == 0.0, aux2)

print("[5] PairBatcher 纯函数 + chosen/rejected 相邻映射")
pb = dx.PairBatcher(data["n_pairs"], batch_size=4, seed=7)
check("同 step 两次一致", np.array_equal(pb.batch(0), pb.batch(0)))
check("不同 seed 不同", not np.array_equal(
    dx.PairBatcher(data["n_pairs"], 4, 8).batch(0), pb.batch(0)))
idx = pb.batch(0)
sc, sr, mc, mr = dx.pair_batch(data, idx, dev)
check("chosen 行 == seq[2i]", torch.equal(sc.cpu(), torch.from_numpy(data["seq"][idx * 2])))
check("rejected 行 == seq[2i+1]", torch.equal(sr.cpu(), torch.from_numpy(data["seq"][idx * 2 + 1])))

print("[6] main 的 DPO 路径 lr == ctx.lr_at(step) * 0.1 + 产物落盘")
tmp = Path(tempfile.mkdtemp(prefix="dpo-"))
try:
    cfg2 = cfg_mod.load("smoketest")
    run_dir = tmp / "run"
    run_dir.mkdir()
    ctx = main_mod.Ctx(cfg2, "ar-3.2m-trans-sft-dpo-t", ROOT / "data/prepared/sft-mini-6400",
                       run_dir, run_dir / "ckpt")

    class A:  # argparse 替身
        impl = "trans"
        base = str(ROOT / "checkpoints/ar-3.2m-trans-sft-smoke")
        dpo_data = DPO_JSONL
        dpo_pairs = 64
    res = main_mod.run_dpo(cfg2, 2, ctx, A())
    lr_now = res["optimizer"].param_groups[0]["lr"]
    check("lr == ctx.lr_at(1) * 0.1", abs(lr_now - ctx.lr_at(1) * 0.1) < 1e-12,
          (lr_now, ctx.lr_at(1) * 0.1))
    check("summary 报 heldout acc", "heldout_acc" in res["dpo"], res["dpo"])
    check("checkpoint 落盘", (run_dir / "ckpt/model.safetensors").exists())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/16-dpo")
