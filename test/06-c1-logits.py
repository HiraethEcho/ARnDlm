"""test/06-c1-logits.py -- SPEC C1: native vs transformers logits max|Δ| <= 1e-5（G1 验收）。

同权重（trans -> native state_dict 直拷）、同 batch、fp32。
顺带验: 参数量 3,213,056（smoketest 解析值）、tied、is_causal=False 双向路径。
跑法: .venv/bin/python test/06-c1-logits.py
"""
import sys
from pathlib import Path

import torch

from _impl import load, MODE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # example 模块内部 import transformers 等时从仓库根找包


cfg_mod = load("config_example", "common/config-example.py")
trans = load("trans_example", "src/models/trans-example.py")
native = load("native_model_example", "src/models/native/model-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


cfg = cfg_mod.load("smoketest")
m = cfg.MODEL
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={dev}  shape=h{m.hidden_size} L{m.num_layers} heads={m.num_heads}/{m.num_kv_heads} "
      f"hd={m.head_dim} ctx={m.context_length}")

print("[1] 构建与参数量")
mt = trans.build(m, seed=42)
mn = native.build_native(m, seed=42)
n = trans.count_params(mt)
check("smoketest 参数量 = 3,213,056", n == 3_213_056, n)
check("trans tied", mt.lm_head.weight.data_ptr() == mt.model.embed_tokens.weight.data_ptr())
check("native tied", mn.lm_head.weight.data_ptr() == mn.model.embed_tokens.weight.data_ptr())

print("[2] state_dict 直拷（名字/形状全对上, 没有改写表）")
missing, unexpected = mn.load_state_dict(mt.state_dict(), strict=False)
check("无缺键", not missing, missing)
check("无多余键", not unexpected, unexpected)
mn = mn.to(dev)
mt = mt.to(dev)
mn.eval()
mt.eval()

print("[3] C1: 因果路径, 3 个 batch")
torch.manual_seed(0)
for bi, B in enumerate((1, 2, 8)):
    ids = torch.randint(28, m.vocab_size, (B, m.context_length), device=dev)  # >=28 避开 special
    with torch.no_grad():
        lt = mt(input_ids=ids).logits.float()
        ln = mn(input_ids=ids).logits.float()
    diff = (lt - ln).abs().max().item()
    check(f"C1 因果 B={B}: max|Δ|={diff:.3e} <= 1e-5", diff <= 1e-5, diff)

print("[4] 双向路径（A2D 开关, 同权重）: HF 传 4D 全零 mask（4.57 的正确做法）")
ids = torch.randint(28, m.vocab_size, (2, m.context_length), device=dev)
zero4d = torch.zeros(2, 1, m.context_length, m.context_length, device=dev)
with torch.no_grad():
    lt_bi = mt(input_ids=ids, attention_mask=zero4d).logits.float()
    ln_bi = mn(input_ids=ids, is_causal=False).logits.float()
diff_bi = (lt_bi - ln_bi).abs().max().item()
check(f"C1 双向: max|Δ|={diff_bi:.3e} <= 1e-5", diff_bi <= 1e-5, diff_bi)
with torch.no_grad():
    lt_ca = mt(input_ids=ids).logits.float()
check("双向与因果确实不同（开关有效）", (lt_bi - lt_ca).abs().max().item() > 1e-3)

print("[5b] tier1 形状的 C1（26.9M, 正式对照前的守门）")
cfg1 = cfg_mod.load("tier1")
m1 = cfg1.MODEL
check("tier1 head_dim 与 26,879,488 锚点一致", trans.count_params(trans.build(m1, seed=0)) == 26_879_488)
mt1 = trans.build(m1, seed=42).to(dev).eval()
mn1 = native.build_native(m1, seed=42).to(dev).eval()
mn1.load_state_dict(mt1.state_dict(), strict=False)
ids1 = torch.randint(28, m1.vocab_size, (1, m1.context_length), device=dev)
with torch.no_grad():
    d1 = (mt1(input_ids=ids1).logits.float() - mn1(input_ids=ids1).logits.float()).abs().max().item()
check(f"tier1 C1 max|Δ|={d1:.3e} <= 1e-5", d1 <= 1e-5, d1)
del mt1, mn1, ids1
torch.cuda.empty_cache()

print("[5] 数值环境是 fp32（对拍前提）")
with torch.no_grad():
    raw_dtype = mt(input_ids=ids).logits.dtype
check("logits 原始 dtype fp32（cast 之前）", raw_dtype == torch.float32)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/06-c1-logits  (C1 因果 {diff:.3e} / 双向 {diff_bi:.3e}, 阈值 1e-5) [IMPL={MODE}]")
