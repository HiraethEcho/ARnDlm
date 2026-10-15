"""test/14-c5-convert.py -- SPEC C5: AR -> A2D 转换 0 丢失 / 0 新增 / 0 形状变化 / 0 漂移。

转换 = config 逐字段拷进 A2DQwen3Config + state_dict 直拷（dllm 的 patched 模型类,
因果 mask 在 modeling 里删掉）。附加: 转换后模型与 trans+4D零mask 的双向 logits 逐位一致。
跑法: .venv/bin/python test/14-c5-convert.py
"""
import sys
from pathlib import Path


import torch

from _impl import load, MODE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


cfg_mod = load("config_example", "common/config-example.py")
trans = load("trans_example", "src/models/trans-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


cfg = cfg_mod.load("smoketest")
dev = torch.device("cuda")

print("[1] config 逐字段拷贝")
from transformers import Qwen3Config  # noqa: E402
import dllm  # noqa: E402
from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel  # noqa: E402

src = trans.build_qwen3_config(cfg.MODEL)
fields = {k: v for k, v in src.to_dict().items() if k != "model_type"}
for key in ("auto_map", "architectures"):
    if hasattr(src, key):
        delattr(src, key)
a2d_cfg = A2DQwen3Config(**fields)
for f in ("hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "intermediate_size",
          "rms_norm_eps", "rope_theta", "tie_word_embeddings", "vocab_size"):
    check(f"config.{f} 一致", getattr(a2d_cfg, f) == getattr(src, f),
          (getattr(a2d_cfg, f), getattr(src, f)))

print("[2] state_dict 直拷（0 丢失 / 0 新增 / 0 形状变化）")
seeded = trans.build(cfg.MODEL, seed=42).to(dev)
torch.manual_seed(0)
mt = trans.build(cfg.MODEL, seed=42).to(dev).eval()
model = A2DQwen3LMHeadModel(a2d_cfg).to(dev).eval()
state = {k: v.to(dev) for k, v in mt.state_dict().items()}
missing, unexpected = model.load_state_dict(state, strict=False)
tied = {"lm_head.weight"}  # tied checkpoint 按设计不含 lm_head
check("0 丢失（tied 别名除外）", {k for k in missing if k not in tied} == set(), missing)
check("0 新增", set(unexpected) == set(), unexpected)

print("[3] 0 形状变化 + 0 漂移")
shape_bad, drift = [], 0
src_sd = mt.state_dict()
dst_sd = model.state_dict()
for k, v in src_sd.items():
    if k not in dst_sd:
        continue
    if dst_sd[k].shape != v.shape:
        shape_bad.append(k)
    elif not torch.equal(dst_sd[k].cpu(), v.cpu()):
        drift += 1
check("0 形状变化", not shape_bad, shape_bad)
check("0 漂移（逐位相等）", drift == 0, drift)
check("tied 双向成立（lm_head 就是 embedding）",
      model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())

print("[4] 转换后 logits = trans + 4D 全零 mask（双向等价, C1 级）")
ids = torch.randint(28, 6400, (2, 128), device=dev)
with torch.no_grad():
    mask4d = torch.zeros(2, 1, 128, 128, device=dev, dtype=mt.dtype)
    lt = mt(input_ids=ids, attention_mask=mask4d).logits.float()
    ln = model(input_ids=ids).logits.float()
diff = (lt - ln).abs().max().item()
check(f"双向 logits max|Δ|={diff:.3e} <= 1e-5", diff <= 1e-5, diff)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/14-c5-convert [IMPL={MODE}]")
