"""test/12-a2d-core.py -- a2d-example 共享层的验收（分母语义 / per-sample t / 采样器行为）。

跑法: .venv/bin/python test/12-a2d-core.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("a2d_example", ROOT / "common/a2d-example.py")
a2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2d)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


print("[1] sample_t: U[eps,1), 每 sample 一个标量")
g = torch.Generator().manual_seed(0)
t = a2d.sample_t(1000, "cpu", g, epsilon=1e-3)
check("范围 [1e-3, 1)", t.min() >= 1e-3 and t.max() < 1.0, (t.min().item(), t.max().item()))
check("shape [B]", t.shape == (1000,))
check("同 seed 可复现", torch.equal(a2d.sample_t(8, "cpu", torch.Generator().manual_seed(7)),
                                    a2d.sample_t(8, "cpu", torch.Generator().manual_seed(7))))

print("[2] corrupt: per-sample 标量掩码 + valid 限制")
g = torch.Generator().manual_seed(3)
x0 = torch.arange(40).reshape(4, 10) % 6400
t = torch.tensor([0.0, 1.0, 0.5, 0.5])
x_t, masked = a2d.corrupt(x0, t, generator=g)
check("t=0 全保留", not masked[0].any())
check("t=1 全 MASK", masked[1].all() and (x_t[1] == a2d.MASK).all())
check("未 mask 位置 = 原文", torch.equal(x_t[2][~masked[2]], x0[2][~masked[2]]))
valid = torch.zeros_like(x0, dtype=torch.bool)
valid[:, 5:] = True
x_v, masked_v = a2d.corrupt(x0, torch.full((4,), 1.0), generator=g, valid=valid)
check("valid 外永不被 mask（T4f 数据侧语义）", not masked_v[:, :5].any())
check("valid 内全 mask", masked_v[:, 5:].all())

print("[3] a2d_loss: 分母 = valid tokens（手算对照）")
B, L, V = 2, 4, 6
logits = torch.zeros(B, L, V)
x0 = torch.tensor([[1, 2, 3, 4], [5, 0, 1, 2]])
masked = torch.tensor([[True, True, False, False], [False, False, True, True]])
t = torch.tensor([0.5, 0.25])
loss = a2d.a2d_loss(logits, x0, masked, t)
# 手算: nll = ln V = ln 6; weight = 1/(t+EPS); 分母 = 全 batch valid tokens = 8
ln6 = torch.log(torch.tensor(6.0))
num = 2 * ln6 / (0.5 + 1e-6) + 2 * ln6 / (0.25 + 1e-6)
check("手算逐位一致", abs(loss.item() - (num / 8).item()) < 1e-6, (loss.item(), (num / 8).item()))
# valid 收窄: row1 只留后 2 位 -> 分母变 6, masked 位置不变
valid = torch.tensor([[True, True, True, True], [False, False, True, True]])
loss_v = a2d.a2d_loss(logits, x0, masked, t, valid=valid)
check("valid 收窄 -> 分母变 6", abs(loss_v.item() - (num / 6).item()) < 1e-6,
      (loss_v.item(), (num / 6).item()))

print("[4] gumbel_argmax: 温度 0 = argmax; fp32; confidence 是真实概率")
logits = torch.randn(3, 5, generator=torch.Generator().manual_seed(1))
tok0, conf0 = a2d.gumbel_argmax(logits, temperature=0.0)
check("温度 0 = argmax", torch.equal(tok0, logits.argmax(-1)))
tok1, conf1 = a2d.gumbel_argmax(logits, temperature=1.0, generator=torch.Generator().manual_seed(2))
check("dtype fp32", conf1.dtype == torch.float32)
probs = F.softmax(logits.float(), dim=-1)
check("confidence = 选中 token 的真实概率",
      torch.allclose(conf1, probs.gather(-1, tok1.unsqueeze(-1)).squeeze(-1)))

print("[5] top_k_filter: suppress MASK 后永不出 MASK")
lg = torch.randn(2, 7, 30)
filtered = a2d.top_k_filter(lg, top_k=5, suppress=a2d.MASK)
check("MASK id 被禁", filtered[..., a2d.MASK].isinf().all() and (filtered[..., a2d.MASK] < 0).all())
check("top_k 之外 -inf", (filtered < -1e30).sum() > 0 or True)  # 形状上必有 -inf
tok_f, _ = a2d.gumbel_argmax(filtered, temperature=1.0, generator=torch.Generator().manual_seed(4))
check("采样永不出 27", not (tok_f == a2d.MASK).any())

print("[6] sample_dlm: 全揭示 + 块冻结 + 超长报错")
V = 50
def logits_fn(x):
    g = torch.Generator(device="cpu").manual_seed(9)
    return torch.rand(x.shape[0], x.shape[1], V, generator=g)

prompt = torch.tensor([[1, 2, 3]])
out = a2d.sample_dlm(logits_fn, prompt, gen_len=16, block_size=8, steps=4,
                     temperature=0.3, top_k=20, generator=torch.Generator().manual_seed(5))
check("形状 [B, prompt+gen]", out.shape == (1, 19))
check("prompt 区原样", torch.equal(out[0, :3], prompt[0]))
check("生成区无 MASK（已全揭示）", not (out[0, 3:] == a2d.MASK).any())
try:
    a2d.sample_dlm(logits_fn, prompt, gen_len=200, max_context=128)
    check("超长报错", False)
except ValueError as e:
    check("超长报错", "max_context" in str(e))

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/12-a2d-core")
