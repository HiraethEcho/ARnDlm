"""test/13-c2-c3-a2d.py -- SPEC C2/C3: 手写 vs dllm loss，同 corruption，相对差 < 1%。

对齐协议（缺一条就不是同 corruption，SPEC 6.1）：
  1. 同 RNG 消耗顺序（t=rand(b) -> mask=rand(b,l)），同 seed 全局重置
  2. 同 eps=1e-3、同权重 1/(t+1e-6)、同分母 valid tokens
  3. 同 forward 语义：两边都双向 —— 这是真实的坑：dllm 的 compute_loss 若收不到
     attention_mask，普通 Qwen3 会**静默走因果 forward**（BOS 前置被
     right_shift_logits=False 关死，从不发生；真正的偏差源是 mask 注入）

C2 = transformers base；C3 = native base（set_causal(False)）。
实测锚点：对齐后相对差 0.0（好于 legacy 7.6e-08）。
跑法: .venv/bin/python test/13-c2-c3-a2d.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("config_example", "common/config-example.py")
mini = load("mini_example", "src/training/a2d_mini-example.py")
dllm_ex = load("dllm_example", "src/training/a2d_dllm-example.py")
trans = load("trans_example", "src/models/trans-example.py")
native = load("native_example", "src/models/native/model-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


cfg = cfg_mod.load("smoketest")
dev = torch.device("cuda")
B, L, SEED = 8, 128, 2024
torch.manual_seed(100)
ids = torch.randint(28, 6400, (B, L), device=dev)

mt = trans.build(cfg.MODEL, seed=42).to(dev).eval()
mn = native.build_native(cfg.MODEL, seed=42).to(dev).eval()
mn.load_state_dict(mt.state_dict(), strict=False)  # 同权重（C3 的 base）
loss_dllm = dllm_ex.DLLMLoss(time_epsilon=1e-3)


def dllm_loss(model, batch):
    torch.cuda.manual_seed(SEED)
    return loss_dllm(model, batch).item()


def mini_loss(model, batch, kind):
    torch.cuda.manual_seed(SEED)
    lf = mini.make_loss_fn(model, kind)  # generator=None -> 全局 RNG（与 dllm 同源）
    with torch.no_grad():
        return lf(model, batch)[0].item()


print("[1] corruption 逐位重放比对（对拍前提, 不是可选检查）")
torch.cuda.manual_seed(SEED)
t = 1e-3 + (1 - 1e-3) * torch.rand(B, device=dev)
probs = torch.rand((B, L), device=dev)
xt_mini = torch.where(probs < t[:, None], torch.full_like(ids, 27), ids)
# dllm 侧同序重放: 同 t 序列; masked = rand < p_mask, p_mask = 1 - alpha(t), alpha=1-t
torch.cuda.manual_seed(SEED)
t_d = 1e-3 + (1 - 1e-3) * torch.rand(B, device=dev)
p_mask = 1.0 - (1.0 - t_d)
probs_d = torch.rand((B, L), device=dev)
xt_dllm = torch.where(probs_d < p_mask[:, None], torch.full_like(ids, 27), ids)
check("t 序列逐位一致", torch.equal(t, t_d))
check("x_t 逐位一致", torch.equal(xt_mini, xt_dllm),
      f"diff {(xt_mini != xt_dllm).sum().item()} 位")

print("[2] C2: transformers base")
l_m = mini_loss(mt, ids, "trans")
l_d = dllm_loss(mt, ids)
rel = abs(l_m - l_d) / max(abs(l_m), 1e-12)
check(f"loss mini={l_m:.6f} dllm={l_d:.6f} 相对差={rel:.2e} < 1%", rel < 0.01, rel)

print("[3] C3: native base（set_causal(False) = A2D 开关）")
mn.set_causal(False)
l_m = mini_loss(mn, ids, "native")
l_d = dllm_loss(mn, ids)
rel = abs(l_m - l_d) / max(abs(l_m), 1e-12)
check(f"loss mini={l_m:.6f} dllm={l_d:.6f} 相对差={rel:.2e} < 1%", rel < 0.01, rel)
mn.set_causal(True)

print("[4] SFT 形状 [B,2,L]：两边都接受 labels 限制且有限")
labels2 = ids.clone()
labels2[:, :64] = -100  # 前半 user 区不监督
batch2 = torch.stack([ids, labels2], dim=1)
l1 = mini_loss(mt, batch2, "trans")
l2 = dllm_loss(mt, batch2)
check(f"mini={l1:.4f} dllm={l2:.4f} 都有限", all(x == x and abs(x) < 1e4 for x in (l1, l2)))

print("[5] 钉子：DLLMLoss 的 4D mask 注入是必要的——去掉就静默变因果")
torch.cuda.manual_seed(SEED)
l_causal = loss_dllm.trainer.compute_loss(
    mt, {"input_ids": ids, "labels": ids}).item()  # 无 attention_mask -> 因果
rel_c = abs(l_causal - l_m) / max(abs(l_m), 1e-12)
check(f"因果 dllm={l_causal:.6f} vs 双向 mini={l_m:.6f} 相对差={rel_c:.2e} > 0 "
      f"（注入口漏掉时这个测试就是能失败的那个）", rel_c > 0)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/13-c2-c3-a2d")
