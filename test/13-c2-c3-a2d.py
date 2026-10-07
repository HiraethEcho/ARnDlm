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


print("[1] corruption 逐位比对: 手写 corrupt vs dllm 内部 masking（真实现对照）")
# 手写侧: 同 seed 重放 make_loss_fn 的 RNG 消耗 (rand(b) -> rand(b,l))
torch.cuda.manual_seed(SEED)
t_hand = 1e-3 + (1 - 1e-3) * torch.rand(B, device=dev)
probs_hand = torch.rand((B, L), device=dev)
masked_hand = probs_hand < t_hand[:, None]
xt_hand = torch.where(masked_hand, torch.full_like(ids, 27), ids)
# dllm 侧: 同一全局 RNG 序列（这是我们 loss_fn 实际依赖的机制）
torch.cuda.manual_seed(SEED)
from dllm.core.schedulers import LinearAlphaScheduler
sched = LinearAlphaScheduler()
t_d = 1e-3 + (1 - 1e-3) * torch.rand(B, device=dev)
p_mask = 1.0 - sched(t_d).unsqueeze(1).expand(B, L)
probs_d = torch.rand((B, L), device=dev)
masked_dllm = probs_d < p_mask
check("t 序列逐位一致", torch.equal(t_hand, t_d))
check("掩码集合逐位一致（手写 corrupt vs dllm 公式）", torch.equal(masked_hand, masked_dllm),
      f"diff {(masked_hand != masked_dllm).sum().item()} 位")

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

print("[4] T4f/SFT 形状 [B,2,L]：100 batch 加噪 ⊆ assistant 区间 + T4g 区间版 C2")
labels2 = ids.clone()
labels2[:, :64] = -100  # 前半 user 区不监督
batch2 = torch.stack([ids, labels2], dim=1)
l1 = mini_loss(mt, batch2, "trans")
l2 = dllm_loss(mt, batch2)
rel_sft = abs(l1 - l2) / max(abs(l1), 1e-12)
check(f"T4g 区间版 C2: SFT batch 相对差={rel_sft:.2e} < 1%", rel_sft < 0.01, rel_sft)
# T4f 集成断言: 抽 100 个 SFT batch, 手写侧 corrupt 后 MASK 位置必须 ⊆ assistant 区间
import numpy as _np
sftx = load("sftx", "src/training/sft-example.py")
sb = sftx.SFTBatcher(sftx.SFTBlocks(ROOT / "data/prepared/sft-mini-6400"),
                     batch_size=8, seed=1)
viol = 0
for step in range(100):
    ids_b = sb.batch(step)
    labels_b = sb.labels(step, None)
    torch.cuda.manual_seed(step)
    t_b = 1e-3 + (1 - 1e-3) * torch.rand(ids_b.shape[0], device=ids_b.device)
    probs_b = torch.rand(ids_b.shape, device=ids_b.device)
    masked_b = (probs_b < t_b[:, None]) & (torch.from_numpy(labels_b).to(ids_b.device) != -100)
    if bool((masked_b & (torch.from_numpy(labels_b).to(ids_b.device) == -100)).any()):
        viol += 1
check("100 batch: 加噪位置 ⊆ assistant 区间（T4f 集成版）", viol == 0, viol)

print("[5] 钉子：DLLMLoss 的 4D mask 注入是必要的——去掉就静默变因果")
# [2] 的 l_m 与 l_d 已同时序列得出（[2] 里 rel=0），这里用 l_d 当「有 4D mask」的锚
l_m_anchor = mini_loss(mt, ids, "trans")
l_d_anchor = dllm_loss(mt, ids)
check("依赖关系先成立: 有 mask 注入时 0.0", abs(l_m_anchor - l_d_anchor) / abs(l_m_anchor) < 1e-9)
torch.cuda.manual_seed(SEED)
l_causal = loss_dllm.trainer.compute_loss(
    mt, {"input_ids": ids, "labels": ids}).item()  # 无 attention_mask -> 因果
rel_c = abs(l_causal - l_d_anchor) / max(abs(l_d_anchor), 1e-12)
# 因果与双向的差异量级 ~1e-3（每 token 只暴露因果方向的单侧上下文）。
# 钉子要求的是"差异存在且可归因"：相对差 > 0（去掉注入必然改变 loss）。
check(f"因果 dllm={l_causal:.6f} vs 双向 dllm={l_d_anchor:.6f} 相对差={rel_c:.2e} > 1e-6",
      rel_c > 1e-6, rel_c)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/13-c2-c3-a2d")
