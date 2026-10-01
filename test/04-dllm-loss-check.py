#!/usr/bin/env python
"""test 04 — MDM 加权 loss 算对了吗。

loss 写错是 DLM 项目最隐蔽的 bug: 数值在合理范围、曲线在下降, 但模型其实
没学到东西。所以这里不用「跑一下看看」, 而是**闭式对拍** —— 构造一个答案能
用 Python 标量手算的 logits, 验证实现返回的就是那个数。

公式 (来自 ~/Public/dllm/dllm/core/trainers/mdlm.py:118-213
         + ~/Public/dllm/dllm/core/schedulers/alpha.py:86-88):

    alpha(t) = 1 - t                                   linear scheduler
    p_mask   = 1 - alpha(t) = t
    weight   = -alpha'(t) / (1 - alpha(t) + 1e-6) = 1/t
    CE[b,l]  = -log softmax(logits[b,l])[input_ids[b,l]]
    loss     = Σ (CE · weight · masked) / (labels != -100 的总数)

**归一化的分母是「可 mask 的 token 总数」, 不是「被 mask 的数量」。**
分母写错 loss 会随 mask 比例漂移, 但曲线照样下降 —— 这就是必须对拍的原因。

Part A 不需要 dllm。Part B 需要 (./scripts/install_dllm.sh)。
"""
import math
import sys

EPS = 1e-6          # dllm/core/schedulers/alpha.py:88 硬编码了这个
VOCAB = 8


def reference_loss(logits, input_ids, labels, p_mask, masked):
    """闭式参照: 全用 Python 标量算, 不借用任何被测代码。"""
    n_valid = 0
    for row in labels:
        n_valid += sum(1 for x in row if x != -100)
    n_valid = max(n_valid, 1)          # clamp_min(1): 全 padding 时不除零

    total = 0.0
    for b in range(len(labels)):
        for l in range(len(labels[b])):
            if not masked[b][l]:
                continue
            row = logits[b][l]
            m = max(row)                                  # 减最大值防 exp 溢出
            exps = [math.exp(x - m) for x in row]
            p = exps[input_ids[b][l]] / sum(exps)
            total += -math.log(p) * (1.0 / (p_mask[b] + EPS))
    return total / n_valid


def main():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        sys.exit("FAIL 没装 torch: ./scripts/setup_env.sh")

    torch.manual_seed(0)
    B, L = 3, 6

    # logits 乘 3.0 是故意的: logits 幅度太小时所有实现的 CE 都≈log(VOCAB),
    # 测不出重加权到底生效没有
    logits = torch.randn(B, L, VOCAB) * 3.0
    input_ids = torch.randint(0, VOCAB, (B, L))

    labels = input_ids.clone()
    # 每个样本的**有效 token 数**刻意不同 (5 / 6 / 4) —— 「分母写错」这个 bug
    # 只有在分子分母不重合时才暴露得出来
    labels[0, 5] = -100
    labels[2, 4] = -100
    labels[2, 5] = -100
    valid = labels != -100

    # 被 mask 的位置**手工指定** (Part A 不走随机采样, loss 必须可复现)。
    # 关键: masked 必须是 valid 的**严格子集**, 且每样本数量不同 (2 / 3 / 1)。
    # 如果 masked 恰好等于 valid, 下面两个反例的 Δ 会变成 0, 测试等于没测。
    masked = torch.zeros(B, L, dtype=torch.bool)
    masked[0, 0] = masked[0, 1] = True
    masked[1, 2] = masked[1, 3] = masked[1, 4] = True
    masked[2, 0] = True
    assert not bool((masked & ~valid).any()), "masked 必须是 valid 的子集"

    # p_mask 是 **per-sample 标量**, 形状 [B] 不是 [B, L]。
    # 三个样本刻意取不同值 —— 如果实现把 p_mask 错当成 per-token 随机量,
    # 权重会在样本内混开, 对拍立刻失败。
    p_mask = torch.tensor([0.2, 0.5, 0.8])

    want = reference_loss(logits.tolist(), input_ids.tolist(),
                          labels.tolist(), p_mask.tolist(), masked.tolist())

    # --- 被测实现: 复现 dllm 的路径 ---
    nll = F.cross_entropy(logits.reshape(-1, VOCAB), input_ids.reshape(-1),
                          reduction="none").reshape(B, L)
    weight = (1.0 / (p_mask + EPS)).unsqueeze(1).expand(B, L)
    got = ((nll * weight * masked.float()).sum()
           / valid.float().sum().clamp_min(1)).item()

    print(f"有效 token 数 (分母) : {int(valid.sum())}")
    print(f"被 mask 位置数       : {int(masked.sum())}")
    print(f"闭式解 : {want:.10f}")
    print(f"实现   : {got:.10f}")

    ok = abs(want - got) <= 1e-5
    if not ok:
        print("FAIL  与闭式解不符")
    else:
        print("  一致")

    # --- 反例: 三种常见写错方式 ---
    # 这些实现全都能跑、曲线全都能降, 但训出来的是错的东西
    n_masked = int(masked.sum())
    n_valid = int(valid.sum())
    # 假装 p_mask 是 per-token 随机量 (真实 bug: 有人写 torch.rand(B, L)
    # 而不是从 per-sample 的 t 推导出标量)。这里用一个**固定的非对称**向量
    # —— 不能用 flip/roll 之类的置换, 置换求和后不变, 抓不到 bug。
    p_token = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2]).unsqueeze(0).expand(B, L)

    variants = {
        "忘了除以 p_mask（没做重加权）":
            (nll * masked.float()).sum().item() / n_valid,
        "分母用 masked 数而不是有效 token 数":
            (nll * weight * masked.float()).sum().item() / n_masked,
        "p_mask 错当 per-token 随机量":
            (nll * (1.0 / (p_token + EPS)) * masked.float()).sum().item() / n_valid,
    }
    print("\n--- 反例: 这些都能跑、曲线都能降, 但都是错的 ---")
    for name, v in variants.items():
        gap = abs(v - want)
        caught = "对拍抓到了" if gap > 1e-3 else "⚠ 抓不到 = 这个测试设计失效了"
        print(f"  {name:34s} {v:8.4f}  Δ={gap:7.4f}   {caught}")
        if gap <= 1e-3:
            ok = False

    # --- Part B: 对拍 dllm 真实的 scheduler ---
    print("\n--- Part B: dllm scheduler (需要 ./scripts/install_dllm.sh) ---")
    try:
        from dllm.core.schedulers import LinearAlphaScheduler
    except ImportError:
        print("SKIP  dllm 未安装")
    else:
        sched = LinearAlphaScheduler()
        for tv in (0.2, 0.5, 0.8):
            t = torch.tensor(tv)
            alpha = sched(t).item()
            p = 1.0 - alpha                       # dllm: p_mask = 1 - alpha(t)
            got_w = sched.weight(t).item()
            want_w = 1.0 / (p + EPS)
            mark = "OK" if abs(got_w - want_w) < 1e-4 else "FAIL"
            print(f"  t={tv}: alpha={alpha:.4f}  p_mask={p:.4f}  "
                  f"weight={got_w:.4f} (期望 {want_w:.4f})  {mark}")
            if mark == "FAIL":
                ok = False

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
