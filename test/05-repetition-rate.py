#!/usr/bin/env python
"""test 05 — 重复率 (repetition rate) 算对了吗。

**这是整个项目的核心指标, 不是 loss。** DLM 失败最典型的表现就是复读,
而 loss 照样下降 —— 只看 loss 会把失败的模型当成成功的。

指标定义: 采样 N 个 token, 切成 n-gram, 重复出现的 n-gram 占比。

    rep_n(text, n) = 1 - (不同的 n-gram 数 / 总 n-gram 数)

n=1 基本等于「 vocab 覆盖率」, 退化; n=4 是常用值。
AR 模型在正常文本上应该 < 5%, 训崩的 DLM 轻松上 60%+。

本脚本分两部分:
  A. 指标实现本身对不对 (用手算得出的文本)
  B. 给定 checkpoint, AR vs DLM 的重复率对比 —— 需要权重, 没权重就 SKIP

用法:
  python test/05-repetition-rate.py
  python test/05-repetition-rate.py --ckpt checkpoints/t1/final.pt
  python test/05-repetition-rate.py --ckpt .../ar.pt --dllm-ckpt .../dllm.pt
"""
import argparse
import importlib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N = 4


def repetition_rate(text, n=DEFAULT_N):
    """重复率: 1 - 去重后的 n-gram 占比。

    先按空白切词再切 n-gram, 不做真正的 subword tokenize ——
    这里要的是「文本在复读没有」, 不是精确的 token 级指标。
    真实评测应该用模型自己的 tokenizer, 见 utils/metrics.py。
    """
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="AR 权重路径")
    ap.add_argument("--dllm-ckpt", type=Path, default=None, help="DLM 权重路径")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--ar-from", default="minimind",
                    choices=["minimind", "transformer"], help="AR 采样走哪条线")
    ap.add_argument("--dllm-from", default="a2d",
                    choices=["a2d", "dllm-a2d", "dllm"], help="DLM 采样走哪条线")
    args = ap.parse_args()

    ok = True

    # ---------------- Part A: 指标实现 ----------------
    print("--- Part A: 指标实现 ---")
    # 用手算得出的例子。
    # rep = 1 - (不同的 n-gram 数 / 总 n-gram 数)
    # 手算 "a b c a b c" n=2: gram = [ab, bc, ca, ab, bc] -> 5 个里 3 个不同 -> 0.4
    #
    # 注意 rep **永远到不了 1.0**: 总有 N 个 gram 时最多 1 - 1/N。
    # "x x x x" n=2 -> gram = [xx, xx, xx] -> 3 个里 1 个不同 -> 0.667
    cases = [
        ("a b c a b c", 2, 0.4),
        ("a b c d e", 2, 0.0),              # 无重复
        ("x x x x", 2, 2 / 3),              # 完全复读
        ("one two three", 2, 0.0),          # 不足 n 个 gram
    ]
    for text, n, want in cases:
        got = repetition_rate(text, n)
        mark = "OK" if abs(got - want) < 1e-9 else "FAIL"
        print(f"  {text!r:20s} n={n}  rep={got:.3f} (期望 {want:.3f})  {mark}")
        if mark == "FAIL":
            ok = False

    # 复读的真实样本长什么样 —— 这就是为什么只看 loss 不够
    print("\n--- 复读样本的形状 (指标的意义) ---")
    loopy = "the cat sat on the mat and the cat sat on the mat and the cat sat on the mat"
    normal = ("machine learning models learn patterns from data and then generalize "
              "those patterns to unseen examples during evaluation")
    for name, t in [("复读", loopy), ("正常", normal)]:
        r = repetition_rate(t, args.n)
        print(f"  {name}  n={args.n}  rep={r:.1%}")
    if repetition_rate(loopy, args.n) <= repetition_rate(normal, args.n):
        print("FAIL  复读样本的重复率应该明显高于正常文本")
        ok = False

    # ---------------- Part B: 实际模型对比 ----------------
    print("\n--- Part B: AR vs DLM (需要 checkpoint) ---")
    if not args.ckpt:
        print("SKIP  没给 --ckpt。这一步等 T1/T2 出权重后再跑。")
        print("\nPASS" if ok else "\nFAIL")
        sys.exit(0 if ok else 1)

    try:
        import torch  # noqa: F401
        import src.transformer, src.minimind, src.a2d, src.dllm  # noqa: F401
    except ImportError as e:
        sys.exit(f"SKIP  implementation not ready ({e})")

    torch.manual_seed(42)
    print(f"prompt : {args.prompt!r}")
    print(f"max_new: {args.max_new}\n")

    results = {}
    # AR 采样在 transformer/ 和 minimind/ 各有一份; DLM 采样在 a2d/ 和
    # dllm-a2d/ 各有一份。这里按 --from 选, 默认走 minimind/ + a2d/。
    runs = [("AR", args.ckpt, f"src.{args.ar_from}.sample")]
    if args.dllm_ckpt:
        runs.append(("DLM", args.dllm_ckpt, f"src.{args.dllm_from}.sample"))

    for name, path, mod_path in runs:
        mod = importlib.import_module(mod_path)
        out = mod.sample(ckpt=path, prompt=args.prompt, max_new_tokens=args.max_new)
        rep = repetition_rate(out, args.n)
        results[name] = rep
        print(f"{name:4s} rep={rep:6.1%}   [{mod_path}]")
        print(f"     {out[:160]!r}\n")

    if "AR" in results and "DLM" in results:
        gap = results["DLM"] - results["AR"]
        print(f"DLM - AR = {gap:+.1%}")
        print("(DLM 更高是正常的 —— 关键是这个差距随训练是否收窄)")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
