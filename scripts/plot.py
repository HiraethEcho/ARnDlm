#!/usr/bin/env python
"""把 logs/run/<实验名>/summary.json 画成 loss 曲线 PNG。

用法:
    python scripts/plot.py logs/run/t1_pretrain
    python scripts/plot.py logs/run/t1_pretrain --out loss.png
    python scripts/plot.py logs/run/t1_pretrain --keys loss train_tps

summary.json 是训练日志的唯一真相 (trainer 追加写, 每行一个 JSON 对象)。
metrics.csv 是同样内容的表格版 —— 这个脚本只读 json, 不读 csv。

设计: 不依赖 tensorboard。matplotlib 装了就画, 没装就提示装。
"""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    """读 summary.json。

    追加写的日志可能最后一行是半截的 (训练被 kill 时), 所以跳过
    解析失败的行而不是直接崩 —— 曲线少最后一个点比看不到图有用。
    """
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  跳过 {bad} 行解析失败 (最后一行可能是被 kill 时的半截)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="logs/run/<实验名>")
    ap.add_argument("--out", type=Path, default=None, help="默认写 <run_dir>/loss.png")
    ap.add_argument("--keys", nargs="*", default=None, help="默认自动挑数值型字段")
    ap.add_argument("--log", action="store_true", help="x 轴用 log1p(step)")
    args = ap.parse_args()

    summary = args.run_dir / "summary.json"
    if not summary.exists():
        sys.exit(f"找不到 {summary}")

    rows = load(summary)
    if not rows:
        sys.exit(f"{summary} 里没有可用数据")

    try:
        import matplotlib
        matplotlib.use("Agg")          # 无显示环境, 不弹窗
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("没装 matplotlib: uv pip install --python .venv/bin/python matplotlib")

    xkey = "step" if "step" in rows[0] else next(iter(rows[0]), "row")
    xs = [r.get(xkey, i) for i, r in enumerate(rows)]
    if args.log:
        xs = [x + 1 for x in xs]
        import math
        xs = [math.log1p(x) for x in xs]

    # 自动挑字段: 数值型 + 至少 2 个不同取值 (常量字段画出来是直线, 没信息)
    if args.keys:
        keys = args.keys
    else:
        keys = [
            k for k, v in rows[0].items()
            if k != xkey and isinstance(v, (int, float))
            and len({r.get(k) for r in rows if isinstance(r.get(k), (int, float))}) > 1
        ][:4]

    if not keys:
        sys.exit(f"{xkey} 之外没有可画的数值字段")

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 2.6 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]                      # subplots 返回标量时的坑

    for ax, k in zip(axes, keys):
        ys = [r.get(k) for r in rows]
        pairs = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float))]
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], lw=1.2)
        ax.set_ylabel(k)
        ax.grid(alpha=0.3)
        # loss 曲线前几个点通常在高位, y 轴从 0 起会压扁有效区间
        if k == "loss":
            ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("log(step)" if args.log else xkey)
    fig.suptitle(f"{args.run_dir.name}  ({len(rows)} points)")
    fig.tight_layout()

    out = args.out or args.run_dir / "loss.png"
    fig.savefig(out, dpi=130)
    print(f"画了 {', '.join(keys)} -> {out}")


if __name__ == "__main__":
    main()
