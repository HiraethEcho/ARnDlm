"""test/11-repl.py -- repl-example 端到端: 四个 ckpt x 两种模式, 管道喂输入。

断言: 正常退出、有 model> 输出、chat 模式多轮不炸。
跑法: .venv/bin/python test/11-repl.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python"
REPL = ROOT / "src/eval/repl-example.py"

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


CASES = [
    ("chat/native-sft", "checkpoints/ar-3.2m-native-sft-smoke", "chat"),
    ("pretrain/native", "checkpoints/ar-3.2m-native-smoke", "pretrain"),
    ("chat/trans-sft", "checkpoints/ar-3.2m-trans-sft-smoke", "chat"),
    ("pretrain/trans", "checkpoints/ar-3.2m-trans-smoke", "pretrain"),
]

for label, ckpt, mode in CASES:
    print(f"[{label}]")
    p = subprocess.run(
        [str(PY), str(REPL), "--ckpt", str(ROOT / ckpt), "--mode", mode],
        input="你好\n:q\n", capture_output=True, text=True, timeout=300, cwd=ROOT)
    check(f"{label}: exit 0", p.returncode == 0, p.stderr[-300:])
    texts = [l.split("model>", 1)[-1].strip() for l in p.stdout.split("\n") if "model>" in l]
    check(f"{label}: 有非空 model> 回答", any(texts), texts)
    if p.returncode != 0:
        print(p.stderr[-800:])

print("[多轮 chat 不炸]")
p = subprocess.run(
    [str(PY), str(REPL), "--ckpt", str(ROOT / "checkpoints/ar-3.2m-native-sft-smoke"),
     "--mode", "chat"],
    input="你好\n今天天气怎么样？\n:c\n再问一次\n:q\n",
    capture_output=True, text=True, timeout=300, cwd=ROOT)
check("exit 0", p.returncode == 0, p.stderr[-300:])
check("三次 model> 回答", p.stdout.count("model>") >= 3, p.stdout)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/11-repl")
