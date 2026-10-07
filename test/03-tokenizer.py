"""test/03-tokenizer.py -- tokenizer-example 的验收（vocab=6400、roundtrip、默认无 special）。

跑法: .venv/bin/python test/03-tokenizer.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tok_example", ROOT / "common/tokenizer-example.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


t = mod.Tok()

print("[1] 词表与 special id")
check("vocab = 6400", t.VOCAB == 6400, t.VOCAB)
check("PAD/BOS/EOS/MASK = 0/1/2/27", (mod.PAD, mod.BOS, mod.EOS, mod.MASK) == (0, 1, 2, 27))
check("decode(1) 是 <|im_start|>", t.tk.id_to_token(1) == "<|im_start|>")
check("decode(2) 是 <|im_end|>", t.tk.id_to_token(2) == "<|im_end|>")
check("id27 是 <|buffer1|>", t.tk.id_to_token(27) == "<|buffer1|>")

print("[2] 默认不加 special token")
ids = t.encode("今天天气不错。")
check("默认无 BOS/EOS", ids[0] != mod.BOS and ids[-1] != mod.EOS, f"head={ids[:2]} tail={ids[-2:]}")
ids_be = t.encode("今天天气不错。", bos=True, eos=True)
check("bos=True 加头", ids_be[0] == mod.BOS)
check("eos=True 加尾", ids_be[-1] == mod.EOS)
check("加包裹后正文一致", ids_be[1:-1] == ids)

print("[3] 中文 roundtrip")
samples = [
    "扩散语言模型把生成当成逐步去噪。",
    "AR 自回归模型按顺序一个 token 一个 token 地出。",
    "混合：Qwen3-style 架构，RoPE + RMSNorm + SwiGLU + GQA。",
    "12345 与 67890 的数字也要能还原。",
]
for s in samples:
    check(f"roundtrip: {s[:12]}…", t.decode(t.encode(s)) == s)

print("[4] MASK 不该被普通文本产出（预留 id，正文禁用）")
mask_hits = [s for s in samples if mod.MASK in t.encode(s)]
check("样例文本不产出 27", not mask_hits, repr(mask_hits))

print("[5] decode 特殊行为")
raw = t.tk.encode("<|im_start|>你好<|im_end|>", add_special_tokens=False).ids
check("文本里的 special 串会被识别", raw[0] == mod.BOS and raw[-1] == mod.EOS, repr(raw))

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/03-tokenizer")
