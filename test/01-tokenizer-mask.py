#!/usr/bin/env python
"""test 01 — tokenizer 和 mask token 能不能用。

回答四件事:
  1. src/base/tokenizer.json 是不是 vocab 6400 的 BPE
  2. 特殊 token 的 id 对不对 (训练循环要按 id 引用, 写死在这里做回归)
  3. mask token (id 27) 是不是真的空闲 —— 自然文本里绝不能出现
  4. tokenizer 够不够快 (它在 dataloader 热路径上, 慢 2x 就是全局慢 2x)

不需要 torch。tokenizers 是独立 Rust 库, 不依赖 transformers。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "src" / "base" / "tokenizer.json"

# 训练循环按 id 引用这些 token。写死在这里 = tokenizer 一改就 FAIL,
# 不会静默用错 id。
EXPECTED_IDS = {
    "<|endoftext|>": 0,      # PAD
    "<|im_start|>": 1,       # BOS
    "<|im_end|>": 2,         # EOS
    "<|buffer1|>": 27,       # MASK (T1/T2)
}
MASK_ID = 27

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Masked diffusion language models corrupt the input by replacing "
    "tokens with a special mask symbol, then learn to recover them."
)


def main():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("FAIL 没装 tokenizers: uv pip install tokenizers")

    if not TOKENIZER.exists():
        sys.exit(f"FAIL 找不到 {TOKENIZER}")

    tok = Tokenizer.from_file(str(TOKENIZER))
    # get_vocab() 返回 token -> id 的 dict; 反转它才能按 id 查回 token 字符串
    vocab = tok.get_vocab()
    id_to_tok = {i: s for s, i in vocab.items()}
    ok = True

    size = tok.get_vocab_size()
    print(f"vocab : {size}")
    if size != 6400:
        print(f"FAIL  vocab 应该是 6400, 实际 {size}")
        ok = False

    for token, want in EXPECTED_IDS.items():
        got = vocab.get(token)
        if got != want:
            print(f"FAIL  {token}: 期望 id {want}, 实际 {got}")
            ok = False
        else:
            print(f"  id {got:<5d} {token}")

    # mask token 必须是「空闲」的: 拿它当 mask 之前要确认自然文本里不出现。
    # 出现的话, 训练时会把它当噪声 label 算进 loss。
    ids = tok.encode(SAMPLE).ids
    leaked = [i for i, t in enumerate(ids) if t == MASK_ID]
    if leaked:
        print(f"FAIL  mask id {MASK_ID} 出现在自然文本的位置 {leaked}")
        ok = False
    else:
        print(f"  mask id {MASK_ID} 空闲 (自然文本 {len(ids)} token 内未出现)")

    # decode(encode(x)) == x, 确认没有 lossy 的 tokenizer 配置
    if tok.decode(ids) != SAMPLE:
        print("FAIL  decode(encode(x)) != x")
        ok = False
    else:
        print("  往返一致")

    # 速度: 故意超过模型, 逼出真实的 per-batch 编码成本
    big = SAMPLE * 200
    t0 = time.perf_counter()
    for _ in range(20):
        tok.encode(big)
    dt = (time.perf_counter() - t0) / 20
    n_tok = len(tok.encode(big).ids)
    print(f"  速度     : {dt * 1000:6.1f} ms / {len(big):,} 字符"
          f"  = {n_tok / dt / 1000:.0f}k tok/s")

    # 打印一个相邻 id, 方便手工核对表里没有空洞
    print(f"  参考     : id 26={id_to_tok.get(26)!r} id 28={id_to_tok.get(28)!r}")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
