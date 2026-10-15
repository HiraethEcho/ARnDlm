"""common/data-example.py -- prepare/data.py 的参考实现。

tokens.bin(裸文档拼接, uint16) + doc_lengths.npy -> 虚拟 [BOS] doc [EOS] 流 -> 定长块。
不物化大流: 位置 -> (doc, local) 用 searchsorted 映射, tier1 332M token 也不用复制。
批次顺序 = f(seed, step) 纯函数, resume 只需 step (C4 的根); epoch 跨界自动接下轮重排。

加载即四道自检 (SPEC 3.3): 长度对账 / vocab 对账 / ids 越界 / MASK 不在正文。
数据格式 (实测自 tinystories-6400): tokens.bin 无 BOS/EOS, doc_lengths 恰好划分它。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAD, BOS, EOS, MASK = 0, 1, 2, 27  # 与 common/tokenizer-example.py 同源 (SPEC 3.2)


class Corpus:
    """prepared 目录 -> 定长块的只读视图。"""

    def __init__(self, prepared_dir: str | Path, block_size: int,
                 vocab: int, max_blocks: int = 0, check_mask: bool = True):
        base = Path(prepared_dir)
        meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
        self.tokens = np.memmap(base / "tokens.bin", dtype=np.uint16, mode="r")
        self.doc_lengths = np.load(base / "doc_lengths.npy")
        problems = []

        # 自检 1: 长度对账 —— doc_lengths 恰好划分 tokens.bin
        if len(self.tokens) != int(self.doc_lengths.sum()) or len(self.tokens) != meta["total_tokens"]:
            problems.append(f"长度对账失败: tokens={len(self.tokens)} "
                            f"sum(doc_lengths)={int(self.doc_lengths.sum())} meta={meta['total_tokens']}")
        # 自检 2: vocab 对账
        if meta["vocab_size"] != vocab:
            problems.append(f"vocab 对账失败: meta={meta['vocab_size']} tokenizer/config={vocab}")
        # 自检 3: ids 越界
        if len(self.tokens) and int(self.tokens.max()) >= vocab:
            problems.append(f"token id 越界: max={int(self.tokens.max())} >= vocab {vocab}")
        # 自检 4: MASK 不在正文 (332M 扫一遍是秒级, 换来「区间污染」不可能发生)
        if check_mask and len(self.tokens) and bool((self.tokens == MASK).any()):
            problems.append(f"MASK id {MASK} 出现在语料正文")
        if problems:
            raise ValueError(f"{base} 数据自检失败 ({len(problems)} 条):\n  " + "\n  ".join(problems))

        n = len(self.doc_lengths)
        tok_start = np.zeros(n, dtype=np.int64)
        np.cumsum(self.doc_lengths[:-1], out=tok_start[1:])  # 每篇在 tokens.bin 里的起点
        virt_start = tok_start + 2 * np.arange(n, dtype=np.int64)  # 包裹 BOS 后的虚拟流起点
        self._tok_start = tok_start
        self._virt_start = virt_start
        self._virt_end = virt_start + self.doc_lengths + 2
        self.stream_len = int(self.doc_lengths.sum()) + 2 * n
        blocks = self.stream_len // block_size  # 末尾不足一块的丢弃
        if max_blocks:
            blocks = min(blocks, max_blocks)
        self.n_blocks = blocks
        self.block_size = block_size

    def _gather(self, positions: np.ndarray) -> np.ndarray:
        """虚拟流位置 -> token id (向量化, BOS/EOS 在边界插入)。"""
        d = np.searchsorted(self._virt_start, positions, side="right") - 1
        if d.min() < 0 or d.max() >= len(self.doc_lengths):
            raise IndexError(f"位置越界: [{positions.min()}, {positions.max()}]")
        local = positions - self._virt_start[d]
        L = self.doc_lengths[d]
        out = np.empty(len(positions), dtype=np.int32)
        is_bos = local == 0
        is_eos = local == L + 1
        body = ~(is_bos | is_eos)
        out[is_bos] = BOS
        out[is_eos] = EOS
        out[body] = self.tokens[self._tok_start[d[body]] + (local[body] - 1)]
        return out

    def block(self, i: int) -> np.ndarray:
        if not 0 <= i < self.n_blocks:
            raise IndexError(f"block {i} 超出 [0, {self.n_blocks})")
        pos = i * self.block_size + np.arange(self.block_size, dtype=np.int64)
        return self._gather(pos)


class Batches:
    """step -> batch 的纯函数。同 seed 同 step 同内容, 与调用历史无关。"""

    def __init__(self, corpus: Corpus, batch_size: int, seed: int):
        if corpus.n_blocks < batch_size:
            raise ValueError(f"n_blocks({corpus.n_blocks}) < batch_size({batch_size})")
        self.corpus = corpus
        self.B = batch_size
        self.seed = seed

    def _perm(self, epoch: int) -> np.ndarray:
        # default_rng([seed, epoch]): 每个 epoch 的重排只依赖 (seed, epoch),
        # 与已消耗多少步无关 -> resume 不需要存 permutation 本身
        return np.random.default_rng([self.seed, epoch]).permutation(self.corpus.n_blocks)

    def position(self, step: int) -> tuple[int, int]:
        pos = step * self.B
        return pos // self.corpus.n_blocks, pos % self.corpus.n_blocks

    def _indices(self, step: int) -> np.ndarray:
        epoch, within = self.position(step)
        cur = self._perm(epoch)[within:]
        if len(cur) >= self.B:
            return cur[:self.B]
        nxt = self._perm(epoch + 1)
        return np.concatenate([cur, nxt[:self.B - len(cur)]])

    def batch(self, step: int) -> np.ndarray:
        """[B, block_size] int32。step 从 0 计。"""
        idx = self._indices(step)
        pos = idx[:, None] * self.corpus.block_size + np.arange(self.corpus.block_size, dtype=np.int64)
        return self.corpus._gather(pos.ravel()).reshape(self.B, self.corpus.block_size)


if __name__ == "__main__":
    c = Corpus("data/prepared/tinystories-6400", block_size=128, vocab=6400, max_blocks=4096)
    b = Batches(c, batch_size=8, seed=42)
    x = b.batch(0)
    print(f"n_blocks={c.n_blocks} stream={c.stream_len} batch0 {x.shape} dtype={x.dtype}")
    print("block0 head:", x[0, :12].tolist())
