"""utils/data.py -- 读语料、拼流、切块、按可恢复的顺序出批次。

依赖 utils/config.py（VOCAB / MASK / BLOCK_SIZE / DATA_DIR）。只用 numpy 和 torch。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as C


@dataclass
class PackedData:
    """定长块的集合 + 语料元信息。"""

    blocks: torch.Tensor      # [n_blocks, block_size] int32
    n_docs: int
    n_tokens: int
    block_size: int
    prepared_dir: Path

    def __len__(self) -> int:
        # 块数
        return int(self.blocks.shape[0])

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        # 取出这些块并转 int64（embedding 索引要求 int64）
        return self.blocks[idx].long()


def ids_of(batch: torch.Tensor) -> torch.Tensor:
    # SFT 批次形状是 [2, B, L]，第 0 行才是 ids；把通道轴拆掉
    return batch[0] if batch.dim() == 3 else batch


def _load(prepared_dir: Path):
    # 读 meta.json、tokens.bin、doc_lengths.npy
    meta = json.loads((prepared_dir / "meta.json").read_text())
    # memmap 按需读盘，dtype 显式小端 uint16
    tokens = np.memmap(prepared_dir / "tokens.bin", dtype="<u2", mode="r")
    lengths = np.load(prepared_dir / "doc_lengths.npy").astype(np.int64)
    # 三件文件互相对账
    if len(lengths) != meta["n_docs"] or int(lengths.sum()) != tokens.shape[0]:
        raise ValueError(
            f"{prepared_dir}: tokens/doc_lengths/meta disagree "
            f"(docs {len(lengths)} vs {meta['n_docs']}, "
            f"tokens {int(lengths.sum())} vs {tokens.shape[0]})"
        )
    return meta, tokens, lengths


def _stream(tokens: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    # [BOS] doc0 [EOS] [BOS] doc1 [EOS] ... 拼成一条 int32 流
    n = lengths.shape[0]
    padded = lengths + 2                                  # 每篇加 BOS + EOS 两格
    total = int(padded.sum())
    # starts 是流里每篇的起点（含 BOS/EOS 位），doc_starts 是 token 文件里的起点
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(padded, out=starts[1:])
    doc_starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lengths[:-1], out=doc_starts[1:])

    # 每个位置属于哪一篇文档、是这篇的第几个 token
    doc_of = np.repeat(np.arange(n, dtype=np.int64), padded)
    pos = np.arange(total, dtype=np.int64) - starts[doc_of]      # 0=BOS 1..L=正文 L+1=EOS
    body = (pos >= 1) & (pos <= lengths[doc_of])

    # 正文位置从 token 文件取；BOS/EOS 直接按下标写
    out = np.zeros(total, dtype=np.int32)                  # 0 = PAD
    out[body] = tokens[doc_starts[doc_of][body] + pos[body] - 1].astype(np.int32)
    out[starts[:-1]] = C.BOS
    out[starts[1:] - 1] = C.EOS
    return out


def build_packed(
    prepared_dir: Path = C.DATA_DIR,
    block_size: int = C.BLOCK_SIZE,
    max_blocks: int | None = None,
) -> PackedData:
    # 读语料、做四道自洽检查、切块、丢尾巴
    meta, tokens, lengths = _load(Path(prepared_dir))
    # 语料的词表大小必须和模型一致
    if meta.get("vocab_size") != C.VOCAB:
        raise ValueError(f"{prepared_dir}: vocab {meta.get('vocab_size')} != {C.VOCAB}")
    stream = _stream(tokens, lengths)
    # 最大 id 不能越界
    if int(stream.max()) >= C.VOCAB:
        raise ValueError(f"{prepared_dir}: token id {int(stream.max())} >= {C.VOCAB}")
    # 语料里不允许出现 MASK 的 id
    n_mask_ids = int((stream == C.MASK).sum())
    if n_mask_ids:
        raise ValueError(f"{prepared_dir}: MASK id {C.MASK} 出现在语料里 "
                         f"({n_mask_ids} 个)")

    # 按块长切分，不足一块的尾巴直接丢掉
    n_blocks = stream.shape[0] // block_size
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)
    usable = n_blocks * block_size

    blocks = torch.from_numpy(stream[:usable].reshape(n_blocks, block_size).copy())
    return PackedData(blocks=blocks, n_docs=int(lengths.shape[0]), n_tokens=int(usable),
                      block_size=block_size, prepared_dir=Path(prepared_dir))


class BlockStream:
    """每步给出下一批块的索引；顺序可存进 checkpoint 再恢复。"""

    def __init__(self, n_blocks: int, batch_size: int, seed: int = 42):
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._order = self._perm()

    def _perm(self) -> torch.Tensor:
        # 由 (seed + epoch) 重新生成 permutation，而不是接着上次往下走
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n_blocks, generator=g)

    def next(self) -> torch.Tensor:
        # 这一批吃完了就换 epoch、重排
        if self.pos + self.batch_size > self.n_blocks:
            self.epoch += 1
            self.pos = 0
            self._order = self._perm()
        idx = self._order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx

    def state(self) -> dict:
        # 只存位置和几何，恢复时用同一套 (seed, epoch) 重算 permutation
        return {"epoch": self.epoch, "pos": self.pos,
                "n_blocks": self.n_blocks, "batch_size": self.batch_size}

    def load_state(self, state: dict) -> None:
        # 几何不一致就不是续训，直接报错
        if state.get("n_blocks") not in (None, self.n_blocks) or \
                state.get("batch_size") not in (None, self.batch_size):
            raise ValueError(
                f"resume geometry differs: saved {state.get('n_blocks')} blocks "
                f"of {state.get('batch_size')}, now {self.n_blocks} of "
                f"{self.batch_size}. Resume with the same --max-blocks and "
                f"--batch-size, or it is not a continuation.")
        self.epoch, self.pos = int(state["epoch"]), int(state["pos"])
        self._order = self._perm()


def fixed_batches(n_batches: int, batch_size: int, n_blocks: int, seed: int = 1234) -> list:
    # 固定的一组批次索引，给验证用；seed 和 BlockStream 的分开
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=g)
    usable = min(n_batches * batch_size, n_blocks)
    return [perm[i * batch_size : (i + 1) * batch_size] for i in range(usable // batch_size)]
