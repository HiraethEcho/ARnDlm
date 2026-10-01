"""Packed token blocks + a resume-exact batch order.

Why packing: on this box zero-padding wastes 1.71x the bandwidth
(logs/dev/02-hardware-perf.md). Why not torch.utils.data.DataLoader: its
sampler state cannot be checkpointed, and acceptance test C4 requires that
`200 steps + resume` match one continuous `220 steps` run *exactly*. A hand-rolled
index stream is the cheapest way to get that property.

Known limitation, carried over from test/02 and MiniMind: blocks concatenate
documents without an attention mask, so the model can see the previous
document's tail. Deliberately identical to the B-version behaviour; flagged in
RESULTS.md rather than silently "fixed".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .config import BOS, EOS, DATA_DIR


@dataclass
class PackedData:
    blocks: torch.Tensor      # [n_blocks, block_size] int32
    n_docs: int
    n_tokens: int
    block_size: int
    prepared_dir: Path

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather rows and cast to int64 once, at batch level."""
        return self.blocks[idx].long()


def _load(prepared_dir: Path):
    meta = json.loads((prepared_dir / "meta.json").read_text())
    # Explicit little-endian: without "<" a big-endian host silently misreads.
    tokens = np.memmap(prepared_dir / "tokens.bin", dtype="<u2", mode="r")
    lengths = np.load(prepared_dir / "doc_lengths.npy").astype(np.int64)
    if len(lengths) != meta["n_docs"] or int(lengths.sum()) != tokens.shape[0]:
        raise ValueError(
            f"{prepared_dir}: tokens/doc_lengths/meta disagree "
            f"(docs {len(lengths)} vs {meta['n_docs']}, "
            f"tokens {int(lengths.sum())} vs {tokens.shape[0]})"
        )
    return meta, tokens, lengths


def _stream(tokens: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """[BOS] doc0 [EOS] [BOS] doc1 [EOS] ... as one flat int32 array.

    Built with index arithmetic rather than a Python loop over documents: the
    minimind corpus has 1.27M docs and a per-doc loop there costs seconds.
    """
    n = lengths.shape[0]
    padded = lengths + 2                                  # BOS + EOS
    total = int(padded.sum())
    # Two offset tables: `starts` indexes the *stream* (with BOS/EOS gaps),
    # `doc_starts` indexes the *token file* (without them). Mixing them up
    # shifts every body token by a few positions, which looks like plausible
    # data and is not.
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(padded, out=starts[1:])
    doc_starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lengths[:-1], out=doc_starts[1:])

    doc_of = np.repeat(np.arange(n, dtype=np.int64), padded)
    pos = np.arange(total, dtype=np.int64) - starts[doc_of]      # 0=BOS 1..L=body L+1=EOS
    body = (pos >= 1) & (pos <= lengths[doc_of])

    out = np.zeros(total, dtype=np.int32)                  # 0 = PAD, written below
    out[body] = tokens[doc_starts[doc_of][body] + pos[body] - 1].astype(np.int32)
    # BOS/EOS live at the padded boundaries; writing them by index beats
    # scattering them through `body`, which excludes both positions.
    out[starts[:-1]] = BOS
    out[starts[1:] - 1] = EOS
    return out


def build_packed(
    prepared_dir: Path = DATA_DIR,
    block_size: int = 128,
    val_frac: float = 0.0,
    max_blocks: int | None = None,
) -> PackedData:
    """Slice the corpus into fixed-length blocks; the ragged tail is dropped.

    max_blocks caps memory for the 332M-token corpora (int32 blocks):
    128-wide blocks are 512 B each, so 100k blocks is ~50 MB.
    """
    meta, tokens, lengths = _load(Path(prepared_dir))
    if meta.get("vocab_size") != C.VOCAB:
        # A corpus tokenized with a different BPE would index the embedding
        # wrongly, and the only symptom would be a mysteriously high loss.
        raise ValueError(f"{prepared_dir}: vocab {meta.get('vocab_size')} != {C.VOCAB}")
    if val_frac > 0:
        n_val = int(lengths.shape[0] * val_frac)
        train_lengths = lengths[:-n_val] if n_val else lengths
        tokens = tokens[: int(train_lengths.sum())]      # docs are stored contiguously
        lengths = train_lengths
        meta = dict(meta, n_docs=int(lengths.shape[0]))

    stream = _stream(tokens, lengths)
    n_blocks = stream.shape[0] // block_size          # drop_last: fixed batch shapes
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)
    usable = n_blocks * block_size

    blocks = torch.from_numpy(stream[:usable].reshape(n_blocks, block_size).copy())
    return PackedData(blocks=blocks, n_docs=int(lengths.shape[0]), n_tokens=int(usable),
                      block_size=block_size, prepared_dir=Path(prepared_dir))


class BlockStream:
    """Deterministic batch indices with a checkpointable position.

    The permutation is re-derived from (seed, epoch) rather than advanced, so
    resume only has to store the epoch and the offset inside it.
    """

    def __init__(self, n_blocks: int, batch_size: int, seed: int = 42):
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._order = self._perm()

    def _perm(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n_blocks, generator=g)

    def next(self) -> torch.Tensor:
        if self.pos + self.batch_size > self.n_blocks:
            self.epoch += 1
            self.pos = 0
            self._order = self._perm()
        idx = self._order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state(self, state: dict) -> None:
        self.epoch, self.pos = int(state["epoch"]), int(state["pos"])
        self._order = self._perm()


def fixed_batches(n_batches: int, batch_size: int, n_blocks: int, seed: int = 1234) -> list[torch.Tensor]:
    """A frozen batch set for cross-implementation loss comparison (C2/C3).

    Two implementations must be scored on the *same* tokens or the comparison
    measures the data sampler. This is a separate generator from BlockStream's,
    so evaluation batches never correlate with the training order.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=g)
    usable = min(n_batches * batch_size, n_blocks)
    return [perm[i * batch_size : (i + 1) * batch_size] for i in range(usable // batch_size)]