"""Thin wrapper over the project tokenizer.

Every cell needs the same two things: turn ids into text for the sample dump,
and turn the fixed prompts into ids. `lru_cache` matters because decoding is
called per sample per cell and `Tokenizer.from_file` re-parses the 6400-entry
BPE each time.
"""
from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

from .config import BOS, EOS, TOKENIZER


@lru_cache(maxsize=4)
def load_tokenizer(path: str | None = None) -> Tokenizer:
    return Tokenizer.from_file(str(path or TOKENIZER))


def encode(text: str, bos: bool = False, eos: bool = False) -> list[int]:
    """Encode *without* special tokens; add them only if asked.

    The training corpus already carries BOS/EOS per document (data.py), so
    adding them here too would double them at the block edges.
    """
    ids = load_tokenizer().encode(text, add_special_tokens=False).ids
    return ([BOS] if bos else []) + ids + ([EOS] if eos else [])


def decode(ids) -> str:
    return load_tokenizer().decode([int(i) for i in ids], skip_special_tokens=False)