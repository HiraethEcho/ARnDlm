"""utils/tokenizer.py -- encode / decode, 默认不加 special token。

lru_cache 是因为 decoder 是 Hot Path。编码时语料已含 BOS/EOS (data.py), 所以
此处不自动加。"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

from .config import BOS, EOS, TOKENIZER


@lru_cache(maxsize=4)
def load_tokenizer(path: str | None = None) -> Tokenizer:
    # tokenizer.json 解析, 按路径缓存
    return Tokenizer.from_file(str(path or TOKENIZER))


def encode(text: str, bos: bool = False, eos: bool = False) -> list[int]:
    # 编码, 不加 special token; 要 BOS/EOS 时显式给
    ids = load_tokenizer().encode(text, add_special_tokens=False).ids
    return ([BOS] if bos else []) + ids + ([EOS] if eos else [])


def decode(ids) -> str:
    # 解码
    return load_tokenizer().decode([int(i) for i in ids], skip_special_tokens=False)
