"""common/tokenizer-example.py -- common/tokenizer.py 的参考实现。

6400-vocab BPE 的 encode / decode（默认不加 special token）。
读 common/tokenizer.json（tokenizers 库直接吃这个格式）。
语料里的 [BOS]/[EOS] 包裹是 data 层的职责；本模块只在显式要时加。

special id（与 configs tokens 段一致，SPEC 3.2）：
  BOS=1 <|im_start|>  EOS=2 <|im_end|>  MASK=27 <|buffer1|>（预留，正文不许出现）
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = ROOT / "common" / "tokenizer.json"

PAD, BOS, EOS, MASK = 0, 1, 2, 27


class Tok:
    def __init__(self, path: Path | str = TOKENIZER_JSON):
        self.tk = Tokenizer.from_file(str(path))
        self.VOCAB = self.tk.get_vocab_size()

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """默认不加 special token；bos/eos 显式开才加（data 层包裹语料时用）。"""
        ids = self.tk.encode(text, add_special_tokens=False).ids
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        """ids -> 文本。默认跳过 special token（采样展示用）；自检时传 False。"""
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)


if __name__ == "__main__":
    t = Tok()
    ids = t.encode("扩散语言模型把生成当成逐步去噪。", bos=True, eos=True)
    print(f"vocab={t.VOCAB} len={len(ids)} head={ids[:3]} tail={ids[-1]}")
    print("decode:", t.decode(ids))
    print("roundtrip:", t.decode(ids) == "扩散语言模型把生成当成逐步去噪。")
