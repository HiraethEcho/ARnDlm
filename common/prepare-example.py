"""common/prepare-example.py -- common/prepare.py 的参考实现。

下载/整理语料为 tokens.bin + doc_lengths.npy + meta.json（与训练无关的提前准备）。
输出格式与盘上已就位的 prepared 目录一致（test/04 实测过）:
  tokens.bin: 裸文档拼接 uint16（BOS/EOS 包裹是 data 层的职责, 不在这里）
  doc_lengths.npy: int32, 恰好划分 tokens.bin
  meta.json: vocab_size / n_docs / total_tokens / source_file / source_sha256

已知坑（必须大声处理, 不能静默）:
  1. minimind 数据集部分行是 conversations 格式（无 "text" 键）——直接喂会产出 0 token。
     跳过并计数, 计数进 meta。
  2. 文本里若真编码出 MASK id 27（<|buffer1|> 是预留 id）——该文档丢弃并计数,
     否则 data 层的 MASK 自检会让整个 prepared 目录作废。
  3. 编码后 0 token 的文档跳过并计数。

下载路径（hf_hub_download minimind_dataset）今晚未跑 —— 22 GiB, 不在 smoke 范围;
本地 jsonl 整理 + tokenize 路径已测（test/08）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PAD, BOS, EOS, MASK = 0, 1, 2, 27


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(in_path: str | Path, out_dir: str | Path, tokenizer_path: str | Path,
            vocab: int, max_docs: int = 0) -> dict:
    """jsonl(行含 "text") -> prepared 三件套。返回 meta。"""
    in_path, out_dir = Path(in_path), Path(out_dir)
    tk = Tokenizer.from_file(str(tokenizer_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []      # 每文档的 id 数组
    lengths: list[int] = []
    skipped = {"no_text": 0, "empty": 0, "mask_token": 0, "zero_tokens": 0}
    kept = 0

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            if max_docs and kept >= max_docs:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped["no_text"] += 1
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped["no_text" if not isinstance(text, str) else "empty"] += 1
                continue
            ids = tk.encode(text, add_special_tokens=False).ids
            if MASK in ids:
                skipped["mask_token"] += 1
                continue
            if not ids:
                skipped["zero_tokens"] += 1
                continue
            chunks.append(np.asarray(ids, dtype=np.uint16))
            lengths.append(len(ids))
            kept += 1

    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    (out_dir / "tokens.bin").write_bytes(tokens.tobytes())
    np.save(out_dir / "doc_lengths.npy", np.asarray(lengths, dtype=np.int32))
    meta = {
        "vocab_size": vocab,
        "n_docs": kept,
        "total_tokens": int(tokens.size),
        "source_file": str(in_path),
        "source_sha256": _sha256(in_path),
        "skipped": skipped,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python common/prepare-example.py <in.jsonl> <out_dir>")
        sys.exit(2)
    root = Path(__file__).resolve().parent.parent
    meta = prepare(sys.argv[1], sys.argv[2], root / "common/tokenizer.json", vocab=6400)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
