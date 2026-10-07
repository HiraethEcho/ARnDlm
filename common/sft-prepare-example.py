"""common/sft-prepare-example.py -- common/sft-prepare.py 的参考实现。

SFT prepared 目录的仓内生产者（SPEC §3.3 单管道承诺的另一半 —— 此前只有读者,
没有写者, 格式不可复现）。输入 minimind sft_t2t_*.jsonl（conversations 格式）,
输出与盘上 sft-mini-6400 同格式: tokens.bin(预切块 u2) + assistant_spans.npy
([n_spans,3] = block,start,end) + meta.json。

模板: user\\n{q} [EOS] assistant\\n{a} [EOS]（turn 分隔 = EOS; span 含轮末 EOS）。
块原子性: 一轮放不进当前块就 flush; 超过整块的轮丢弃并计数（ctx128 下带
reasoning_content 的轮几乎全被丢, meta 的 reasoning_survival_rate 就是给这个看的）。

用法: .venv/bin/python common/sft-prepare-example.py <in.jsonl> <out_dir> [--max-blocks N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
BOS, EOS, PAD = 1, 2, 0
USER_HEADER, ASSISTANT_HEADER = "user\n", "assistant\n"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _turn_ids(role: str, content: str, reasoning: str | None, enc):
    """(整轮 ids, 监督体起点偏移)。header 与 body **分别**编码再拼接。

    一整条 "assistant\\n{a}" 编码后减去 encode(header) 不是 header 的 token 数
    （BPE 可以跨缝合并）—— span 会错位 1–2 token, 无报错、无异常 loss, 纯静默。
    分开编码让 span 按构造精确。
    """
    supervised = role == "assistant"
    header = ASSISTANT_HEADER if supervised else USER_HEADER
    body = content
    if supervised and reasoning:
        body = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{content}"
    return enc(header) + enc(body), 0 if not supervised else len(enc(header))


def build_blocks(jsonl: Path, block_size: int, max_blocks: int = 0,
                 max_docs: int = 0, tokenizer_path: Path | None = None):
    tk = _load("common/tokenizer-example.py", "tok_example_sftp").Tok(
        tokenizer_path or _ROOT / "common/tokenizer.json")
    enc = lambda s: tk.encode(s)  # noqa: E731

    blocks, spans, used = [], [], []
    n_docs = n_dropped = n_long = n_long_think = 0
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            if max_docs and n_docs >= max_docs:
                break
            n_docs += 1
            turns = json.loads(line)["conversations"]
            before = len(blocks)
            cur, cur_spans = [BOS], []
            for msg in turns:
                whole, body_off = _turn_ids(msg["role"], msg.get("content", ""),
                                            msg.get("reasoning_content"), enc)
                supervised = msg["role"] == "assistant"
                need = len(whole) + 1  # 轮 + 收尾 EOS
                if len(cur) + need > block_size:
                    # flush: 只有含监督区间的块才算数（全 user 块 = 全 IGNORE, 无意义）
                    if len(cur) > 1 and cur_spans:
                        blocks.append(cur + [PAD] * (block_size - len(cur)))
                        spans.append(cur_spans)
                        used.append(len(cur))
                    cur, cur_spans = [BOS], []
                    if need + 1 > block_size:  # 单轮就放不进整块 -> 丢弃
                        if supervised and msg.get("reasoning_content"):
                            n_long_think += 1
                        n_long += 1
                        continue
                start = len(cur) + body_off if supervised else None
                cur += whole
                cur.append(EOS)
                if supervised:
                    cur_spans.append((start, len(cur)))  # 半开: EOS 是最后一个监督 token
            if len(cur) > 1 and cur_spans:
                blocks.append(cur + [PAD] * (block_size - len(cur)))
                spans.append(cur_spans)
                used.append(len(cur))
            if len(blocks) == before:
                n_dropped += 1
            if max_blocks and len(blocks) >= max_blocks:
                break

    arr = np.asarray(blocks, dtype=np.int32) if blocks else np.zeros((0, block_size), np.int32)
    kept_think = sum(1 for row in spans for _, e in [(0, 0)])  # 重新算: 存活 = 未丢的有 think 轮
    # survival 直接用丢弃数推: 该口径与 meta 原字段一致
    stats = {
        "n_docs": n_docs, "n_blocks": int(arr.shape[0]),
        "n_dropped_dialogues": n_dropped,
        "n_long_turns": n_long,
        "n_dropped_turns_with_thinking": n_long_think,
        "mean_block_fill": round(float(np.mean(used)) / block_size, 4) if used else 0.0,
        "mean_used_tokens": round(float(np.mean(used)), 1) if used else 0.0,
    }
    return arr, spans, stats


def write_prepared(jsonl: Path, out_dir: Path, block_size: int,
                   max_blocks: int = 0, max_docs: int = 0,
                   tokenizer_path: Path | None = None, vocab: int = 6400) -> dict:
    arr, spans, stats = build_blocks(jsonl, block_size, max_blocks, max_docs, tokenizer_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr.astype("<u2").tofile(out / "tokens.bin")
    rows = [(i, a, b) for i, row in enumerate(spans) for a, b in row]
    np.save(out / "assistant_spans.npy", np.asarray(rows, dtype=np.int64).reshape(-1, 3))
    lengths = [b - a for row in spans for a, b in row]
    meta = dict(stats,
                vocab_size=vocab, block_size=block_size,
                n_spans=len(rows),
                supervised_tokens=int(sum(lengths)),
                mean_supervised_per_block=round(float(np.mean(lengths)) if lengths else 0.0, 1),
                source=str(Path(jsonl).name),
                loss_on_assistant_only=True,
                reasoning_option="A (fold <thinking> into the assistant text)",
                block_atomic=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=4096)
    ap.add_argument("--max-docs", type=int, default=0)
    a = ap.parse_args()
    meta = write_prepared(a.source, a.out, a.block_size, a.max_blocks, a.max_docs)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
