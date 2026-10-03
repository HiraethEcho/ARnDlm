"""SFT: dialogues -> packed blocks with per-token loss spans.

Why this file exists: the pretraining corpus is `{"text": ...}` one document per
line, and `data.py` turns that into blocks. The SFT corpus is
`{"conversations": [{"role": "user", "content": ...}, ...]}` -- **no `text` key**
-- so `data.py` silently produces zero tokens. Confirmed by reading the file:
300 sampled rows, none of them has `text`.

The one rule everything else serves (GOAL.md Q4): **loss is computed on
assistant tokens only.** The user half of a turn is context, not a target. That
is what `spans` is for; get it wrong and the model trains on the wrong objective
while every number still looks plausible.

Two facts about this corpus that are not visible from the schema:

* 19% of assistant messages carry `reasoning_content`, a ~300-character think
  block against a ~28-character answer. GOAL.md Q4 chose **option A**: fold it
  into the assistant text and supervise it. The cost, recorded so it is not
  rediscovered: the think block is 10x the answer, so an SFT loss is dominated by
  think text and **cannot be compared against a pretraining loss**.
* There is no system message. 20,000 sampled dialogues all start with `user`.
  This module does not invent one.

No ChatML markers: the 6400-vocab tokenizer has `<think>`/`</think>` (25/26)
and the `buffer` tokens, but no `<|im_start|>`/`<|im_end|>`. SPEC 3.3 forbids
growing the vocab, so role headers are plain text.

Block boundaries are **turn-atomic**. A half-cut turn would leave an assistant
span truncated at the block edge, which is exactly what makes "assistant-only
loss" quietly untrue. GOAL.md §8 chose whole-turn drop over mid-turn truncation
for the same reason: a turn that cannot fit is dropped, not cut.

Shape of a block:

    [BOS] user\n{q}\n[EOS] assistant\n<thinking>{r}</thinking>{a}\n[EOS]
                                              ^^^^^^ span ^^^^^^^^^^
                                                  supervised, header excluded
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import PackedData
from .tokenizer import encode

IGNORE = -100          # cross_entropy's ignore_index
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
USER_HEADER, ASSISTANT_HEADER = "user\n", "assistant\n"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    reasoning: str | None = None

    @property
    def supervised(self) -> bool:
        return self.role == "assistant"

    def header(self) -> str:
        return ASSISTANT_HEADER if self.supervised else USER_HEADER

    def body(self) -> str:
        """The supervised part: the think block then the answer (option A)."""
        if self.reasoning:
            return f"{THINK_OPEN}{self.reasoning}{THINK_CLOSE}{self.content}"
        return self.content

    def text(self) -> str:
        return f"{self.header()}{self.body()}\n"


def _turn_ids(turn: Turn, enc=encode) -> tuple[list[int], int]:
    """(ids to append, length of the supervised body).

    The header and the body are encoded **separately and appended separately**,
    never as one string. Encoding "assistant\\n答案\\n" as a whole and subtracting
    `len(encode("答案"))` does not work: BPE is free to merge across the seam, so
    the difference is not the header length, and the span lands one or two tokens
    off. That offset is invisible -- the block still decodes to readable text and
    the loss still looks like a loss -- so it is worth stating why the obvious
    implementation is wrong.

    Consequence: the ids we append are not exactly `enc(turn.text())`. They
    decode to the same characters (section 2 of the test checks that), and the
    span is exact by construction.

    `enc` is a parameter so the Qwen3 row can flatten the same SFT corpus on its
    own tokenizer track (SPEC 3.3: the two tracks' token ids are not comparable).
    """
    head = enc(turn.header())
    body = enc(turn.body())
    return head + body, len(body)


def _pack(turns: list[Turn], block_size: int, atomic: bool = True, enc=encode,
          bos: int | None = C.BOS, eos: int = C.EOS):
    """Encode turns into whole-turn blocks. Returns [(ids, spans), ...].

    `bos`/`eos` are the block opener and the turn separator, and they are
    parameters because the two tokenizer tracks number their specials completely
    differently (SPEC 3.3). On Qwen3, id 1 is `<|im_start|>` and id 2 is an
    ordinary BPE token, so the 6400 track's ids become meaningful text in a Qwen
    block. Qwen3 has no BOS (`bos_token_id is None`) so `bos=None` there; its EOS
    is `<|endoftext|>` = 151643. Turn spans are relative to the block start.
    """
    blocks: list[tuple[list[int], list[tuple[int, int]]]] = []
    cur: list[int] = [bos] if bos is not None else []
    cur_spans: list[tuple[int, int]] = []

    def flush():
        nonlocal cur, cur_spans
        if len(cur) > 1 and cur_spans:
            blocks.append((cur, cur_spans))
        cur = ([bos] if bos is not None else [])
        cur_spans = []

    for turn in turns:
        whole, body_len = _turn_ids(turn, enc)
        need = len(whole) + 1                    # the turn plus its closing EOS
        # cur already holds a BOS, so the block total is len(cur) + need. Writing
        # `len(cur) - 1 + need` here lets a block reach block_size + 1, which the
        # np.asarray in build_sft_blocks then rejects with an inhomogeneous-shape
        # error -- an off-by-one that only shows up on the real corpus.
        if len(cur) + need > block_size:
            if atomic:
                flush()
                if need + 1 > block_size:
                    continue                     # this turn alone cannot fit
            else:
                # Truncate the turn at the block edge and record the clipped
                # span. Only the test exercises this; training always uses
                # atomic=True, where a clipped span would make "assistant-only
                # loss" quietly untrue at the seam.
                room = block_size - len(cur) - 1
                if room <= 0:
                    flush()
                    continue
                start = len(cur) + (len(whole) - body_len if turn.supervised else 0)
                cur.extend(whole[:room])
                if turn.supervised and start < len(cur):
                    cur_spans.append((start, len(cur)))
                cur.append(eos)
                flush()
                continue
        start = len(cur) + (len(whole) - body_len if turn.supervised else 0)
        cur.extend(whole)
        if turn.supervised:
            cur_spans.append((start, len(cur)))
        cur.append(eos)
    flush()
    return blocks


def flatten_dialogue(turns: list[Turn], block_size: int,
                     block_atomic: bool = True) -> tuple[list[int], list[tuple[int, int]]]:
    """The first block of one dialogue -- the shape the test inspects."""
    got = _pack(turns, block_size, atomic=block_atomic)
    return got[0] if got else ([], [])


def build_sft_blocks(jsonl: Path, block_size: int, max_blocks: int = 0,
                     max_docs: int = 0, enc=encode, bos: int | None = C.BOS,
                     eos: int = C.EOS, pad_id: int = C.PAD):
    """jsonl of conversations -> (blocks [n, L] int32, spans, stats).

    `n_dropped_turns_with_thinking` is the number the reviewer should look at
    first, and it is why `reasoning_survival_rate` exists in meta.json rather than
    a bare `reasoning_in_loss: true`.

    At ctx 128 almost every message carrying `reasoning_content` is longer than a
    whole block (median 276 tokens), so turn-atomic dropping discards them: a
    5,000-dialogue sample gives **3.0% survival**. GOAL.md Q4 chose option A
    (supervise the think block); at this context length that option delivers
    almost nothing. Measured survival by block size:

        block  128   192   256   384   512
        surv  3.0% 19.5% 41.1% 82.4% 97.7%

    The user accepted 3.0% for the experimental version -- it is a smoke test, not
    a chat model. The practical version should not: its tier1.yaml is still empty,
    so giving SFT a longer context costs nothing there. ISSUES.md OPEN-7.
    """
    blocks: list[list[int]] = []
    spans: list[list[tuple[int, int]]] = []
    used: list[int] = []
    n_docs = n_dropped = n_long = 0
    n_long_think = 0
    with Path(jsonl).open() as fh:
        for line in fh:
            if max_docs and n_docs >= max_docs:
                break
            n_docs += 1
            turns = [Turn(role=m["role"], content=m["content"],
                          reasoning=m.get("reasoning_content"))
                     for m in json.loads(line)["conversations"]]
            before = len(blocks)
            for ids, sp in _pack(turns, block_size, enc=enc, bos=bos, eos=eos):
                # `max_blocks=0` means unlimited, so it has to be tested for
                # truthiness first. `len(blocks) >= 0` is true on the very first
                # block, which silently produced zero blocks -- and only for
                # callers that pass 0, which is why the 6400 track (always 4096)
                # never saw it.
                if max_blocks and len(blocks) >= max_blocks:
                    break
                blocks.append(ids + [pad_id] * (block_size - len(ids)))
                spans.append(sp)
                used.append(len(ids))
            for t in turns:
                if len(enc(t.text())) + 2 > block_size:
                    n_long += 1
                    n_long_think += bool(t.reasoning)
            if len(blocks) == before:
                n_dropped += 1
            if max_blocks and len(blocks) >= max_blocks:
                break
    arr = (np.asarray(blocks, dtype=np.int32) if blocks
           else np.zeros((0, block_size), np.int32))
    return arr, spans, {"n_docs": n_docs, "n_blocks": int(arr.shape[0]),
                        "n_dropped_dialogues": n_dropped,
                        "n_long_turns": n_long,
                        "n_dropped_turns_with_thinking": n_long_think,
                        "mean_block_fill": round(float(np.mean(used)) / block_size, 4) if used else 0.0,
                        "mean_used_tokens": round(float(np.mean(used)), 1) if used else 0.0}


def _n_think_blocks(blocks, spans) -> int:
    """How many blocks actually kept a think block. Counted from the data, not
    estimated -- the survival rate in meta.json is what OPEN-7 turns on, so it has
    to be measured from the blocks that were emitted."""
    from .tokenizer import decode

    n = 0
    for i, row in enumerate(spans):
        for a, b in row:
            if THINK_OPEN in decode(blocks[i, a:b].tolist()):
                n += 1
                break
    return n


def write_prepared(jsonl: Path, out_dir: Path, block_size: int,
                   max_blocks: int = 0, max_docs: int = 0) -> dict:
    """Materialise a prepared directory the loader can memmap.

    Spans are stored **per block** -- `[n_spans, 2]` rows of (start, end) inside
    one 128-wide block -- not as offsets into a flat stream. An earlier version
    wrote them flat and had `load_sft` divide by block_size to recover the block
    index, which silently put every span in block 0 and marked the padding as
    supervised. Spans-per-block is the shape that cannot be misread.
    """
    arr, spans, stats = build_sft_blocks(jsonl, block_size, max_blocks, max_docs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr.astype("<u2").tofile(out / "tokens.bin")
    # (block, start, end) per span. Carrying the block index makes the format
    # impossible to misread. An earlier version stored (start, end) and had the
    # loader divide by block_size to recover the index, which silently put every
    # span in block 0 and marked the padding as supervised.
    rows = [(i, a, b) for i, row in enumerate(spans) for a, b in row]
    np.save(out / "assistant_spans.npy",
            np.asarray(rows, dtype=np.int64).reshape(-1, 3))
    lengths = np.asarray([sum(b - a for a, b in row) for row in spans], dtype=np.int64)
    np.save(out / "supervised_tokens.npy", lengths)
    kept_think = _n_think_blocks(arr, spans)
    dropped_think = stats["n_dropped_turns_with_thinking"]
    survival = round(kept_think / max(kept_think + dropped_think, 1), 4)
    meta = dict(stats, vocab_size=C.VOCAB, block_size=block_size,
                n_spans=int(sum(len(r) for r in spans)),
                supervised_tokens=int(lengths.sum()),
                mean_supervised_per_block=round(float(lengths.mean()) if lengths.size else 0.0, 1),
                source=str(Path(jsonl).name),
                loss_on_assistant_only=True,
                # Option A was chosen, but at ctx 128 it keeps almost none of the
                # think content. Writing `reasoning_in_loss: true` here would be
                # the kind of field nobody re-checks; these two say what is
                # actually in the corpus.
                reasoning_option="A (fold <thinking> into the assistant text)",
                reasoning_survival_rate=survival,
                block_atomic=True)
    # Turn-atomicity buys correct spans at the cost of wasted padding: a block is
    # flushed early whenever the next turn does not fit. `mean_block_fill` is the
    # price, recorded rather than asserted away.
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class SFTData(PackedData):
    """Packed blocks plus labels; `.batch` returns both stacked as [B, 2, L].

    `Loop.fit` hands `data.batch(idx)` straight to `loss_fn(model, batch)`, so
    row 0 is input ids and row 1 is labels with IGNORE outside assistant spans.
    Doing it this way leaves the shared training loop untouched, which is what
    keeps the A/B comparison honest -- loop.py's docstring makes that point.
    """

    def __init__(self, blocks: torch.Tensor, spans: list[list[tuple[int, int]]],
                 block_size: int, prepared_dir: Path):
        super().__init__(blocks=blocks, n_docs=len(spans),
                         n_tokens=int(blocks.numel()), block_size=block_size,
                         prepared_dir=prepared_dir)
        labels = torch.full_like(blocks, IGNORE, dtype=torch.int64)
        for i, row in enumerate(spans):
            for a, b in row:
                labels[i, a:b] = blocks[i, a:b].long()
        self.labels = labels

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.blocks[idx].long(), self.labels[idx]])


def load_sft(prepared_dir: Path, block_size: int, max_blocks: int = 0) -> SFTData:
    """Read a directory written by `write_prepared`."""
    prepared = Path(prepared_dir)
    meta = json.loads((prepared / "meta.json").read_text())
    if meta.get("vocab_size") != C.VOCAB:
        raise ValueError(f"{prepared}: vocab {meta.get('vocab_size')} != {C.VOCAB}")
    tokens = np.fromfile(prepared / "tokens.bin", dtype="<u2")
    spans = np.load(prepared / "assistant_spans.npy").reshape(-1, 3)
    n_blocks = int(tokens.shape[0] // block_size)
    if meta.get("n_blocks") != n_blocks:
        raise ValueError(f"{prepared}: meta says {meta.get('n_blocks')} blocks, "
                         f"the binary holds {n_blocks}")
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)
    usable = n_blocks * block_size
    blocks = torch.from_numpy(tokens[:usable].reshape(n_blocks, block_size).copy())

    per_block: list[list[tuple[int, int]]] = [[] for _ in range(n_blocks)]
    for blk, a, b in spans:
        blk, a, b = int(blk), int(a), int(b)
        if not (0 <= a < b <= block_size):
            raise ValueError(f"{prepared}: span ({a}, {b}) does not fit a "
                             f"{block_size}-wide block")
        if blk < n_blocks:
            per_block[blk].append((a, b))
    return SFTData(blocks, per_block, block_size, prepared)