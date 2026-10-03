#!/usr/bin/env python
"""test_sft_flatten -- does the SFT flattener produce blocks whose assistant
spans actually cover the assistant text?

The whole SFT column rests on one claim: **loss is computed on assistant tokens
only** (GOAL.md Q4). If the spans are off by one, or cover the user's half of a
turn, the model trains on the wrong objective and every number downstream is
confidently wrong. Nothing about that failure looks like a bug.

So the checks here are written against *decoded text*, not against indices:

  1 schema     -- the real file's role/content/reasoning_content shape, and the
                  three cases that silently drop content if mishandled
  2 flatten    -- a block decodes back to a readable dialogue, in order
  3 spans      -- every assistant span decodes to assistant text, and *only*
                  assistant text; user text is never inside a span
  4 boundaries -- spans never contain the role header, never cross a block
                  edge, and never contain PAD
  5 loss mask  -- a CE computed over the labels equals CE over the masked
                  positions and nothing else (this is the property the training
                  loop will rely on)
  6 packing    -- over-length turns are dropped whole, never cut mid-span
  7 coverage   -- the 19% of messages that carry reasoning_content survive
                  flattening, and no system message is invented

Run:  .venv/bin/python example/tests/test_sft_flatten.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.data import PackedData
from common.sft import (SFTData, Turn, flatten_dialogue, load_sft,
                        write_prepared)
from common.tokenizer import decode, encode
FAILS: list[str] = []
IGNORE = -100


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------- 1. schema

def section_schema() -> list[dict]:
    print("\n[1] schema of the real file")
    raw = C.ROOT / "data" / "raw" / "minimind_dataset" / "sft_t2t_mini.jsonl"
    check("corpus exists", raw.exists(), str(raw))
    if not raw.exists():
        return []

    rows = []
    with raw.open() as fh:
        for i, line in enumerate(fh):
            if i >= 300:
                break
            rows.append(json.loads(line))

    check("top level is a list of conversations",
          all(isinstance(r.get("conversations"), list) for r in rows))
    check("no 'text' key anywhere (that is why prepare_data.py yields 0 tokens)",
          all("text" not in r for r in rows))

    roles = {m["role"] for r in rows for m in r["conversations"]}
    check("only user/assistant roles", roles <= {"user", "assistant"}, str(sorted(roles)))
    check("every dialogue starts with user",
          all(r["conversations"][0]["role"] == "user" for r in rows))

    n_assistant = sum(1 for r in rows for m in r["conversations"]
                      if m["role"] == "assistant")
    n_think = sum(1 for r in rows for m in r["conversations"]
                  if m.get("reasoning_content"))
    check("reasoning_content present", n_think > 0,
          f"{n_think}/{n_assistant} assistant messages = {100*n_think/max(n_assistant,1):.0f}%")
    return rows


# -------------------------------------------------------------- 2. flatten

def section_flatten(rows: list[dict]) -> None:
    print("\n[2] flatten")
    # A two-turn dialogue fits in one 128-token block. The real corpus is 14
    # turns long, so section 8 is what checks the multi-block path -- checking a
    # whole dialogue against one block would be checking the wrong thing.
    turns = [Turn(role="user", content="你是谁", reasoning=None),
             Turn(role="assistant", content="我是小模型", reasoning=None)]
    ids, spans = flatten_dialogue(turns, C.BLOCK_SIZE)
    text = decode(ids)

    check("block decodes to something readable", "assistant" in text and "user" in text,
          repr(text))
    check("no newlines doubled", "\n\n\n" not in text)
    check("starts with BOS, ends with EOS",
          ids[0] == C.BOS and ids[-1] == C.EOS, f"{ids[0]} .. {ids[-1]}")
    check("fits the block", len(ids) <= C.BLOCK_SIZE, f"{len(ids)} / {C.BLOCK_SIZE}")
    check("turn order preserved", text.find("user") < text.find("assistant"))

    # Header and body are encoded separately (see sft._turn_ids). The rendered
    # characters must still be identical to the turn's text.
    t = turns[1]
    check("separate header/body encoding renders the same text",
          decode(encode(t.header()) + encode(t.body())).strip() == t.text().strip(),
          repr(decode(encode(t.header()) + encode(t.body()))))

    missing = [t.content for t in turns if t.content not in text]
    check("every utterance survives", not missing, str(missing))

    # The rendered form looks ChatML because BOS/EOS *are* <|im_start|>/<|im_end|>
    # in this vocabulary (ids 1/2). The role itself is plain text -- there is no
    # role token, which is what SPEC 3.3 forces by refusing to grow the vocab.
    check("BOS renders as <|im_start|>", decode([C.BOS]) == "<|im_start|>",
          repr(decode([C.BOS])))
    check("role identity is plain text, not a token",
          C.USER_HEADER_ID is None if hasattr(C, "USER_HEADER_ID") else True)


# ---------------------------------------------------------------- 3. spans

def section_spans() -> None:
    print("\n[3] spans cover assistant text and nothing else")
    turns = [Turn(role="user", content="你是谁", reasoning=None),
             Turn(role="assistant", content="我是小模型", reasoning=None),
             Turn(role="user", content="你会什么", reasoning=None),
             Turn(role="assistant", content="我会聊天", reasoning=None)]
    ids, spans = flatten_dialogue(turns, C.BLOCK_SIZE)
    text = decode(ids)

    check("at least one span", len(spans) >= 1, f"{len(spans)} spans")
    check("no user header inside a span",
          not any(decode(ids[a:b]).startswith("user") for a, b in spans))

    assistant_contents = [t.content for t in turns if t.supervised]
    covered = "".join(decode(ids[a:b]) for a, b in spans)
    missing = [c for c in assistant_contents if c not in covered]
    check("every assistant answer is inside some span", not missing, str(missing[:2]))

    # The real trap: user text leaking into the loss.
    user_contents = [t.content for t in turns if not t.supervised]
    leaked = [u for u in user_contents if u in covered]
    check("no user text inside any span", not leaked, str(leaked[:2]))

    check("spans sorted and disjoint",
          all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)),
          str(spans))


# ------------------------------------------------------------ 4. boundaries

def section_boundaries() -> None:
    print("\n[4] span boundaries")
    turns = [Turn(role="user", content="你是谁", reasoning=None),
             Turn(role="assistant", content="我是小模型", reasoning="他在想", ),
             Turn(role="user", content="你会什么", reasoning=None),
             Turn(role="assistant", content="我会聊天", reasoning=None)]
    ids, spans = flatten_dialogue(turns, C.BLOCK_SIZE)

    check("no span starts at 0 (BOS must stay out of the loss)",
          all(a > 0 for a, _ in spans))
    check("no span contains BOS",
          all(C.BOS not in ids[a:b] for a, b in spans))
    check("no span contains PAD",
          all(C.PAD not in ids[a:b] for a, b in spans))
    check("no span contains the MASK id",
          all(C.MASK not in ids[a:b] for a, b in spans),
          "a mask token inside a supervised span is the A2D marker leaking in")

    # These two used to assert the opposite, which is how DONE-12a survived a
    # review: the corpus put the assistant's EOS *outside* the span, and the
    # tests below were written to match. The model then never learned to stop --
    # 64 generated tokens, zero EOS. The terminator is the one token that
    # teaches "stop", so it has to be supervised.
    check("every supervised span ends with EOS",
          all(ids[b - 1] == C.EOS for _, b in spans),
          "an EOS-terminated answer is the only place the model learns to stop")
    check("EOS appears exactly once per span, at the end",
          all([i for i, t in enumerate(ids[a:b]) if t == C.EOS] == [b - a - 1]
              for a, b in spans),
          "a second EOS inside the span means the turn was cut")
    check("the last span may reach the block's last token",
          spans[-1][1] <= len(ids),
          f"last span ends at {spans[-1][1]}, block is {len(ids)} tokens")


# ------------------------------------------------------------- 5. loss mask

def section_loss_mask() -> None:
    print("\n[5] the label tensor masks exactly the spans")
    turns = [Turn(role="user", content="你是谁", reasoning=None),
             Turn(role="assistant", content="我是小模型", reasoning=None),
             Turn(role="user", content="你会什么", reasoning=None),
             Turn(role="assistant", content="我会聊天", reasoning=None)]
    ids, spans = flatten_dialogue(turns, C.BLOCK_SIZE)

    labels = np.full(len(ids), IGNORE, dtype=np.int64)
    for a, b in spans:
        labels[a:b] = ids[a:b]
    check("supervised tokens are exactly the spans",
          int((labels != IGNORE).sum()) == sum(b - a for a, b in spans))
    check("BOS is ignored", labels[0] == IGNORE, f"labels[0]={labels[0]}")

    # The property the loop depends on: CE over `labels` == CE over exactly the
    # rows the spans name, and nothing else contributes.
    torch.manual_seed(0)
    logits = torch.randn(len(ids), C.VOCAB)
    full = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels),
                                             ignore_index=IGNORE)
    rows = torch.cat([torch.arange(a, b, dtype=torch.long) for a, b in spans])
    targets = torch.tensor([ids[i] for i in rows], dtype=torch.long)
    mean = torch.nn.functional.cross_entropy(logits[rows], targets)
    check("CE(labels) == CE(span rows only)", abs(float(full) - float(mean)) < 1e-5,
          f"{float(full):.6f} vs {float(mean):.6f}")
    check("span rows cover exactly the supervised positions",
          int((labels != IGNORE).sum()) == int(rows.numel()))

    # The failure this whole file exists to prevent: if the user half were
    # supervised too, the loss would not change shape but would be measuring the
    # wrong objective. Compare against a deliberately all-supervised label row.
    naive = torch.tensor(ids, dtype=torch.long)
    naive_loss = float(torch.nn.functional.cross_entropy(logits, naive))
    check("assistant-only loss differs from all-token loss",
          abs(naive_loss - float(full)) > 1e-3,
          f"assistant-only {float(full):.4f} vs all-token {naive_loss:.4f}")

    # Documented hazard, not a bug in this module: PyTorch returns **nan** (not
    # 0) when every target is ignored. reduction='mean' divides by the number of
    # non-ignored targets, so one all-ignored row inside a batch is harmless --
    # but a whole unsupervised *batch* poisons the run. `block_atomic=True`
    # guarantees every emitted block carries at least one span, and
    # common.sft_loss guards the remaining case. This assertion pins the
    # behaviour so the guard cannot be deleted by someone who "fixes" the test.
    all_ignored = torch.full((len(ids),), IGNORE, dtype=torch.long)
    zero = float(torch.nn.functional.cross_entropy(logits, all_ignored, ignore_index=IGNORE))
    check("PyTorch returns nan for a fully-ignored batch (guard required)",
          np.isnan(zero), f"got {zero}")


# -------------------------------------------------------------- 6. packing

def section_packing() -> None:
    print("\n[6] over-length turns are dropped whole")
    long_turn = Turn(role="user", content="字" * 400, reasoning=None)
    turns = [long_turn,
             Turn(role="user", content="短问题", reasoning=None),
             Turn(role="assistant", content="短答案", reasoning=None)]
    ids, spans = flatten_dialogue(turns, C.BLOCK_SIZE, block_atomic=True)
    text = decode(ids)
    check("the 400-char turn is gone", "字" not in text)
    check("the short exchange survives", "短问题" in text and "短答案" in text)
    check("spans still point at the answer",
          len(spans) == 1 and "短答案" in decode(ids[spans[0][0]:spans[0][1]]))

    ids2, spans2 = flatten_dialogue(turns, C.BLOCK_SIZE, block_atomic=False)
    check("an all-user truncated block is dropped (no supervision, no gradient)",
          "字" not in decode(ids2), f"{len(ids2)} ids, first block was discarded")

    # The dangerous clipped case is a *supervised* turn cut at the block edge:
    # the span then runs to the edge and the answer is half-missing. This is why
    # training uses atomic=True.
    long_answer = [Turn(role="user", content="短问题", reasoning=None),
                   Turn(role="assistant", content="答" * 400, reasoning=None)]
    ids3, spans3 = flatten_dialogue(long_answer, C.BLOCK_SIZE, block_atomic=False)
    check("non-atomic clips a supervised turn at the edge",
          bool(spans3) and any(b >= len(ids3) - 1 for _, b in spans3),
          f"spans={spans3} len={len(ids3)}")
    ids4, spans4 = flatten_dialogue(long_answer, C.BLOCK_SIZE, block_atomic=True)
    check("atomic mode refuses to clip it instead",
          not (spans4 and any(b >= len(ids4) - 1 for _, b in spans4)),
          f"spans={spans4}")

    section_long_run()

    only_long = [long_turn,
                 Turn(role="assistant", content="答", reasoning=None)]
    empty, no_spans = flatten_dialogue(only_long, C.BLOCK_SIZE)
    check("a block is emitted only if it carries supervision",
          bool(empty) == bool(no_spans), f"{len(empty)} ids, {len(no_spans)} spans")


def section_real_corpus(rows: list[dict]) -> None:
    print("\n[7] the real corpus packs into whole-turn blocks")
    conv = rows[0]["conversations"]
    turns = [Turn(role=m["role"], content=m["content"],
                  reasoning=m.get("reasoning_content")) for m in conv]
    from common.sft import _pack
    blocks = _pack(turns, C.BLOCK_SIZE)
    check("a 14-turn dialogue needs more than one block", len(blocks) > 1,
          f"{len(blocks)} blocks")
    # Turn-atomicity means a block is flushed early whenever the next turn does
    # not fit -- a ~300-character think block against a 128-token window does
    # not. Underfull blocks are the *price* of correct spans, measured as
    # meta.json's mean_block_fill rather than asserted away.
    fills = [round(len(i) / C.BLOCK_SIZE, 2) for i, _ in blocks]
    check("no block exceeds the block size", all(len(i) <= C.BLOCK_SIZE for i, _ in blocks),
          f"max {max(len(i) for i, _ in blocks)} / {C.BLOCK_SIZE}")
    check("no block is a stub (all carry a span)", all(sp for _, sp in blocks),
          f"fills {fills}")
    check("fill rate is measured, not assumed", True,
          f"fills {fills} -> mean {sum(fills)/len(fills):.2f}")
    check("every block ends with EOS", all(ids[-1] == C.EOS for ids, _ in blocks))
    check("every block is supervised", all(sp for _, sp in blocks))

    # Turn-atomicity: no block may contain a user header with no answer after it,
    # because that block would train on context only.
    for i, (ids, sp) in enumerate(blocks):
        text = decode(ids)
        n_asst = text.count("assistant\n")
        check(f"block {i}: {n_asst} answer(s) for its spans", n_asst >= len(sp),
              f"spans={len(sp)}")


# ------------------------------------------------------------- 7. coverage

def section_long_run() -> None:
    print("\n[6c] a long run of real dialogues never overflows a block")
    from common.sft import build_sft_blocks
    src = C.ROOT / "data" / "raw" / "minimind_dataset" / "sft_t2t_mini.jsonl"
    arr, spans, stats = build_sft_blocks(src, C.BLOCK_SIZE, max_blocks=400)
    check("blocks are rectangular", arr.ndim == 2 and arr.shape[1] == C.BLOCK_SIZE,
          str(arr.shape))
    check("every span is inside its block",
          all(0 <= a < b <= C.BLOCK_SIZE for row in spans for a, b in row),
          f"{sum(len(r) for r in spans)} spans")
    check("no PAD inside a supervised span",
          all(int(arr[i, a:b].min()) > 0 for i, row in enumerate(spans) for a, b in row))
    check("stats report the think-block cost",
          stats["n_dropped_turns_with_thinking"] > 0,
          f"{stats['n_dropped_turns_with_thinking']} of {stats['n_long_turns']} "
          f"over-length turns carried thinking")
    check("mean_block_fill is a fill ratio, not a span count",
          0.3 <= stats["mean_block_fill"] <= 1.0, str(stats["mean_block_fill"]))


def section_coverage(rows: list[dict]) -> None:
    print("\n[7] reasoning_content -- option A, and what ctx 128 costs")
    from common.sft import THINK_CLOSE, THINK_OPEN, _pack
    with_think = next((r for r in rows
                       if any(m.get("reasoning_content") for m in r["conversations"])), None)
    check("found a dialogue with thinking", with_think is not None)
    if with_think is None:
        return
    msg = next(m for m in with_think["conversations"] if m.get("reasoning_content"))
    turns = [Turn(role=m["role"], content=m["content"],
                  reasoning=m.get("reasoning_content"))
             for m in with_think["conversations"]]

    # Option A is implemented: at a block size that fits, the think block is
    # inside a supervised span. Checked at 512 because Tier 0's ctx 128 cannot
    # hold a ~300-token think block -- see the next check.
    big = _pack(turns, 512)
    covered_big = "".join(decode(b[a:e]) for b, sp in big for a, e in sp)
    check("option A: think block is supervised when it fits",
          msg["reasoning_content"][:40] in covered_big)
    rendered_big = "".join(decode(b) for b, _ in big)
    check("think block is delimited",
          THINK_OPEN in rendered_big and THINK_CLOSE in rendered_big)
    check("no system role was invented", "system" not in rendered_big)

    # And the cost, stated rather than hidden: at ctx 128 that same turn does not
    # fit, so turn-atomic dropping discards it. Option A delivers nothing at this
    # context length. ISSUES.md OPEN-7 tracks the decision.
    small = _pack(turns, C.BLOCK_SIZE)
    covered_small = "".join(decode(b[a:e]) for b, sp in small for a, e in sp)
    check("at ctx 128 the think block is dropped (known cost, OPEN-7)",
          msg["reasoning_content"][:40] not in covered_small,
          "if this ever passes, the block size changed and OPEN-7 is stale")
    think_turn = Turn(role="assistant", content=msg["content"],
                      reasoning=msg["reasoning_content"])
    check("that turn is genuinely longer than the block",
          len(encode(think_turn.text())) + 2 > C.BLOCK_SIZE,
          f"{len(encode(think_turn.text()))+2} > {C.BLOCK_SIZE}")


def section_roundtrip() -> None:
    print("\n[8] write_prepared -> load_sft round trip")
    import tempfile
    from common.sft import load_sft, write_prepared
    src = C.ROOT / "data" / "raw" / "minimind_dataset" / "sft_t2t_mini.jsonl"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sft"
        meta = write_prepared(src, out, C.BLOCK_SIZE, max_blocks=200)
        check("meta reports the block count", meta["n_blocks"] == 200, str(meta["n_blocks"]))
        sft = load_sft(out, C.BLOCK_SIZE)

        check("loader agrees on the block count", len(sft) == 200, str(len(sft)))
        lab = sft.labels
        n_sup = int((lab != IGNORE).sum())
        check("supervised token count survives the round trip",
              n_sup == meta["supervised_tokens"],
              f"{n_sup} vs {meta['supervised_tokens']}")

        # The bug this section exists for: spans read back into the wrong block
        # make the padding look supervised, and a loss over padding is a loss
        # over nothing. Padding shows up as PAD in `ids`; its label is IGNORE by
        # construction, so the check has to look at ids to have teeth.
        ids_all = sft.blocks
        n_pad = int((ids_all == C.PAD).sum())
        both = int(((ids_all == C.PAD) & (lab != IGNORE)).sum())
        check("padding is never supervised", both == 0, f"{both} positions")
        check("the corpus really does contain padding (so that check has teeth)",
              n_pad > 1000, f"{n_pad} PAD positions")
        check("every block has at least one unsupervised position",
              bool((lab == IGNORE).any(dim=1).all()))

        # Channel-first: batch[0] is ids, batch[1] is labels, both [B, L].
        batch = sft.batch(torch.arange(4))
        check("batch is [2, B, L]", tuple(batch.shape) == (2, 4, C.BLOCK_SIZE),
              str(tuple(batch.shape)))
        ids, labels = batch[0][0], batch[1][0]
        sup = labels != IGNORE
        check("labels equal ids where supervised", bool((ids[sup] == labels[sup]).all()))
        text = decode(ids[sup].tolist())
        check("supervised text is assistant content only",
              "assistant" not in text and "user" not in text, repr(text[:60]))


def section_sampler_prefix() -> None:
    """The sampler must continue the string the corpus actually contains.

    DONE-12b: `sample_sft` built its prefix with `f"user\\n{p}\\nassistant\\n"`,
    which puts a newline where the corpus has EOS. Same token count, different
    tokens, no error -- and every sampled text and rep-4 number then described a
    prefix the model was never trained on.
    """
    print("\n[7] sampler prefix matches the corpus, token for token")
    from common.sft import encode_prompt

    for q in ("1+1等于几", "什么是深度学习？"):
        naive = encode(f"user\n{q}\nassistant\n")
        proper = encode_prompt(q)
        check(f"encode_prompt != f-string form: {q!r}", naive != proper,
              "if these are equal the test below cannot distinguish them")
        check(f"prompt ends with the assistant header: {q!r}",
              proper[-len(encode("assistant\n")):] == encode("assistant\n"))
        check(f"exactly one EOS, one token before the header: {q!r}",
              [i for i, t in enumerate(proper) if t == C.EOS]
              == [len(proper) - len(encode("assistant\n")) - 1],
              "the corpus separates turns with EOS, not a newline")

    # The decisive one: the same question, taken from a real block, must have its
    # prefix reproduced exactly.
    from common.cli import SFT_DATA_DIR

    meta_path = Path(SFT_DATA_DIR) / "tokens.bin"
    if not meta_path.exists():
        print("  (skipping the corpus check: no prepared SFT data)")
        return
    tok = np.memmap(meta_path, dtype="<u2", mode="r")
    spans = np.load(meta_path.parent / "assistant_spans.npy")
    blk, start, end = int(spans[0][0]), int(spans[0][1]), int(spans[0][2])
    base = blk * C.BLOCK_SIZE
    prefix = tok[base: base + start].tolist()
    answer = decode(tok[base + start: base + end].tolist())
    check("real block: span ends with EOS", int(tok[base + end - 1]) == C.EOS)

    # Slice by token, never by decoded text: `decode` expands <|im_start|> into
    # literal characters, so recovering the question by string surgery gives
    # back a different string than the one that was encoded.
    check("real block: prefix opens with BOS", prefix[0] == C.BOS)
    check("real block: the turn is closed by exactly one EOS before the header",
          [i for i, t in enumerate(prefix) if t == C.EOS] == [len(prefix) - 5],
          f"EOS at {[i for i, t in enumerate(prefix) if t == C.EOS]}, "
          f"prefix is {len(prefix)} tokens")
    header = encode("assistant\n")
    question = decode(prefix[1:len(prefix) - 1 - len(header)])
    check("real block: encode_prompt reproduces the prefix",
          encode_prompt(question) == prefix[1:],
          f"corpus {prefix[1:8]}... vs built {encode_prompt(question)[:7]}...")
    check("real block: the supervised answer is non-empty", len(answer.strip()) > 0)


def main() -> int:
    rows = section_schema()
    if rows:
        section_flatten(rows)
        section_spans()
        section_boundaries()
        section_loss_mask()
        section_coverage(rows)
        section_real_corpus(rows)
    section_packing()
    section_roundtrip()
    section_sampler_prefix()
    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())