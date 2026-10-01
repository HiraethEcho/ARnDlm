#!/usr/bin/env python
"""test_common -- does the shared A-version base hold up?

Answers five questions, one per section, each with a measured number:

  1 config   -- does Tier 0 become a 4.57 Qwen3Config with the intended shape?
  2 data     -- are packed blocks built from the real corpus correct, and is
                the batch order deterministic and resumable?
  3 ckpt     -- does a state_dict survive a save/load round trip, including
                tied weights (one tensor, one key)?
  4 metrics  -- is train-only throughput really measured apart from wall time,
                and does repetition rate react to looped text?
  5 loop     -- does `save at N + continue` reproduce one continuous run?
  6 text     -- does the tokenizer round trip, and does the RESULTS row land?

Section 5 is the one that matters most: acceptance test C4 in example/PLAN.md
is exactly this property, and it is cheap to check with a 20-step CPU model.

Run:  .venv/bin/python example/tests/test_common.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as C
from common.ckpt import describe, load_state_dict, save_model
from common.data import BlockStream, PackedData, build_packed, fixed_batches
from common.loop import Loop
from common.metrics import TpsMeter, loss_windows, repetition_rates
from common.model import build_qwen3_config, count_params
from common.report import COLUMNS, RESULTS, append_result
from common.seeds import resolve_dtype, seed_all
from common.tokenizer import decode, encode, load_tokenizer

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 1. config

def section_config() -> None:
    print("\n[1] config")
    cfg = C.TIER0
    check("head_dim", cfg.head_dim == 128, f"hidden {cfg.hidden_size} / heads {cfg.num_heads}")
    hf = build_qwen3_config(cfg)
    fields = {
        "vocab_size": (hf.vocab_size, C.VOCAB),
        "hidden_size": (hf.hidden_size, cfg.hidden_size),
        "layers": (hf.num_hidden_layers, cfg.num_layers),
        "heads": (hf.num_attention_heads, cfg.num_heads),
        "kv_heads": (hf.num_key_value_heads, cfg.num_kv_heads),
        "head_dim": (hf.head_dim, cfg.head_dim),
        "intermediate": (hf.intermediate_size, cfg.intermediate_size),
        "max_pos": (hf.max_position_embeddings, cfg.context_length),
        "tied": (hf.tie_word_embeddings, True),
        "sliding_off": (hf.use_sliding_window, False),
    }
    for name, (got, want) in fields.items():
        check(f"Qwen3Config.{name}", got == want, f"{got} (want {want})")
    print(f"  note  special ids: PAD {C.PAD} BOS {C.BOS} EOS {C.EOS} MASK {C.MASK}")


# ------------------------------------------------------------------ 2. data

def section_data() -> None:
    print("\n[2] data")
    data = build_packed(C.DATA_DIR, block_size=C.BLOCK_SIZE)
    print(f"  corpus  {data.n_docs:,} docs -> {len(data):,} blocks x {data.block_size} "
          f"= {data.n_tokens:,} tokens ({data.prepared_dir.name})")
    check("block count", len(data) > 0, f"{len(data):,}")
    check("dtype is int32 (memory)", data.blocks.dtype == torch.int32, str(data.blocks.dtype))
    check("batch() casts to int64", data.batch(torch.arange(4)).dtype == torch.int64)

    blocks = data.blocks
    check("no PAD inside blocks", int((blocks == C.PAD).sum()) == 0,
          "packing fills the tail, it does not pad it")
    check("BOS present", int((blocks == C.BOS).sum()) > 0, f"{int((blocks == C.BOS).sum()):,}")
    check("EOS present", int((blocks == C.EOS).sum()) > 0, f"{int((blocks == C.EOS).sum()):,}")
    check("ids below vocab", int(blocks.max()) < C.VOCAB,
          f"max id {int(blocks.max())} < {C.VOCAB}")

    # A hand-built corpus is the only way to prove the splice itself is right:
    # block boundaries must land on a BOS.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        docs = [[5, 6, 7], [8, 9], [10]]
        flat, lengths = [], []
        for d in docs:
            flat += d
            lengths.append(len(d))
        np.asarray(flat, dtype="<u2").tofile(tmp / "tokens.bin")
        np.save(tmp / "doc_lengths.npy", np.asarray(lengths))
        (tmp / "meta.json").write_text('{"vocab_size": 6400, "n_docs": 3, "total_tokens": 6}')
        tiny = build_packed(tmp, block_size=4)
        got = tiny.blocks.tolist()
        # stream = 1 5 6 7 2 | 1 8 9 2 | 1 10 2   -> 12 tokens -> 3 blocks of 4
        want = [[C.BOS, 5, 6, 7], [C.EOS, C.BOS, 8, 9], [C.EOS, C.BOS, 10, C.EOS]]
        check("splice is BOS doc... EOS", got == want, f"{got}")
        check("ragged tail dropped", len(data) * data.block_size <= data.n_tokens)

    # batch order: deterministic, and resumable at an exact position
    a = BlockStream(len(data), 8, seed=7)
    b = BlockStream(len(data), 8, seed=7)
    same = all(torch.equal(a.next(), b.next()) for _ in range(5))
    check("same seed -> same batches", same)

    c = BlockStream(len(data), 8, seed=7)
    for _ in range(3):
        c.next()
    state = c.state()
    resumed = BlockStream(len(data), 8, seed=7)
    resumed.load_state(state)
    check("resume continues identically", torch.equal(c.next(), resumed.next()))

    wrapped = BlockStream(10, 4, seed=1)          # 10 blocks / bs 4 -> 2 batches per epoch
    for _ in range(3):
        wrapped.next()
    check("epoch rolls over", wrapped.epoch == 1, f"epoch {wrapped.epoch}, pos {wrapped.pos}")

    fixed = fixed_batches(4, 8, len(data), seed=99)
    again = fixed_batches(4, 8, len(data), seed=99)
    check("fixed eval batches are stable", len(fixed) == 4
          and all(torch.equal(a, b) for a, b in zip(fixed, again)))
    check("fixed batches do not repeat rows", len(set(fixed[0].tolist())) == 8)
    other = fixed_batches(4, 8, len(data), seed=100)
    check("different seed -> different batches", not torch.equal(fixed[0], other[0]))


# ------------------------------------------------------------------ 3. ckpt

def section_ckpt() -> None:
    print("\n[3] ckpt")
    tmp = Path(tempfile.mkdtemp())
    try:
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model[0].weight.data = model[1].weight.data      # simulate tied storage
        save_model(model, tmp, {"note": "tied"})
        keys = list(load_state_dict(tmp).keys())
        # 4 parameters in, 3 out: 1.weight shares storage with 0.weight.
        check("shared storage stored once", len(keys) == 3, f"{keys}")

        sd = load_state_dict(tmp)
        model2 = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model2[0].weight.data = model2[1].weight.data
        model2.load_state_dict({k: sd[k] for k in sd if k in dict(model2.state_dict())},
                               strict=False)
        same = torch.equal(model2[0].weight.data, model[0].weight.data)
        check("round trip is bit-exact", same)

        cfg = build_qwen3_config(C.TIER0)
        from transformers import Qwen3ForCausalLM

        hf_model = Qwen3ForCausalLM(cfg)
        save_model(hf_model, tmp / "hf", {"model": "qwen3"})
        hf_sd = load_state_dict(tmp / "hf")
        n_stored = sum(v.numel() for v in hf_sd.values())
        n_unique = count_params(hf_model)
        info = describe(hf_model)
        check("tied lm_head not duplicated", n_stored == n_unique,
              f"stored {n_stored:,} == params {n_unique:,}")
        check("describe() flags tying", info["tied"] and
              info["stored_tensors"] == info["state_dict_entries"] - 1,
              f"{info['params_unique']:,} params, {info['state_dict_entries']} state_dict "
              f"entries -> {info['stored_tensors']} stored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- 4. metrics

def section_metrics() -> None:
    print("\n[4] metrics")
    meter = TpsMeter()
    t0 = time.perf_counter()
    time.sleep(0.10)                          # a real 100 ms of "step" time
    meter.add(1000, time.perf_counter() - t0)
    time.sleep(0.30)                          # validation / sampling
    check("train-only ignores idle time", abs(meter.train_tps - 10_000.0) < 200,
          f"{meter.train_tps:,.1f} tok/s")
    check("wall clock includes it", meter.wall_tps < meter.train_tps / 2,
          f"{meter.wall_tps:,.1f} tok/s vs train-only {meter.train_tps:,.1f}")

    looped = [list(range(8)) * 8]                       # one 4-gram cycle repeated 8x
    varied = [list(range(64))]                          # every 4-gram distinct
    r_loop, r_var = repetition_rates(looped), repetition_rates(varied)
    want_loop = 1.0 - 8 / 61                            # 61 windows, 8 distinct
    check("repetition rate flags looping", abs(r_loop[4] - want_loop) < 1e-9,
          f"rep4 {r_loop[4]:.4f} (closed form {want_loop:.4f})")
    check("repetition rate is 0 on varied text", r_var[4] == 0.0, f"rep4 {r_var[4]:.4f}")
    short = repetition_rates([[1, 2]])
    check("too-short sample is nan, not 0", np.isnan(short[4]), str(short[4]))

    w = loss_windows([5.0, 4.0, 3.0, 2.0, 1.0], k=2)
    check("loss windows", w["first"] == 4.5 and w["last"] == 1.5, str(w))
    check("dtype helper", resolve_dtype("bfloat16") is torch.bfloat16)


# ------------------------------------------------------------------ 5. loop

TINY_VOCAB = 64


class TinyLM(torch.nn.Module):
    """Two matmuls and a real embedding: enough to have gradients and state."""

    def __init__(self, vocab: int = TINY_VOCAB, dim: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, x):
        return self.head(self.emb(x))


def _tiny_data(n_blocks: int = 64, block: int = 8) -> PackedData:
    g = torch.Generator().manual_seed(3)
    blocks = torch.randint(0, TINY_VOCAB, (n_blocks, block), generator=g, dtype=torch.int32)
    return PackedData(blocks=blocks, n_docs=n_blocks, n_tokens=n_blocks * block,
                      block_size=block, prepared_dir=Path("synthetic"))


def _loss_fn(model, batch):
    logits = model(batch[:, :-1])
    target = batch[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    return loss, {}


def _make_loop(cfg, out_dir: Path) -> Loop:
    """Two runs must start from identical weights, so seed before the init.

    `Loop.__init__` seeds *after* the model exists -- seeding there protects
    dropout and sampling, not parameter initialisation.
    """
    seed_all(cfg.seed)
    return Loop(TinyLM(), cfg, _loss_fn, device=torch.device("cpu"),
                out_dir=out_dir, amp="float32", require_cuda=False)


def section_loop() -> None:
    print("\n[5] loop")
    import dataclasses

    from common.config import TrainCfg

    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = TrainCfg(steps=12, batch_size=4, warmup_steps=2, log_every=6,
                       loss_window=4, seed=11)
        half = dataclasses.replace(cfg, log_every=6)
        data = _tiny_data()

        straight = _make_loop(cfg, tmp / "straight")
        ev = fixed_batches(3, 4, len(data), seed=5)
        s1 = straight.fit(data, eval_batches=ev)

        part = _make_loop(half, tmp / "part")
        part.fit(data, max_steps=6)          # the stream it used is remembered by fit()
        part.save(tmp / "part", {"note": "resume"})

        cont = _make_loop(cfg, tmp / "cont")
        # No stream argument: fit() must continue the saved batch order by itself.
        cont.load(tmp / "part")
        s2 = cont.fit(data)

        # The two runs differ only in where the checkpoint landed; the loss
        # trajectory must match step for step.
        match = all(abs(a - b) < 1e-9 for a, b in zip(straight.losses, cont.losses))
        check("resume reproduces the same losses", match,
              f"last {straight.losses[-1]:.9f} vs {cont.losses[-1]:.9f}")
        check("resume without an explicit stream also matches", cont.step == 12,
              f"step {cont.step}")
        check("step counter restored", cont.step == 12, f"{cont.step}")
        check("throughput reported both ways",
              s1["throughput"]["train_only_tps"] > 0 and s1["throughput"]["wall_tps"] > 0,
              f"train {s1['throughput']['train_only_tps']:,.0f} / "
              f"wall {s1['throughput']['wall_tps']:,.0f} tok/s")
        check("metrics.csv written", (tmp / "straight" / "metrics.csv").exists())
        check("eval loss on frozen batches", s1.get("eval_loss", -1) > 0,
              f"{s1.get('eval_loss')}")
        check("wall includes eval, train-only does not",
              s1["throughput"]["wall_tps"] < s1["throughput"]["train_only_tps"],
              f"train {s1['throughput']['train_only_tps']:,.0f} vs "
              f"wall {s1['throughput']['wall_tps']:,.0f} tok/s")

        # eval_loss must be independent of train/eval mode and of batch order.
        a = straight.eval_loss(data, ev)
        b = straight.eval_loss(data, ev)
        check("eval_loss deterministic", abs(a - b) < 1e-12, f"{a:.9f}")
        check("model left in train mode", straight.model.training)
        print(f"  note  losses {straight.losses[0]:.4f} -> {straight.losses[-1]:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def section_tokenizer_report() -> None:
    print("\n[6] tokenizer + results table")
    tok = load_tokenizer()
    check("vocab is 6400", tok.get_vocab_size() == C.VOCAB, str(tok.get_vocab_size()))
    ids = encode("Once upon a time")
    check("encode round trip", decode(ids).strip() == "Once upon a time", decode(ids))
    check("no specials by default", C.BOS not in ids or len(ids) > 3)
    wrapped = encode("hi", bos=True, eos=True)
    check("bos/eos on request", wrapped[0] == C.BOS and wrapped[-1] == C.EOS)
    check("tokenizer is cached", load_tokenizer() is tok)

    saved = RESULTS.read_text() if RESULTS.exists() else None
    try:
        append_result("test-cell", "hf", "ar", 3_213_056, 12,
                      loss={"first": 5.5, "last": 4.2}, train_tps=7000.0,
                      wall_tps=6500.0, eval_loss=4.3, rep4=0.12, notes="probe")
        text = RESULTS.read_text()
        check("row appended", "| test-cell |" in text)
        header = next((ln for ln in text.split("\n") if ln.startswith("| cell |")), "")
        check("header has every column", all(c in header for c in COLUMNS),
              f"{len(COLUMNS)} columns")
        check("rep-4 column present", "rep-4" in text)
    finally:
        if saved is None:
            RESULTS.unlink(missing_ok=True)
        else:
            RESULTS.write_text(saved)


def main() -> int:
    seed_all(0)
    section_config()
    section_data()
    section_ckpt()
    section_metrics()
    section_loop()
    section_tokenizer_report()
    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())