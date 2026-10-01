#!/usr/bin/env python
"""test_lora_0p6b -- row 3, before its 6-minute runs are trusted.

  1 tokenizer -- the mask token exists and is the id dllm will substitute
  2 lora      -- what is trainable, and what is frozen (notes/03 §10 claims
                 attention is what changes; the flag can check that claim)
  3 parity    -- dllm's loss on the LoRA model matches the hand-written one (C2
                 on the Qwen track)
  4 sampling  -- the decoder fills the block and keeps the prompt

Model loading is the slow part (~20 s), so the sections share one instance.

Run:  .venv/bin/python example/tests/test_lora_0p6b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example"))

from common.a2d import a2d_loss, corrupt, sample_t
from base.hf.a2d_dllm import DLLMLoss
from base.qwen3_0p6b.common_qwen import (CELL_DLLM, CELL_MINIMIND, MASK_TOKEN,
                                          build_a2d_lora, build_blocks,
                                          special_ids, trainable_report)
from base.qwen3_0p6b.lora_common import prompts_for, sample_lora_dlm

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("\n[1] tokenizer")
    model, tokenizer = build_a2d_lora(seed=42)
    ids = special_ids(tokenizer)
    check("mask token was added", ids["mask"] == 151669,
          f"{MASK_TOKEN} -> {ids['mask']}")
    check("vocab grew by one", ids["vocab"] == 151670, f"{ids['vocab']}")
    check("Qwen3 has no BOS", ids["bos"] is None, str(ids["bos"]))
    check("embedding matches the tokenizer",
          model.get_input_embeddings().weight.shape[0] == ids["vocab"],
          str(tuple(model.get_input_embeddings().weight.shape)))

    print("\n[2] LoRA scope")
    report = trainable_report(model)
    print(f"  note  {report}")
    check("under 1% trainable", report["ratio_pct"] < 1.0, f"{report['ratio_pct']}%")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    check("only LoRA adapters train",
          all("lora_" in n for n in trainable_names),
          f"{len(trainable_names)} trainable tensors")
    check("attention projections targeted",
          any("q_proj" in n for n in trainable_names)
          and any("k_proj" in n for n in trainable_names)
          and any("v_proj" in n for n in trainable_names)
          and any("o_proj" in n for n in trainable_names))
    check("FFN frozen", not any(".mlp." in n for n in trainable_names))

    print("\n[3] data + loss parity")
    blocks = build_blocks(tokenizer, n_docs=64, block_size=128)
    check("blocks are 128-wide", blocks.shape[1] == 128, str(tuple(blocks.shape)))
    check("ids inside the tokenizer", int(blocks.max()) < ids["vocab"],
          f"max {int(blocks.max())}")
    batch = blocks[:2].long().to(next(model.parameters()).device)

    eps = 1e-3
    torch.manual_seed(13)
    t = sample_t(2, batch.device, epsilon=eps)
    x_t, masked = corrupt(batch, t, ids["mask"])
    from base.qwen3_0p6b.lora_common import forward_fn

    with torch.no_grad():
        ours = a2d_loss(forward_fn(model)(x_t), batch, masked, t)

    harness = DLLMLoss(special_ids={"mask_token_id": ids["mask"],
                                    "eos_token_id": ids["eos"],
                                    "bos_token_id": ids["eos"]})
    torch.manual_seed(13)
    theirs = harness(model, batch).detach()
    rel = abs(float(ours) - float(theirs)) / max(abs(float(theirs)), 1e-9)
    check("hand-written vs dllm loss within 1%", rel < 0.01,
          f"ours {float(ours):.4f} / dllm {float(theirs):.4f}  rel {rel:.2e}")

    print("\n[4] sampling")
    out = sample_lora_dlm(model, tokenizer, prompts_for(tokenizer), 16,
                          mask_id=ids["mask"], block_size=8, steps=8, temperature=0.0)
    check("two prompts", len(out["ids"]) == 2)
    check("ids in range", all(0 <= t < ids["vocab"] for s in out["ids"] for t in s))
    check("no mask left", all(ids["mask"] not in s for s in out["ids"]))
    check("text decodes", all(isinstance(t, str) for t in out["texts"]))

    print("\n[5] checkpoints")
    from common.config import cell_dir

    for cell in (CELL_DLLM, CELL_MINIMIND):
        path = cell_dir(f"qwen3-0.6b/{cell.removeprefix('qwen3-0.6b-')}")
        if not (path / "model.safetensors").exists():
            print(f"  SKIP  {cell} has not been run")
        else:
            from common.ckpt import load_state_dict

            check(f"{cell} saved", len(load_state_dict(path)) > 0,
                  f"{len(load_state_dict(path))} tensors")

    print()
    if FAILS:
        print(f"FAIL  {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())