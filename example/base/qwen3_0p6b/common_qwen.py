"""Row 3 of SPEC 2.3: Qwen3-0.6B + LoRA, on both tokenizer tracks' real model.

Three things differ from the 3.2M Tier 0 cells and each needs a decision:

* **The mask token.** SPEC 3.2 originally said id 151669, which does not exist in
  this tokenizer (its highest id is 151668). 151669 is exactly the id a *newly
  added* token receives, so that number assumed one was added. We **reuse
  `<|fim_pad|>` (151662) instead** and never touch the embedding shape:
  - `resize_token_embeddings(151670)` would shrink the checkpoint's 151,936
    rows down to 151,670, dropping 266 rows the tokenizer can never emit. That
    is harmless for correctness but silently moves `params_unique` by 272,384.
  - Reuse keeps the parameter count exact, so the A/B census lines up.
  - Verified: neither `tinystories-6400` nor `minimind3-6400` contains any
    token id >= 151000, so a corpus can never collide with this id.
  - `<|fim_pad|>` is a real Qwen3 special token (the fill-in-middle pad), so
    its embedding is *not* random. Irrelevant here -- LoRA freezes the
    embedding -- but do not read it as a blank row.
  See SPEC.md §3.2 and example/ISSUES.md DONE-1.
* **The tokenizer.** Row 3 uses the Qwen BPE (151,669), not the project's 6400.
  Loss and perplexity are only comparable within a track (SPEC 3.3), so the
  numbers in RESULTS for these two cells are read against each other and not
  against cells 1-8.
* **The base is frozen.** Only LoRA r=8 on the four attention projections trains
  -- 2,293,760 of 595,777,536 parameters, 0.384%. That is the A2D recipe from
  notes/03 §10: attention is what has to change from causal to bidirectional.
"""
from __future__ import annotations

import json
import os

import numpy as np

# Must be set before any from_pretrained call; the hub cache lives on /mnt/extra.
os.environ.setdefault("HF_HOME", "/mnt/extra/hf")

import torch

from common.config import ROOT
from common.sft import SFTData

CELL_DLLM = "qwen3-0.6b-dllm_a2d"
CELL_MINIMIND = "qwen3-0.6b-mini_a2d"
BASE = "Qwen3-0.6B"
MODEL_ID = "Qwen/Qwen3-0.6B"
MASK_TOKEN = "<|fim_pad|>"          # 151662, already in the vocabulary
MASK_ID = 151662
LORA_R = 8
LORA_ALPHA = 16
BLOCK_SIZE = 128
CONTEXT = 128


def load_tokenizer():
    """Load the Qwen3 tokenizer unchanged.

    No `add_special_tokens`, no resize. The mask is a token the tokenizer
    already knows, so nothing here mutates it -- that is the whole point of
    reusing 151662 over adding one.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if MASK_TOKEN not in tok.get_vocab():
        raise RuntimeError(f"{MASK_TOKEN} is not in the Qwen3 vocabulary; "
                           f"SPEC 3.2's mask id needs revisiting")
    if tok.convert_tokens_to_ids(MASK_TOKEN) != MASK_ID:
        raise RuntimeError(f"{MASK_TOKEN} moved to "
                           f"{tok.convert_tokens_to_ids(MASK_TOKEN)}, not {MASK_ID}")
    return tok


def build_a2d_lora(seed: int = 42, trainable: str = "all"):
    """Qwen3-0.6B -> A2DQwen3 (bidirectional) -> LoRA.

    `trainable` mirrors the A2D ablations from notes/03 §10: "all" (the default
    here), "freeze_ffn" and "qk". Kept as a flag because the ablation is the
    cheapest interesting experiment in this row.
    """
    import dllm
    from dllm.pipelines.a2d import A2DQwen3Config
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from common.seeds import seed_all

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    target = A2DQwen3Config(**{k: v for k, v in cfg.to_dict().items()
                               if k != "model_type"})
    seed_all(seed)

    # Load the *pretrained* Qwen3-0.6B weights, then convert. Building the A2D
    # model straight from the config gives a randomly initialised 600M backbone,
    # and LoRA on top of that is a 600M-parameter scratch run wearing a LoRA
    # costume -- it looks identical in the parameter census and trains fine.
    source = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    state = source.state_dict()
    pretrained_embedding = state["model.embed_tokens.weight"].clone()
    model = dllm.pipelines.a2d.A2DQwen3LMHeadModel(target)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in ("lm_head.weight",)]   # tied alias
    if missing or unexpected:
        raise RuntimeError(f"Qwen3 weights did not transfer: "
                           f"missing={missing[:5]} unexpected={unexpected[:5]}")
    del source, state

    tokenizer = load_tokenizer()
    # No resize: the embedding stays exactly as the checkpoint ships it. Assert
    # that rather than trusting it -- the old code resized here, and a resize
    # that silently succeeded would move params_unique by 272,384 with nothing
    # in the output to say so.
    rows = model.get_input_embeddings().weight.shape[0]
    if rows != pretrained_embedding.shape[0]:
        raise RuntimeError(f"embedding was resized: {rows} rows, checkpoint "
                           f"has {pretrained_embedding.shape[0]}")
    if not torch.equal(model.get_input_embeddings().weight.detach(),
                       pretrained_embedding.detach()):
        raise RuntimeError("embedding rows changed; the mask reuse path must "
                           "leave the embedding bit-identical")

    modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if trainable == "qk":
        modules = ["q_proj", "k_proj"]
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                      target_modules=modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    if trainable == "freeze_ffn":
        # notes/03 §10: FFN stores knowledge, attention stores the pattern.
        for name, param in model.named_parameters():
            if ".mlp." in name and param.requires_grad:
                param.requires_grad_(False)
    return model, tokenizer


def build_blocks(tokenizer, n_docs: int = 2000, block_size: int = BLOCK_SIZE,
                 seed: int = 42) -> torch.Tensor:
    """Tokenise a slice of the raw corpus and pack it into fixed-length blocks.

    Row 3 cannot reuse `common.data` -- that reads a corpus pre-tokenised with
    the project's 6400-vocab BPE, and this row is on the Qwen tokenizer track
    (SPEC 3.3). Encoding is ~1.5M tok/s, so re-tokenising a slice costs nothing.
    """
    raw = ROOT / "data" / "raw" / "tinystories.jsonl"
    ids: list[int] = []
    with raw.open() as fh:
        for i, line in enumerate(fh):
            if i >= n_docs:
                break
            text = json.loads(line)["text"]
            # Qwen3 has no BOS (bos_token_id is None): documents are separated
            # by EOS alone. Prepending None would poison the tensor build.
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            ids.append(tokenizer.eos_token_id)

    usable = len(ids) // block_size * block_size
    tensor = torch.tensor(ids[:usable], dtype=torch.int32).reshape(-1, block_size)
    g = torch.Generator().manual_seed(seed)
    return tensor[torch.randperm(tensor.shape[0], generator=g)]


def build_sft_data(tokenizer, n_docs: int = 4000, block_size: int = BLOCK_SIZE,
                    raw: Path | None = None):
    """Flatten `sft_t2t_mini.jsonl` on the **Qwen** tokenizer, with loss spans.

    Row 3 cannot reuse the 6400-vocab prepared corpus at all: SPEC 3.3 keeps the
    two tracks' token ids apart, and the SFT spans have to come from the same
    tokenizer the model speaks. `common.sft` supplies the span machinery and takes
    the encoder as a parameter, so the template is identical on both tracks and
    only the ids differ.

    Qwen3 has no BOS (`bos_token_id is None`) and its EOS is
    `<|endoftext|>` = 151643, while on the 6400 track id 1 is `<|im_start|>`
    and id 2 is an ordinary BPE token. Using the 6400 ids here would put
    meaningful text in meaningless positions, so all three ids are taken from
    the tokenizer.
    """
    from common.sft import IGNORE, build_sft_blocks

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    src = raw or (ROOT / "data" / "raw" / "minimind_dataset" / "sft_t2t_mini.jsonl")
    # bos=None: Qwen3 has no BOS, and id 1 on this tokenizer is `<|im_start|>`,
    # a real token. Prepending the 6400 track's BOS would put a meaningful token
    # where the block is supposed to just start.
    arr, spans, stats = build_sft_blocks(src, block_size, max_docs=n_docs,
                                         enc=enc, bos=None,
                                         eos=tokenizer.eos_token_id,
                                         pad_id=tokenizer.pad_token_id)
    labels = np.full(arr.shape, IGNORE, dtype=np.int64)
    for i, row in enumerate(spans):
        for a, b in row:
            labels[i, a:b] = arr[i, a:b]
    blocks = torch.from_numpy(arr.astype(np.int32))
    return SFTData(blocks, spans, block_size, src)


def special_ids(tokenizer) -> dict:
    return {"pad": tokenizer.pad_token_id, "eos": tokenizer.eos_token_id,
            "bos": tokenizer.bos_token_id,   # None on Qwen3
            "mask": tokenizer.convert_tokens_to_ids(MASK_TOKEN),
            "vocab": len(tokenizer)}


def trainable_report(model) -> dict:
    from common.ckpt import describe

    info = describe(model)
    return {"trainable": info["params_trainable"], "total": info["params_unique"],
            "ratio_pct": round(100 * info["params_trainable"] / info["params_unique"], 4)}


def data_dir_note() -> str:
    return f"tokenised on the fly from {(ROOT / 'data' / 'raw' / 'tinystories.jsonl').name}"