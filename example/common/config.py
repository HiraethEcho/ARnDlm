"""Frozen Tier-0 smoke constants. One source of truth for all 10 cells.

Every number here is SPEC 4 Tier 0: the goal is pipeline validity, not model
quality. They live in one dataclass instead of being re-typed per script
because a silent mismatch between two cells would invalidate every A/B number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # example/common/ -> repo root

TOKENIZER = ROOT / "tokenizer" / "tokenizer.json"
DATA_DIR = ROOT / "data" / "prepared" / "tinystories-6400"
CKPT_ROOT = ROOT / "checkpoints" / "example"
RUN_ROOT = ROOT / "logs" / "run" / "example"

# Special ids. MASK reuses a reserved buffer token instead of growing the vocab
# (SPEC 3.2): a new row would change the embedding shape and silently break
# every weight copied from the AR checkpoint during A2D.
PAD, BOS, EOS, MASK = 0, 1, 2, 27
VOCAB = 6400

# Fixed generation prompts (SPEC "Evaluation plan").
PROMPTS = ("Once upon a time", "The future of AI is")


@dataclass(frozen=True)
class ModelCfg:
    """Qwen-style decoder shape (SPEC 3.1)."""

    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 2
    num_kv_heads: int = 1
    intermediate_size: int = 768
    context_length: int = 128
    vocab_size: int = VOCAB
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


TIER0 = ModelCfg()


@dataclass(frozen=True)
class TrainCfg:
    steps: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    min_lr_ratio: float = 0.1      # cosine floor, as a fraction of lr
    warmup_steps: int = 20
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 20
    loss_window: int = 20          # avg of first/last N steps -> RESULTS.md


AR_TRAIN = TrainCfg(lr=3e-4)
DLM_TRAIN = TrainCfg(lr=1e-4)                      # A2D fine-tuning is gentler
# T2c only: real Qwen3-0.6B is 200x Tier 0, so the smoke run stays tiny.
QWEN3_LORA = TrainCfg(steps=20, batch_size=2, lr=1e-4, warmup_steps=2, log_every=5)

BLOCK_SIZE = TIER0.context_length                  # packing block == context


def as_dict(obj) -> dict:
    """dataclass -> plain dict, for json snapshots in checkpoints."""
    return asdict(obj)


def cell_dir(name: str) -> Path:
    """checkpoints/example/<name> -- the `-example` suffix keeps A/B apart."""
    return CKPT_ROOT / name


def run_dir(name: str) -> Path:
    return RUN_ROOT / name