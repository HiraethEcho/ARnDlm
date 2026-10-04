"""The experimental version's configuration, read from configs/example/tier0.yaml.

SPEC 1.1.1: nothing here is a literal. The dataclasses keep the *shape* (which
fields a model config has), and every value comes from the YAML through the
shared loader, so the practical version's configs/practical/tier1.yaml takes
exactly the same path. A constant written here instead would be a value the
practical version never validates.

Imports below stay module-level and eager so a malformed config fails at import
time rather than on the first batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .version import cell_path, load_config

ROOT = Path(__file__).resolve().parents[2]          # example/common/ -> repo root

VERSION = "example"
TIER = "tier0"
_RAW = load_config(VERSION, TIER)

TOKENIZER = ROOT / "tokenizer" / "tokenizer.json"
DATA_DIR = ROOT / _RAW["data"]["prepared_dir"]
# No CKPT_ROOT / RUN_ROOT here. Output paths go through version.cell_path, which
# owns the -example suffix; a second spelling of the same root is a second thing
# to keep in sync, and there are zero call sites for these (ISSUES DONE-15).

_tok = _RAW["tokens"]
# MASK reuses a reserved buffer token instead of growing the vocab (SPEC 3.2):
# a new row would change the embedding shape and silently break every weight
# copied from the AR checkpoint during A2D.
PAD, BOS, EOS, MASK = _tok["pad"], _tok["bos"], _tok["eos"], _tok["mask"]

# Fixed generation prompts (SPEC "Evaluation plan").
PROMPTS = ("Once upon a time", "The future of AI is")


def _from(raw: dict, cls):
    """Build a dataclass from a YAML mapping. Unknown keys are an error.

    A typo in the config that is silently dropped would leave the default in
    place, and a default that differs between the two versions is exactly what
    this indirection is meant to prevent.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class ModelCfg:
    """Qwen-style decoder shape (SPEC 3.1)."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    context_length: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class TrainCfg:
    steps: int
    batch_size: int
    lr: float
    dlm_lr: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    beta1: float
    beta2: float
    seed: int
    log_every: int
    loss_window: int

    @property
    def betas(self) -> tuple[float, float]:
        return (self.beta1, self.beta2)

    @property
    def dlm(self) -> "TrainCfg":
        """Same schedule, the fine-tuning learning rate (SPEC 3.3 A2D)."""
        return TrainCfg(**{**asdict(self), "lr": self.dlm_lr})


TIER0 = _from(_RAW["model"], ModelCfg)
AR_TRAIN = _from(_RAW["train"], TrainCfg)
DLM_TRAIN = AR_TRAIN.dlm

# Derived from TIER0 rather than read out of _RAW a second time: two sources for
# one number can only drift, and the drift shows up as an embedding-shape error
# far from its cause. No module from-imports VOCAB -- every call site is C.VOCAB,
# which is why defining it below TIER0 is safe.
VOCAB = TIER0.vocab_size

# No QWEN3_LORA here either: it was defined with steps=20 / batch_size=2 and had
# zero call sites. The LoRA cells run on DLM_TRAIN (RESULTS: 200 steps), so the
# 20-step geometry here was an intention that never got wired up (ISSUES DONE-15).

BLOCK_SIZE = TIER0.context_length                  # packing block == context

# Row -> the base label that goes into every RESULTS row and meta.json. One
# spelling, kept next to the config it belongs to: three scripts wrote the same
# string twice, and one of them drifted to a different literal once.
BASE_OF = {"hf": "Qwen-mini-HF", "native": "Qwen-mini-MiniMind"}
MAX_BLOCKS = _RAW["data"]["max_blocks"]
DTYPE = _RAW["runtime"]["dtype"]

EVAL_BATCHES = 4                                   # frozen batches for eval loss


def as_dict(obj) -> dict:
    """dataclass -> plain dict, for json snapshots in checkpoints."""
    return asdict(obj)


def cell_dir(name: str) -> Path:
    """checkpoints/<name>-example (SPEC 1.1.1)."""
    return cell_path(name, VERSION, "ckpt")


def run_dir(name: str) -> Path:
    return cell_path(name, VERSION, "run")
