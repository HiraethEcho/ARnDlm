"""utils/config.py -- 把 YAML 的 dict 变成 frozen dataclass 和一组共享常量。

用途
    实践版里唯一读配置的文件。其它模块 —— utils/data.py、utils/loop.py、
    models/hf_qwen/、training/ar_hf/ —— 的数字全部从这里 import。在训练脚本里
    手打一个字面量，就意味着有一个数字既没被本文件校验过，也没被实验版承认过。

    对应 example/teach/00-common-base.md 的 Step 0.2。配置通过
    utils/version.py::load_config 从 configs/practical/*.yaml 读进来。

要写什么
    两个 frozen dataclass。字段名必须和 YAML 的键逐字相同，因为 splat 进去的就是
    YAML 那几段：

        ModelCfg (frozen)      <- `model:` 段，10 个字段
            hidden_size  num_layers  num_heads  num_kv_heads
            intermediate_size  context_length  vocab_size
            rope_theta  rms_norm_eps  tie_word_embeddings
            + @property head_dim = hidden_size // num_heads

        TrainCfg (frozen)      <- `train:` 段，13 个字段
            steps  batch_size  lr  dlm_lr  min_lr_ratio  warmup_steps
            weight_decay  grad_clip  beta1  beta2  seed  log_every  loss_window
            + @property betas = (beta1, beta2)
            + @property dlm  -> 同一套调度、lr 换成 dlm_lr（C 段用它）

    _from(raw: dict, cls) -> 上面两个之一
        用 mapping 构造 dataclass；未知键就报错。

    模块级常量，全部派生，一个字面量都不写：
        PAD, BOS, EOS, MASK   <- `tokens:`
        VOCAB                 <- model.vocab_size
        BLOCK_SIZE = TIER0.context_length
        DATA_DIR   = ROOT / data.prepared_dir      # yaml 里写的是相对路径
        MAX_BLOCKS <- data.max_blocks
        DTYPE      <- runtime.dtype                # 由 utils/device.py 消费
        EVAL_BATCHES = 4                           # 评测几何，留在代码里
        PROMPTS = ("Once upon a time", "The future of AI is")
        TIER0, AR_TRAIN, DLM_TRAIN
        as_dict(obj) -> dict                       # 给 checkpoint 的 json 快照
        cell_dir(name) / run_dir(name)             # cell_path 的薄包装

注意
    1. `head_dim` 是 property，不是字段。到了 models/hf_qwen/ 那边要把它显式写进
       Qwen3Config，因为库自己的推导规则随版本变 —— 那是 A1 的事，不在这里。
       在这里它保持派生，这样它不可能和 hidden_size / num_heads 互相矛盾。
    2. `_from` 必须在构造**之前**检查未知键。先 `cls(**raw)` 的话 Python 一次只报
       一个坏字段，你修掉一个才发现下一个。为什么未知键要命：`temperary: 1e-4` 这种
       拼写错误否则会静静落在默认值上，而两版默认值不一致，正是这层间接要拦住的幽灵
       A/B 差值。
    3. MASK 复用保留的 buffer token 27，不扩词表（SPEC 3.2）。多一行 embedding 就改
       了张量形状，A2D 从 AR checkpoint 拷权重时会静默对不上。
    4. BLOCK_SIZE 是**绑定到** context_length，不是恰好等于它。两个独立数字迟早一个
       改了另一个没改，DLM 的分块去噪就会跑在模型没见过的上下文上。
    5. `MAX_BLOCKS = 0` 表示不限量，靠的是真值判断（`if max_blocks:`）。写成
       `if max_blocks is not None:`，0 就把语料截成 0 块 —— 你得到空数据集，而不是
       一个报错。
    6. DATA_DIR 必须和 ROOT 拼。YAML 里写相对路径是为了配置可移植；直接拿来用，脚本
       就只有在仓库根目录下启动才能跑。
    7. checkpoints/ 和 logs/run/ 与 main worktree 共用（checkpoints ->
       /mnt/extra/ckpt），而本版本的后缀是空串。实验版格子永远带 `-example`，所以
       实践版的名字覆盖不到它 —— 会撞的是**本 worktree 和 main 各自跑出的实践版名字**。
       给自己的名字加前缀（`ar-512-smoke`），别让一次 smoke 落在一次 tier1 上。
       撞上了不会报错，只会静默覆盖权重。
    8. 在这里声明了 DTYPE，就必须有代码读它。没人检查的常量是一份静默分歧的邀请函
       （GOAL.md 清单第 7 条）。读它的是 utils/device.py；在它存在之前，用注释说明
       这一点，别让它看起来已经生效。

依赖
    utils/version.py（load_config、cell_path、ROOT；as_dict 来自
    dataclasses.asdict）。与 example/common/config.py 字段级对齐 —— 同名、同集合、
    不多一个。

验收
    .venv/bin/python -c "import utils.config as C; \\
        print(C.TIER0.head_dim, C.BLOCK_SIZE, C.MASK, C.AR_TRAIN.betas, C.DLM_TRAIN.lr)"
        -> 128 128 27 (0.9, 0.95) 0.0001

    未知键检查：拿 configs/practical/smoketest.yaml 的副本，在 `train:` 下加一个
    `spint: 1`，确认报错**一次列出全部**坏字段，不是一个一个挤牙膏。

    然后把那行临时加的删掉。configs/ 里不该留一个没人读的旋钮。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .version import cell_path, load_config

ROOT = Path(__file__).resolve().parents[1]
VERSION = "practical"
TIER = "smoketest"
_RAW = load_config(VERSION, TIER)
TOKENIZER = ROOT / "tokenizer" / "tokenizer.json"
DATA_DIR = ROOT / _RAW["data"]["prepared_dir"]
CKPT_ROOT = ROOT / "checkpoints"
RUN_ROOT = ROOT / "logs" / "run"

_tok = _RAW["tokens"]
PAD, BOS, EOS, MASK = _tok["pad"], _tok["bos"], _tok["eos"], _tok["mask"]
PROMPTS = ("Once upon a time", "The future of AI is")
# PROMPTS = ("The capital of France is", "The future of AI is")

def _from(raw: dict, cls):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"wdnmd wrong config keys for {cls.__name__}: {sorted(unknown)} you idot!")
    return cls(**raw)

@dataclass(frozen=True)
class ModelCfg:
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
    def dlm(self) -> TrainCfg:
        """Same schedule, the fine-tuning learning rate (SPEC 3.3 A2D)."""
        return TrainCfg(**{**asdict(self), "lr": self.dlm_lr})

MODEL = _from(_RAW["model"], ModelCfg)
AR_TRAIN = _from(_RAW["train"], TrainCfg)
DLM_TRAIN = AR_TRAIN.dlm
BLOCK_SIZE = MODEL.context_length
VOCAB = MODEL.vocab_size
MAX_BLOCKS = _RAW["data"]["max_blocks"]
DTYPE = _RAW["runtime"]["dtype"]
EVAL_BATCHES = 4

def as_dict(obj) -> dict:
    # json snapshots in ckpt
    return asdict(obj)
def cell_dir(name: str) -> Path:
    """checkpoints/<name> (SPEC 1.1.1)."""
    return cell_path(name, VERSION, "ckpt")
def run_dir(name: str) -> Path:
    return cell_path(name, VERSION, "run")
