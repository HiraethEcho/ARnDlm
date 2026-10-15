"""common/config-example.py -- models/config.py 的参考实现。

pyyaml 读 configs/<tier>.yaml -> frozen dataclass(MODEL/TRAIN/DATA) + 共享常量。
未知键 / 缺键 / 浮点写成 3e-4（无小数点, YAML 1.1 会给 str）都在加载时报错,
一次列全, 不留静默。加载即验证: 类型错了不让进训练。

用法:
    from common.config_example import load  # 实际经 importlib 从本文件加载
    cfg = load("smoketest")
    cfg.MODEL.context_length, cfg.BLOCK_SIZE, cfg.DTYPE ...
"""
import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

# ---- frozen dataclass：字段集就是「已知键」的全部，yaml 多一个少一个都不行 ----


@dataclass(frozen=True)
class MODEL:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    context_length: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool


@dataclass(frozen=True)
class TRAIN:
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


@dataclass(frozen=True)
class DATA:
    prepared_dir: str
    max_blocks: int  # 0 = 不限量


@dataclass(frozen=True)
class RUNTIME:
    dtype: str


@dataclass(frozen=True)
class Config:
    tier: str
    MODEL: MODEL
    TRAIN: TRAIN
    DATA: DATA
    RUNTIME: RUNTIME
    # 共享常量（SPEC 3.2 tokens 段；VOCAB/BLOCK_SIZE 从 MODEL 推导, 不给第二处来源）
    PAD: int
    BOS: int
    EOS: int
    MASK: int
    VOCAB: int
    BLOCK_SIZE: int
    DTYPE: str

    @property
    def DATA_DIR(self) -> Path:
        return ROOT / self.DATA.prepared_dir


_SECTIONS: dict[str, type] = {"model": MODEL, "train": TRAIN, "data": DATA, "runtime": RUNTIME}


def _check_types(section: str, cls: type, d: dict) -> list[str]:
    """yaml 浮点坑：3e-4 会被 YAML 1.1 解析成 str。这里把坑变成显式报错。"""
    bad = []
    types = {f.name: f.type for f in fields(cls)}
    for k, v in d.items():
        t = types[k]
        if t is float and not isinstance(v, float):
            bad.append(f"{section}.{k}={v!r} (float 字段收到 {type(v).__name__}; "
                       f"yaml 数值必须写 3.0e-4 形式, 带小数点)")
        if t is int and not isinstance(v, int):
            bad.append(f"{section}.{k}={v!r} (int 字段收到 {type(v).__name__})")
        if t is bool and not isinstance(v, bool):
            bad.append(f"{section}.{k}={v!r} (bool 字段收到 {type(v).__name__})")
    return bad


def load(tier: str) -> Config:
    """读 configs/<tier>.yaml，产出 frozen Config。错键/缺键/错型一次报完。"""
    path = CONFIG_DIR / f"{tier}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config 不存在: {path} (可选: "
                                f"{sorted(p.stem for p in CONFIG_DIR.glob('*.yaml'))})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    problems: list[str] = []
    parsed: dict[str, dict] = {}
    known_sections = set(_SECTIONS) | {"tier", "purpose", "tokens"}
    # 未知顶层段
    unknown_top = sorted(set(raw) - known_sections)
    if unknown_top:
        problems.append(f"未知顶层键: {unknown_top}")

    for name, cls in _SECTIONS.items():
        got = raw.get(name)
        if not isinstance(got, dict):
            problems.append(f"缺段: {name}")
            continue
        want = {f.name for f in fields(cls)}
        miss = sorted(want - set(got))
        unk = sorted(set(got) - want)
        if miss:
            problems.append(f"{name} 缺键: {miss}")
        if unk:
            problems.append(f"{name} 未知键: {unk}")
        problems += _check_types(name, cls, {k: got[k] for k in want & set(got)})
        parsed[name] = got

    # tokens 段: 4 个 id, 一个都不能少
    tok = raw.get("tokens")
    if not isinstance(tok, dict) or sorted(tok) != ["bos", "eos", "mask", "pad"]:
        problems.append(f"tokens 段必须恰好是 bos/eos/mask/pad 四键, 实际: {sorted(tok) if isinstance(tok, dict) else tok}")
    if not problems:
        tok = {k: int(v) for k, v in tok.items()}

    if problems:
        raise ValueError(f"configs/{tier}.yaml 校验失败 ({len(problems)} 条):\n  " + "\n  ".join(problems))

    m = MODEL(**parsed["model"])
    # 派生常量只有一个来源
    if m.num_heads % m.num_kv_heads != 0:
        problems.append(f"num_heads({m.num_heads}) 必须被 num_kv_heads({m.num_kv_heads}) 整除 (GQA)")
    if problems:
        raise ValueError(f"configs/{tier}.yaml 校验失败:\n  " + "\n  ".join(problems))

    return Config(
        tier=str(raw["tier"]),
        MODEL=m,
        TRAIN=TRAIN(**parsed["train"]),
        DATA=DATA(**parsed["data"]),
        RUNTIME=RUNTIME(**parsed["runtime"]),
        PAD=tok["pad"], BOS=tok["bos"], EOS=tok["eos"], MASK=tok["mask"],
        VOCAB=m.vocab_size,
        BLOCK_SIZE=m.context_length,  # 打包块 = 上下文长（不设第二个来源）
        DTYPE=parsed["runtime"]["dtype"],
    )


def lr_at(cfg: Config, step: int) -> float:
    """warmup 线性 -> cosine 衰减到 min_lr_ratio。两条 impl 路线共用同一公式。

    warmup 语义与 HF cosine_with_warmup 一致（2026-10-07 review 后定）：
    step0 = 0，峰值在 step == warmup 整点。之前是 (step+1)/warmup（峰在
    warmup-1），与 HF 相差一步 —— 两条路线 warmup 段每步 lr 都不同。
    """
    t = cfg.TRAIN
    warmup = t.warmup_steps
    if step < warmup:
        return t.lr * step / warmup
    span = max(t.steps - warmup, 1)
    progress = min((step - warmup) / span, 1.0)
    floor = t.lr * t.min_lr_ratio
    return floor + (t.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


if __name__ == "__main__":
    for tier in ("smoketest", "tier1"):
        c = load(tier)
        n = c.TRAIN
        print(f"{tier}: {c.MODEL.num_layers}L h{c.MODEL.hidden_size} ctx{c.BLOCK_SIZE} "
              f"steps={n.steps} lr={n.lr} dtype={c.DTYPE} data={c.DATA.prepared_dir}")
