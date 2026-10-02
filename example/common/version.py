"""Load a run configuration by version and tier.

Both versions load through this one function on purpose. If the experimental
version kept its constants hard-coded in `example/common/config.py` while the
practical version read a YAML file, then "the experimental run works" and "the
practical config is correct" would be two code paths that never touched, and the
only way to find out they disagree is to run both and diff the numbers.

Layout (SPEC 1.1.1):

    configs/example/tier0.yaml      256 / 2 layers / ctx 128   -- experimental
    configs/practical/tier1.yaml    512 / 8 layers / ctx 512   -- practical

The loader resolves the version's suffix itself, so nothing downstream has to
spell out `-example`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"

EXPERIMENT_SUFFIX = "-example"
VERSION_DIR = {"example": "example", "practical": "practical"}
SUFFIX = {"example": EXPERIMENT_SUFFIX, "practical": ""}


def config_path(version: str, tier: str) -> Path:
    if version not in VERSION_DIR:
        raise ValueError(f"version must be one of {sorted(VERSION_DIR)}, got {version!r}")
    return CONFIG_ROOT / VERSION_DIR[version] / f"{tier}.yaml"


def load_config(version: str, tier: str) -> dict:
    """Parse a YAML config. Kept dependency-free on purpose (no pyyaml in .venv)."""
    path = config_path(version, tier)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Every run is described by a config file: the "
            f"experimental version hard-codes nothing, and the practical one "
            f"must not borrow the experimental numbers.")
    return parse_yaml(path.read_text())


def parse_yaml(text: str) -> dict:
    """Minimal YAML subset: nested mappings, scalars, `#` comments, no lists.

    The .venv has no pyyaml and adding it for a hundred lines of config is not
    worth it; if this ever grows a list, swap in `yaml.safe_load` behind the same
    signature.
    """
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(value.strip())
    return root


def _scalar(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.lower() in ("null", "none", "~"):
        return None
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def cell_path(name: str, version: str = "example", kind: str = "ckpt") -> Path:
    """checkpoints/<name><suffix> or logs/run/<name><suffix>."""
    if version not in SUFFIX:
        raise ValueError(f"version must be one of {sorted(SUFFIX)}, got {version!r}")
    # `/` binds tighter than the conditional expression, so the string concat
    # has to happen *inside* the else branch. Written the other way round this
    # raises TypeError on the "run" path only -- and every test here goes
    # through the "ckpt" path, so nothing catches it.
    root = ROOT / ("checkpoints" if kind == "ckpt" else "logs/run")
    return root / f"{name}{SUFFIX[version]}"


@dataclass(frozen=True)
class VersionSpec:
    version: str
    suffix: str
    tier: str


def version_spec(version: str, tier: str) -> VersionSpec:
    return VersionSpec(version, SUFFIX[version], tier)