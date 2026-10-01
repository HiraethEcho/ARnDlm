#!/usr/bin/env python
"""Download the MiniMind dataset files listed in ~/Public/minimind/README.md.

Example:
    python scripts/download_minimind.py                      # every .jsonl in the repo
    python scripts/download_minimind.py --only dpo rlaif   # a subset

Source: HF dataset repo `jingyaogong/minimind_dataset`, reached through
HF_ENDPOINT (https://hf-mirror.com locally, see ~/.config/zsh/dev.zsh).
Files land in data/raw/minimind_dataset/ -- data/ is the symlink to
/mnt/extra/data, so nothing enters the Git worktree.

Two files may already exist one level up (data/raw/pretrain_t2t_mini.jsonl,
data/raw/sft_t2t_mini.jsonl, tokenized into data/prepared/minimind3-6400/).
Those are symlinked instead of downloaded again: the contents are identical and
tokenizing the original path is what meta.json records.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "jingyaogong/minimind_dataset"
# Files already present in data/raw/, reused via symlink.
PREEXISTING = ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl")
DEFAULT_OUTPUT = Path("data/raw/minimind_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", help="File names to download; default is every .jsonl in the repo.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--list", action="store_true", help="List repo files with sizes and exit.")
    p.add_argument("--force-symlink", action="store_true", help="Redo the symlinks even if the target exists.")
    return p.parse_args()


def link_existing(name: str, out_dir: Path, force: bool) -> bool:
    """Symlink data/raw/<name> into the output dir when we already have the file."""
    source = Path("data/raw") / name
    target = out_dir / name
    if not source.exists():
        return False
    if target.is_symlink() or target.exists():
        if not force:
            print(f"exists  {target}")
            return True
        target.unlink()
    target.symlink_to(Path("..") / source.name)
    print(f"symlink {target} -> {source}")
    return True


def main() -> int:
    args = parse_args()
    if "HF_ENDPOINT" not in os.environ:
        print("warning: HF_ENDPOINT unset; huggingface.co may time out on this network")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".jsonl")}
    if not files:
        raise SystemExit(f"no .jsonl files found in {REPO_ID}")

    if args.list:
        for name, size in sorted(files.items(), key=lambda kv: kv[1]):
            print(f"{size / 2**30:8.2f} GiB  {name}")
        return 0

    names = args.only or sorted(files)
    missing = [n for n in names if n not in files]
    if missing:
        raise SystemExit(f"not in {REPO_ID}: {', '.join(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in PREEXISTING and link_existing(name, args.output, args.force_symlink):
            continue
        # Resume from a partial .incomplete file if a previous run was killed.
        path = hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=str(args.output))
        print(f"ok      {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())