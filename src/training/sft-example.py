"""src/training/sft-example.py -- src/training/sft.py 的参考实现。

chat 模板 + 只在 assistant 区间算 loss（SPEC 6.2 T4a/T4b）。
ar-trans 与 ar-native 两个 base 共用这一层（SPEC §8: 同 sft 层）。

数据 = prepared SFT 目录（实测格式, sft-mini-6400）:
  tokens.bin: 预切好的定长块 uint16, [n_blocks, block_size] —— 不是裸文档流
  assistant_spans.npy: int64 [n_spans, 3] = (block_idx, start, end-excl)
  硬对账: sum(end-start) == meta.supervised_tokens
模板: user\\n{q}assistant\\n{a}，轮次分隔是 EOS（id 2, decode 默认不可见）。

批次顺序仍是 f(seed, step) 纯函数（C4 的根同款规则）。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOS, EOS = 1, 2
IGNORE = -100  # HF 与我们的 CE 同一约定


class SFTBlocks:
    """prepared SFT 目录的只读视图: 块 + 每 block 的 assistant 区间。"""

    def __init__(self, prepared_dir: str | Path):
        base = Path(prepared_dir)
        meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.bs = int(meta["block_size"])
        self.vocab = int(meta["vocab_size"])
        toks = np.memmap(base / "tokens.bin", dtype=np.uint16, mode="r")
        if len(toks) != int(meta["n_blocks"]) * self.bs:
            raise ValueError(f"tokens.bin {len(toks)} != n_blocks*bs "
                             f"{meta['n_blocks'] * self.bs}")
        if 27 in np.unique(toks):
            raise ValueError("SFT 语料正文出现 MASK id 27（T4f 会必炸, 先修数据）")
        self.blocks = toks  # [n_blocks*bs] 虚拟 [n_blocks, bs]
        self.n_blocks = int(meta["n_blocks"])
        spans = np.load(base / "assistant_spans.npy")
        if int((spans[:, 2] - spans[:, 1]).sum()) != int(meta["supervised_tokens"]):
            raise ValueError("spans 覆盖量 != meta.supervised_tokens")
        # 按 block 分组（spans 已按 block 排序则直接切; 排序一次代价小）
        order = np.argsort(spans[:, 0], kind="stable")
        spans = spans[order]
        self._span_starts = np.searchsorted(spans[:, 0], np.arange(self.n_blocks),
                                            side="left")
        self._span_ends = np.searchsorted(spans[:, 0], np.arange(self.n_blocks),
                                          side="right")
        self._spans = spans
        if int(meta["supervised_tokens"]) == 0:
            raise ValueError("没有任何监督 token —— 会静默训出一个什么都不学的模型")

    def labels(self, block_idx: int) -> np.ndarray:
        """[bs] int64: 区间内 = 原 token, 区间外 = -100（T4b 的数据侧本体）。"""
        ids = np.asarray(self.blocks[block_idx * self.bs:(block_idx + 1) * self.bs],
                         dtype=np.int64).copy()
        labels = np.full(self.bs, IGNORE, dtype=np.int64)
        for s, e in self._spans[self._span_starts[block_idx]:self._span_ends[block_idx], 1:3]:
            labels[s:e] = ids[s:e]
        return labels


class SFTBatcher:
    """step -> (blocks, labels) 的纯函数。epoch 重排与 common/data 同一规则。"""

    def __init__(self, sft: SFTBlocks, batch_size: int, seed: int):
        if sft.n_blocks < batch_size:
            raise ValueError(f"n_blocks({sft.n_blocks}) < batch_size({batch_size})")
        self.sft = sft
        self.B = batch_size
        self.seed = seed

    def _perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.sft.n_blocks)

    def position(self, step: int) -> tuple[int, int]:
        pos = step * self.B
        return pos // self.sft.n_blocks, pos % self.sft.n_blocks

    def _indices(self, step: int) -> np.ndarray:
        epoch, within = self.position(step)
        cur = self._perm(epoch)[within:]
        if len(cur) >= self.B:
            return cur[:self.B]
        return np.concatenate([cur, self._perm(epoch + 1)[:self.B - len(cur)]])

    def batch(self, step: int) -> np.ndarray:
        idx = self._indices(step)
        out = np.stack([np.asarray(self.sft.blocks[i * self.sft.bs:(i + 1) * self.sft.bs])
                        for i in idx])
        return out.astype(np.int64)

    def labels(self, step: int, batch: np.ndarray) -> np.ndarray:
        return np.stack([self.sft.labels(i) for i in self._indices(step)])


def encode_prompt(question: str) -> list[int]:
    """采样前缀, 与语料模板逐 token 同构: [BOS] user\\n{q} [EOS] assistant\\n。

    f"user\\n{q}\\nassistant" 会把 EOS 位置写成换行 —— token 数相同、内容不同、
    无报错, 采样与 rep-4 就会描述一个模型没训过的前缀（legacy 踩过的坑）。
    """
    tk = _load("common/tokenizer-example.py", "tokenizer_example_sft").Tok()
    return ([BOS] + tk.encode(f"user\n{question}")
            + [EOS] + tk.encode("assistant\n"))


def run_sft(cfg, steps: int, ctx, impl: str, base_ckpt: str | Path | None = None,
            resume_path: str | Path | None = None) -> dict:
    """在 SFT 数据上跑 --impl。base_ckpt: 从 AR pretrain 权重继续（A2D/SFT 的起点）。"""
    sft_dir = Path(ctx.data_dir)  # main 把 data_dir 指到 prepared SFT 目录
    sft = SFTBlocks(sft_dir)
    sb = SFTBatcher(sft, batch_size=cfg.TRAIN.batch_size, seed=cfg.TRAIN.seed)

    if impl == "native":
        native = _load("src/training/ar_native-example.py", "ar_native_example_sft")
        res = native.run(cfg, steps, ctx, resume_path=resume_path,
                         batcher=sb, labels_fn=sb.labels,
                         init_weights_path=base_ckpt)
    else:
        trans = _load("src/training/ar_trans-example.py", "ar_trans_example_sft")

        class _SFTBlocksDataset(IterableDataset):
            # 有序 IterableDataset: 逐 item yield {input_ids, labels}
            def __init__(self, batcher, n_items: int):
                self.sb, self.n = batcher, n_items  # 不定义 __len__: 防 Trainer 走 RandomSampler

            def __iter__(self):
                for b in range(self.n // self.sb.B):
                    blocks = torch.from_numpy(self.sb.batch(b)).long()
                    labels = torch.from_numpy(self.sb.labels(b, None)).long()
                    for w in range(self.sb.B):
                        yield {"input_ids": blocks[w], "labels": labels[w].clone()}

        ds = _SFTBlocksDataset(sb, steps * cfg.TRAIN.batch_size)
        res = trans.run(cfg, steps, ctx, dataset=ds, positioner=sb,
                        init_weights_path=base_ckpt)
    return {**res, "sft": True}
