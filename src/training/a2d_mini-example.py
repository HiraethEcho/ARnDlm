"""src/training/a2d_mini-example.py -- src/training/a2d_mini.py 的参考实现。

A2D，minimind-dlm 风格手写 masked CE（G5 的手写侧）。两个 base 共用这一层：
  trans  -> 双向 = 传 4D 全零 attention mask（4.57 的 create_causal_mask 原样放行）
  native -> 双向 = forward(is_causal=False)（model-example 的开关）

A2D 转换零权重改动：MASK 用预留 id 27，不扩词表（扩了 embedding 形状,
AR 权重全部作废）。objective 在 common/a2d-example.py（四组合共享）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_a2d = _load("common/a2d-example.py", "a2d_example_mini")
IGNORE = -100  # 与 sft-example 同约定


def dlm_forward_trans(model):
    """transformers Qwen3 的双向 logits_fn：4D 全零 mask（SPEC §5 的 4.57 做法）。"""
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        mask4d = torch.zeros(input_ids.shape[0], 1, t, t,
                             device=input_ids.device, dtype=model.dtype)
        return model(input_ids, attention_mask=mask4d).logits
    return logits_fn


def dlm_forward_native(model):
    """native 模型的双向 logits_fn：显式 is_causal=False（不动全局开关状态）。"""
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids, is_causal=False).logits
    return logits_fn


def make_loss_fn(model, kind: str, generator: torch.Generator | None = None):
    """loss_fn(batch) -> (loss, {"t_mean"})，两个 base 一个闭包。

    batch 两种形状都收（同一闭包，复制会漂移）：
      [B, L]     pretrain/A2D 格子：全位置 supervised
      [B, 2, L]  SFT 格子：row0 ids, row1 labels（区间外 IGNORE -> valid 掩码）
    """
    if kind not in ("trans", "native"):
        raise ValueError(f"未知 kind: {kind!r}")
    logits_fn = (dlm_forward_trans if kind == "trans" else dlm_forward_native)(model)

    def loss_fn(_model, batch: torch.Tensor):
        if batch.dim() == 3:
            ids, labels = batch[:, 0], batch[:, 1]
            valid = labels != IGNORE
        else:
            ids, labels, valid = batch, batch, None
        t = _a2d.sample_t(ids.shape[0], ids.device, generator)
        x_t, masked = _a2d.corrupt(ids, t, _a2d.MASK, generator, valid=valid)
        # t=0 附近会除到 ~1e-6；该 sample 本来就几乎没有 masked 位置。
        # clamp 进新名字（别覆盖 t）：日志里的 t_mean 必须是抽取值，不是 clamp 后的。
        t_safe = t.clamp_min(1e-3)
        logits = logits_fn(x_t)
        loss = _a2d.a2d_loss(logits, ids, masked, t_safe, valid=valid)
        return loss, {"t_mean": float(t.mean())}
    return loss_fn
