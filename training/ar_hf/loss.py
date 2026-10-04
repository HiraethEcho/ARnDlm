"""training/ar_hf/loss.py -- 下一个 token 的 cross entropy (打包块上)。

对照 example/common/losses.py::ar_loss。SFT 版的 [2, B, L] 批次在 B2 再加。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ar_loss(model, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """batch 是 [B, L] 的打包块; 每块 L-1 个预测位置。返回 (loss, extra)。"""
    # 输入是前 L-1 个, 标签是后 L-1 个
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs).logits
    # cross_entropy 要二维输入, logits 本身不连续所以用 reshape
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, {"target_tokens": int(targets.numel())}
