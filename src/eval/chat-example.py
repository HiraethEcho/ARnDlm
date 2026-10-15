"""src/eval/chat-example.py -- scripts/chat.py 的参考实现。

「能 chat」四门槛（SPEC 6.3）的测量与报告:
  1. assistant 段 eval loss <= 1.5x 同条件 AR-SFT（参照缺失时报 pending, 不装绿）
  2. rep-4 < 0.5（>=64 token 生成; 正式评测几百 token x >=5 prompt, 这里是 smoke 口径）
  3. 三个 prompt 在 assistant 段后输出非空、格式正确
  4. 不回声提问（回答不含 prompt 原文）
产出 dict 报告; 判定阈值写在报告里, 不替用户做「合格」结论。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def assistant_eval_loss(model, sft_dir: str | Path, n_blocks: int = 32,
                        bidirectional: bool = False) -> float:
    """只在 assistant 区间上算的 eval loss（fp32; 与训练同一 span 语义）。

    bidirectional=True（DLM 系）: trans 走 4D 全零 mask; native 走 set_causal 开关（用完恢复）。"""
    sft = _load("src/training/sft-example.py", "sft_example_chat").SFTBlocks(sft_dir)
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if bidirectional and hasattr(model, "set_causal"):
        model.set_causal(False)

    def forward(x):
        if bidirectional and not hasattr(model, "set_causal"):
            m4 = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1],
                             device=dev, dtype=model.dtype)
            return model(x, attention_mask=m4).logits
        return model(x).logits

    total, count = 0.0, 0
    nb = min(n_blocks, sft.n_blocks)
    for b in range(sft.n_blocks - nb, sft.n_blocks):
        ids = torch.from_numpy(np.asarray(sft.blocks[b * sft.bs:(b + 1) * sft.bs],
                                          dtype=np.int64))[None].long().to(dev)
        labels = torch.from_numpy(sft.labels(b))[None].long().to(dev)
        logits = forward(ids).float()
        # shift 后: logits[:, :-1] 预测 labels[:, 1:]; -100 位置不计
        lo = logits[:, :-1].reshape(-1, logits.shape[-1])
        tg = labels[:, 1:].reshape(-1)
        keep = tg != -100
        if keep.any():
            total += torch.nn.functional.cross_entropy(lo[keep], tg[keep],
                                                       reduction="sum").item()
            count += int(keep.sum())
    if count == 0:
        raise ValueError("eval 块里没有任何监督 token, assistant loss 无从算起")
    if bidirectional and hasattr(model, "set_causal"):
        model.set_causal(True)
    if was_training:
        model.train()
    return total / count


def check_thresholds(prompts: list[str], texts: list[str], rep4s: list[float],
                     loss: float, ref_ar_sft_loss: float | None,
                     min_new_tokens: int = 8) -> dict:
    """四门槛判定。thresholds 只报告数字与 bool, 不汇总成单一「分」。"""
    checks = []

    # 门槛 1: assistant eval loss 比
    if ref_ar_sft_loss is None:
        checks.append({"name": "t1_loss_ratio", "value": loss, "ref": None,
                       "pass": None, "note": "pending: 无 AR-SFT 参照（P6 DLM-SFT 后才有）"})
    else:
        ratio = loss / ref_ar_sft_loss
        checks.append({"name": "t1_loss_ratio", "value": loss, "ref": ref_ar_sft_loss,
                       "ratio": ratio, "pass": ratio <= 1.5})

    # 门槛 2: rep-4
    worst = max(rep4s)
    checks.append({"name": "t2_rep4", "value": worst, "threshold": 0.5, "pass": worst < 0.5})

    # 门槛 3: assistant 段后非空且格式正确（本模板: 回答就是 assistant\\n 之后的内容）
    fmt_ok = []
    for p, t in zip(prompts, texts):
        answer = t.split("assistant")[-1].strip() if "assistant" in t else t.strip()
        fmt_ok.append(len(answer) >= min_new_tokens and "user" not in answer)
    checks.append({"name": "t3_format", "value": fmt_ok, "pass": all(fmt_ok)})

    # 门槛 4: 不回声
    echo = []
    for p, t in zip(prompts, texts):
        q = p.replace("user\n", "")
        echo.append(bool(q) and q in t)
    checks.append({"name": "t4_no_echo", "value": echo, "pass": not any(echo)})

    return {"assistant_eval_loss": loss, "ref_ar_sft_loss": ref_ar_sft_loss,
            "rep4": rep4s, "texts": texts, "checks": checks,
            "caveat": "smoke 口径: 短生成 rep-4 会跳, 正式评测要几百 token x >=5 prompt"}


def report(model, sft_dir, prompts: list[str], sample_fn, decode_fn,
           ref_ar_sft_loss: float | None = None, max_new_tokens: int = 64,
           bidirectional: bool = False) -> dict:
    """一步到位: assistant eval loss + 采样 + 四门槛。sample_fn/decode_fn 由调用方注入。

    bidirectional=True（DLM 系）: 双向 forward 的 assistant 段 eval loss —— 与该模型
    **训练时的 forward 语义一致**。因果模型评 DLM 权重会静默少一半上下文。"""
    loss = assistant_eval_loss(model, sft_dir, bidirectional=bidirectional)
    texts, rep4s = [], []
    mx = _load("common/metrics-example.py", "metrics_example_chat")
    for q in prompts:
        ids = sample_fn(q, max_new_tokens=max_new_tokens)
        texts.append(decode_fn(ids))
        rep4s.append(mx.rep4(ids))
    return check_thresholds(prompts, texts, rep4s, loss, ref_ar_sft_loss)
