"""src/training/ar_native-example.py -- src/training/ar_native.py 的参考实现。

G1 格 2: native 包走同一条 main，产出 ar-<size>-native。
自管逐步循环; 每一步的秒数只进 train-only 分母（TpsMeter.add_train）。
批次顺序来自 common/data-example 的 Batches（f(seed, step) 纯函数, C4 的根）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- 连字符文件名不能被 import: 显式装载同仓 example 模块（x.py 填完后用正常 import）----
_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_data = _load("common/data-example.py", "data_example")
_dev = _load("common/device-example.py", "device_example")
_mx = _load("common/metrics-example.py", "metrics_example")
_mm = _load("src/models/native/model-example.py", "native_model_example")
Corpus, Batches = _data.Corpus, _data.Batches
autocast, pick_device, to_dtype = _dev.autocast, _dev.pick_device, _dev.to_dtype
LossHistory, TpsMeter = _mx.LossHistory, _mx.TpsMeter
build_native = _mm.build_native


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """因果 LM loss = HF labels 语义: 第 i 个 logits 预测第 i+1 个 token。
    ignore_index=-100: SFT 的 assistant 区间外标签（AR 路径不含 -100, 不受影响）。"""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           targets[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def sample(model, prompt_ids: list[int], max_new_tokens: int = 32,
           temperature: float = 0.3, top_k: int = 50, max_context: int | None = None,
           eos_id: int = 2) -> list[int]:
    """AR 采样, 无 KV cache: 每步重跑前缀（smoke 尺寸下便宜, 且不引 transformers 机制）。
    生成到 EOS 即停（EOS 本身也进返回值, 调用方决定是否展示）。"""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    prefix = torch.tensor([prompt_ids], device=dev)
    max_context = max_context or model.cfg.context_length
    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = prefix[:, -max_context:]
        logits = model(window, is_causal=True).logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
            probs = torch.softmax(values / temperature, dim=-1)
            nxt = int(indices[torch.multinomial(probs, 1)])
        generated.append(nxt)
        if nxt == eos_id:
            break
        prefix = torch.cat([prefix, torch.tensor([[nxt]], device=dev)], dim=1)
    if was_training:
        model.train()
    return generated


def run(cfg, steps: int, ctx, resume_path: str | Path | None = None,
        batcher=None, labels_fn=None, init_weights_path: str | Path | None = None) -> dict:
    """训练循环。ctx 由 main-example 提供（IO: eval/summary/ckpt 路径）。

    resume_path 给出时: 从该目录的 checkpoint 权重 + train_state.pt 续跑,
    目标 steps 是**全局步数**（如 resume 自 step20, 传 steps=30 就再训 10 步）。
    batcher/labels_fn: SFT 层的注入点（sft-example.py）—— 缺省为 Corpus+Batches 纯函数。
    """
    m = cfg.MODEL
    t = cfg.TRAIN
    device = pick_device()
    dtype = to_dtype(cfg.DTYPE)

    if batcher is None:
        corpus = Corpus(ctx.data_dir, block_size=cfg.BLOCK_SIZE, vocab=cfg.VOCAB,
                        max_blocks=cfg.DATA.max_blocks)
        batches = Batches(corpus, batch_size=t.batch_size, seed=t.seed)
    else:
        batches = batcher  # SFT 等注入的批次器（接口: .batch(step) / .position(step)）

    model = build_native(m, seed=t.seed).to(device)
    if init_weights_path is not None:
        # SFT/A2D 的起点: 从 AR 权重出发。只拿权重, 不拿 optimizer/RNG（那是 resume 的事）
        from safetensors.torch import load_file
        sd = load_file(str(Path(init_weights_path) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing  # tied 自动带上
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr,
                                  betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)

    start_step = 0
    if resume_path is not None:
        state = _load("src/training/state-example.py", "state_example_arnative")
        st = state.load_state(Path(resume_path) / "train_state.pt", optimizer)
        # 权重也要从 checkpoint 来, 不能用新种子初始化（否则 resume 不是同一个模型）
        from safetensors.torch import load_file
        sd = load_file(str(Path(resume_path) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # lm_head.weight 与 embed 共享存储, 不在文件里 -> 由 tie 自动带上
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing
        if st["data_position"] != batches.position(st["step"]):
            raise ValueError(f"data_position 不自洽: {st} vs step {st['step']} -> "
                             f"{batches.position(st['step'])}")
        start_step = st["step"] + 1

    meter = TpsMeter()
    history = LossHistory()
    tokens_per_step = t.batch_size * cfg.BLOCK_SIZE

    model.train()
    for step in range(start_step, steps):
        lr_now = ctx.lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        batch = torch.from_numpy(batches.batch(step)).long().to(device, non_blocking=True)
        targets = batch if labels_fn is None else torch.from_numpy(
            np.asarray(labels_fn(step, batch))).long().to(device)
        torch.cuda.synchronize()
        t0 = meter.now()

        with autocast(cfg.DTYPE):
            logits = model(batch).logits
            loss = ce_loss(logits, targets)

        loss.backward()
        grad_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        meter.add_train(tokens_per_step, meter.now() - t0)
        history.add(step, loss.item())

        if step % t.log_every == 0 or step == steps - 1:
            ctx.log_step("native", step, loss.item(), lr_now,
                         float(grad_clip), meter.report())

    ctx.save_ckpt(model, "native")
    return {"model": model, "optimizer": optimizer, "meter": meter,
            "history": history, "step": steps - 1, "data_position": batches.position(steps - 1),
            "sample_fn": sample}