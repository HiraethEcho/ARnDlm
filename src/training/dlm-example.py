"""src/training/dlm-example.py -- src/training/dlm.py 的参考实现。

Direct-DLM：**不经过 AR**，随机初始化一张 Qwen3，直接在双向 + 加权 masked CE 上训
（G2 的 dlm 侧）。两条 impl 与 a2d 同义：
  trans -> 手写 masked CE（common/a2d-example 的 objective）
  dllm  -> dllm 包的 MDLMTrainer.compute_loss

与 a2d_dllm/a2d_mini 的关系：**objective 共享**（common/a2d-example.py），差别只在
起点——a2d 从 AR/SFT 权重出发（零权重改动），dlm 从随机初始化出发。此前这段逻辑
寄生在 main-example.run_a2d 的 base=None 分支里，对拍基准缺一格，本文件补齐。

结论必须报 rep-4（notes/03 §11: 复读而 loss 照降是 DLM 最典型失败）。
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


_data = _load("common/data-example.py", "data_example_dlm")
_dev = _load("common/device-example.py", "device_example_dlm")
_mx = _load("common/metrics-example.py", "metrics_example_dlm")
_a2d = _load("common/a2d-example.py", "a2d_example_dlm")
_te = _load("src/models/trans-example.py", "trans_example_dlm")
_mini = _load("src/training/a2d_mini-example.py", "a2d_mini_example_dlm")


def make_loss_fn(impl: str, model):
    """Direct-DLM 的 loss 闭包。trans = 手写 objective；dllm = 调包。"""
    if impl == "trans":
        return _mini.make_loss_fn(model, "trans")
    if impl == "dllm":
        dllm_ex = _load("src/training/a2d_dllm-example.py", "a2d_dllm_example_dlm")
        return dllm_ex.make_loss_fn(dllm_ex.DLLMLoss())
    raise ValueError(f"dlm 的 impl 是 trans|dllm（SPEC §1.2: dlm 没有 native），收到 {impl!r}")


def make_sampler(model, cfg):
    """DLM 采样: 块状置信度揭示（双向, 无 KV cache）。返回 (prompt_ids)->ids。"""
    logits_fn = _mini.dlm_forward_trans(model)

    def sample(prompt_ids: list[int], max_new_tokens: int = 32,
               block_size: int = 8, steps: int = 8,
               temperature: float = 0.3, top_k: int = 50) -> list[int]:
        dev = next(model.parameters()).device
        prefix = torch.tensor([prompt_ids], device=dev)
        out = _a2d.sample_dlm(logits_fn, prefix, max_new_tokens, mask_id=27,
                              block_size=block_size, steps=steps,
                              temperature=temperature, top_k=top_k,
                              max_context=cfg.BLOCK_SIZE)
        gen = out[0, len(prompt_ids):].tolist()
        if 2 in gen:
            gen = gen[:gen.index(2) + 1]
        return gen
    return sample


def run(cfg, steps: int, ctx, impl: str = "trans") -> dict:
    """Direct-DLM smoke 训练循环。与 a2d 循环同构, 唯一差别是 base=None。"""
    t = cfg.TRAIN
    device = _dev.pick_device()
    corpus = _data.Corpus(ctx.data_dir, block_size=cfg.BLOCK_SIZE, vocab=cfg.VOCAB,
                          max_blocks=cfg.DATA.max_blocks)
    batches = _data.Batches(corpus, batch_size=t.batch_size, seed=t.seed)

    model = _te.build(cfg.MODEL, seed=t.seed).to(device)  # 随机初始化（不是 A2D）
    loss_fn = make_loss_fn(impl, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr,
                                  betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)
    meter, history = _mx.TpsMeter(), _mx.LossHistory()
    tokens_per_step = t.batch_size * cfg.BLOCK_SIZE

    model.train()
    for step in range(steps):
        lr_now = ctx.lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        batch = torch.from_numpy(batches.batch(step)).long().to(device)
        torch.cuda.synchronize()
        t0 = meter.now()
        with _dev.autocast(cfg.DTYPE):
            loss, _aux = loss_fn(model, batch)
        loss.backward()
        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        meter.add_train(tokens_per_step, meter.now() - t0)
        history.add(step, loss.item())
        if step % t.log_every == 0 or step == steps - 1:
            ctx.log_step(impl, step, loss.item(), lr_now, float(clip), meter.report())

    ctx.save_ckpt(model, impl)
    return {"model": model, "optimizer": optimizer, "meter": meter, "history": history,
            "step": steps - 1, "data_position": batches.position(steps - 1),
            "loss_fn": loss_fn, "sample_fn": make_sampler(model, cfg)}
