"""src/training/ar_trans-example.py -- src/training/ar_trans.py 的参考实现。

G1 格 1: transformers Trainer 走 AR pretrain（用包, 代码最少），产出 ar-<size>-trans。
所有训练数值（max_steps/warmup/cosine+min_lr/bf16/clip）只从 config 读, 不出现第二处数字。
Trainer 不暴露单步秒数 -> 吞吐用 callback 夹计时, train-only 口径是近似
（不含 dataloader 等待）; 逐 log 点的 loss 是 HF 的区间均值, 不是单步值。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainingArguments, TrainerCallback

# ---- 连字符文件名不能被 import: 显式装载同仓 example 模块（x.py 填完后用正常 import）----
_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_data = _load("common/data-example.py", "data_example_artrans")
_mx = _load("common/metrics-example.py", "metrics_example_artrans")
_te = _load("src/models/trans-example.py", "trans_example")
Corpus, Batches = _data.Corpus, _data.Batches
TpsMeter = _mx.TpsMeter
build = _te.build


class _Blocks(IterableDataset):
    """惰性 + 有序: 按顺序 yield Batches 纯函数的块。

    必须是 IterableDataset: Trainer 对 map-style 数据集默认 RandomSampler,
    会打乱批次顺序 —— 批次位置就推不回了 (C4 的根)。无长度数据集无 sampler,
    按序消费; max_steps 已显式给出, Trainer 不需要 len。
    """

    def __init__(self, batches: Batches, n_items: int):
        self.batches = batches
        self.n = n_items  # 不定义 __len__: 一旦有 len, Trainer 就走 RandomSampler

    def __iter__(self):
        for b in range(self.n // self.batches.B):
            batch = torch.from_numpy(self.batches.batch(b)).long()
            for w in range(self.batches.B):
                yield {"input_ids": batch[w], "labels": batch[w].clone()}


class _MeterCallback(TrainerCallback):
    """夹计时（train-only 近似口径）+ 把 HF 的 log 转给 main 的 ctx。"""

    def __init__(self, ctx, meter: TpsMeter, tokens_per_step: int, impl: str):
        self.ctx, self.meter, self.tps_tokens, self.impl = ctx, meter, tokens_per_step, impl
        self._t0 = None

    def on_step_begin(self, args, state, control, **kw):
        self._t0 = self.meter.now()

    def on_step_end(self, args, state, control, **kw):
        self.meter.add_train(self.tps_tokens, self.meter.now() - self._t0)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.ctx.log_step(self.impl, int(state.global_step) - 1, float(logs["loss"]),
                              float(logs.get("learning_rate", 0.0)),
                              float(logs.get("grad_norm", 0.0)), self.meter.report())


@torch.no_grad()
def sample(model, prompt_ids: list[int], max_new_tokens: int = 32,
           temperature: float = 0.3, top_k: int = 50,
           pad_id: int = 0, bos_id: int = 1, eos_id: int = 2) -> list[int]:
    was_training = model.training
    model.eval()
    prefix = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    generated = out[0, prefix.shape[1]:].tolist()
    if was_training:
        model.train()
    return generated


def run(cfg, steps: int, ctx, dataset=None, positioner=None,
        init_weights_path: str | Path | None = None) -> dict:
    """dataset/positioner: SFT 层的注入点 —— 缺省为 Corpus+Batches 纯函数顺序。"""
    t = cfg.TRAIN
    if dataset is None:
        corpus = Corpus(ctx.data_dir, block_size=cfg.BLOCK_SIZE, vocab=cfg.VOCAB,
                        max_blocks=cfg.DATA.max_blocks)
        batches = Batches(corpus, batch_size=t.batch_size, seed=t.seed)
        dataset = _Blocks(batches, n_items=steps * t.batch_size)
        positioner = batches
    tokens_per_step = t.batch_size * cfg.BLOCK_SIZE

    model = build(cfg.MODEL, seed=t.seed)
    if init_weights_path is not None:
        from safetensors.torch import load_file
        sd = load_file(str(Path(init_weights_path) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing  # tied 自动带上

    args = TrainingArguments(
        output_dir=str(ctx.tmp_dir),
        max_steps=steps,
        per_device_train_batch_size=t.batch_size,
        learning_rate=t.lr,
        lr_scheduler_type="cosine_with_min_lr",   # 与 common lr_at 同一公式: cosine 到 floor
        lr_scheduler_kwargs={"min_lr_rate": t.min_lr_ratio},
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        adam_beta1=t.beta1,
        adam_beta2=t.beta2,
        max_grad_norm=t.grad_clip,
        bf16=(cfg.DTYPE == "bfloat16"),
        seed=t.seed,
        dataloader_num_workers=0,
        # smoke 步数小于 log_every 时, 最后一四分之一至少出一个 loss 点, 否则 "loss 能降" 无从判定
        logging_steps=min(t.log_every, max(1, steps // 4)),
        save_strategy="no",          # 权重存盘由 main 统一管（safetensors + train_state）
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        log_level="error",
    )
    meter = TpsMeter()
    trainer = Trainer(model=model, args=args,
                      train_dataset=dataset,
                      callbacks=[_MeterCallback(ctx, meter, tokens_per_step, "trans")])
    from transformers.trainer_callback import PrinterCallback
    trainer.remove_callback(PrinterCallback)   # disable_tqdm 下它仍逐条 print log dict
    trainer.train()

    # HF 的 log 是区间均值, LossHistory 记的就是这些点（口径差异记进 summary）
    history = _mx.LossHistory()
    for rec in trainer.state.log_history:
        if "loss" in rec:
            history.add(int(rec.get("step", 0)) - 1, float(rec["loss"]))

    ctx.save_ckpt(trainer.model, "trans")
    return {"model": trainer.model, "optimizer": trainer.optimizer, "meter": meter,
            "history": history, "step": steps - 1,
            "data_position": positioner.position(steps - 1), "sample_fn": sample}
