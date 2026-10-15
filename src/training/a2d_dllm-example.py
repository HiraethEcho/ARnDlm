"""src/training/a2d_dllm-example.py -- a2d/a2d_dllm.py 的参考实现。

A2D，调 dllm 包（G5 的 dllm 侧）。**故意不 import 手写的 corrupt/a2d_loss** ——
这一格存在的意义就是跑上游路径，复制 objective 会让 C2 的对照失效。

哈尼斯：
* 模型 = A2DQwen3LMHeadModel（patched modeling_qwen3，因果 mask 已删），或
  直接用我们的 trans model —— compute_loss 只需要 model(input_ids, attention_mask)
  双向出 logits。本实现走后者（少一份 config 拷贝，C5 由 M14 单独测）。
* loss = MDLMTrainer.compute_loss 真调。构造完整 Trainer 要 dataset/tokenizer/
  output 目录，都不在被测路径上 —— __new__ 造对象、填 compute_loss 读的六个属性。
* 采样 = MDLMSampler（smoke 里先用手写 sample_dlm 保持口径统一，dllm sampler
  留给正式格子）。

transformers 4.57（SPEC §5）；5.x 改了 create_causal_mask 签名，接不上。
"""
from __future__ import annotations

import torch

MASK, BOS, EOS = 27, 1, 2  # SPEC 3.2（与 tokenizer/config/example 同源，不 import 连字符文件）


class _ProcessingStub:
    """MDLMTrainer.compute_loss 从 tokenizer 读的四个属性。"""

    def __init__(self, mask_token_id: int = MASK, bos_token_id: int = BOS,
                 eos_token_id: int = EOS):
        self.padding_side = "right"
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class _MeterStub:
    def update(self, **kwargs) -> None:
        pass  # smoke 不做 dllm 侧指标聚合；loss/吞吐走 common.metrics


class DLLMLoss:
    """dllm 的 MDLMTrainer.compute_loss，不构造 Trainer。

    与手写侧（a2d_mini-example）C2 可比的口径对齐：
      time_epsilon = 1e-3（两侧同值）
      loss_weight_type="scheduler" -> weight = 1/(t+1e-6)（LinearAlphaScheduler，
          与手写 1/(t+EPS) 同公式同 EPS）—— 两侧同权
      loss_norm_type="token"       -> ÷ valid tokens（与手写同分母语义）
      RNG 消耗顺序                  t=rand(b) -> mask=rand(b,l)，与 sample_t -> corrupt 同序
    """

    def __init__(self, time_epsilon: float = 1e-3,
                 loss_weight_type: str = "scheduler",
                 loss_norm_type: str = "token"):
        from dllm.core.schedulers import LinearAlphaScheduler
        from dllm.core.trainers.mdlm import MDLMTrainer

        self.trainer = MDLMTrainer.__new__(MDLMTrainer)
        self.trainer.scheduler = LinearAlphaScheduler()
        self.trainer.time_epsilon = time_epsilon
        self.trainer.loss_weight_type = loss_weight_type
        self.trainer.loss_norm_type = loss_norm_type
        self.trainer.right_shift_logits = False
        self.trainer.processing_class = _ProcessingStub()
        self.trainer.meter = _MeterStub()

    def __call__(self, model, batch: torch.Tensor) -> torch.Tensor:
        # [B, L] pretrain/A2D 原样过；[B, 2, L] SFT batch 拆 labels —— dllm 自己
        # 推 maskable_mask = labels != -100 并与随机掩码相交，assistant-only
        # 训练不需要改上游包。
        ids, labels = (batch[:, 0], batch[:, 1]) if batch.dim() == 3 else (batch, batch)
        # 4D 全零 mask = 双向（SPEC §5 的 4.57 做法）。不加这个, 普通 Qwen3 收到
        # attention_mask=None 会走因果 forward —— dllm 侧就偷偷比手写侧少一半上下文。
        # （若用 dllm 自带的 A2DQwen3LMHeadModel（patched 后缺 mask 即双向）, 则可省。）
        mask4d = torch.zeros(ids.shape[0], 1, ids.shape[1], ids.shape[1],
                             device=ids.device, dtype=next(model.parameters()).dtype)
        return self.trainer.compute_loss(
            model, {"input_ids": ids, "labels": labels, "attention_mask": mask4d})


def make_loss_fn(loss: DLLMLoss):
    def loss_fn(model, batch: torch.Tensor):
        return loss(model, batch), {"source": "dllm.MDLMTrainer.compute_loss"}
    return loss_fn
