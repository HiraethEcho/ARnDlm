"""models/hf_qwen/model.py -- 建模型 (种子在初始化之前打) 和 AR 采样。

对照 example/base/hf/model.py 的 build_model / sample_ar。
"""

from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from .config import build_qwen3_config
from utils.config import BOS, EOS, PROMPTS, ModelCfg
from utils.seed import seed_all
from utils.tokenizer import decode, encode


def build_model(cfg: ModelCfg, seed: int = 42) -> Qwen3ForCausalLM:
    # 种子必须在构造之前打, 否则两次 "同 seed" 的运行从不同权重开始
    seed_all(seed)
    return Qwen3ForCausalLM(build_qwen3_config(cfg))


@torch.no_grad()
def sample_ar(model: Qwen3ForCausalLM, prompts=PROMPTS,
              max_new_tokens: int = 32, temperature: float = 0.3,
              top_k: int = 50, prompt_ids: list | None = None) -> dict:
    """自回归采样, 走 generate; 结束后把 model 还原回 train 模式。"""
    was_training = model.training
    model.eval()

    ids, texts = [], []
    for prompt in (prompt_ids if prompt_ids is not None else
                   [*map(lambda p: encode(p, bos=True), prompts)]):
        # encode 不加 special token, BOS 由这里显式给出
        prompt_tensor = torch.tensor([prompt], device=model.device)
        out = model.generate(
            input_ids=prompt_tensor,
            attention_mask=torch.ones_like(prompt_tensor),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=BOS,
            eos_token_id=EOS,
            pad_token_id=0,
        )
        generated = out[0, prompt_tensor.shape[1]:]
        ids.append(generated.tolist())
        texts.append(decode(generated))
    if was_training:
        model.train()
    return {"prompts": list(prompts), "ids": ids, "texts": texts,
            "params": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                       "top_k": top_k}}
