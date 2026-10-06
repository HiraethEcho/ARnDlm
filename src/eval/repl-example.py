"""src/eval/repl-example.py -- src/eval/repl.py 的参考实现。

终端下直观看模型输出，两种模式：
  pretrain -- 续写模式：输入 -> encode(bos=True) -> 续 48 token（每轮独立, 无历史）
  chat     -- 对话模式：逐轮累积 [BOS] user\\n{q} [EOS] assistant\\n, EOS 停, 裁到 ctx 128

ckpt 自动识别存法：
  有 config.json  -> transformers Qwen3ForCausalLM.from_pretrained（trans 存法）
  只有 model.safetensors -> native 模型自建 + load（native 存法, lm_head 靠 tied 带上）

用法:
  .venv/bin/python src/eval/repl-example.py --ckpt checkpoints/ar-3.2m-native-smoke --mode pretrain
  .venv/bin/python src/eval/repl-example.py --ckpt checkpoints/ar-3.2m-native-sft-smoke --mode chat
REPL 命令: :q 退出 :m chat|pretrain :c 清历史 :t 0.5 :k 30 :h 帮助
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOS, EOS, PAD = 1, 2, 0


def load_model(ckpt: str | Path):
    """按目录内容自动选择加载路径。返回 (model, kind)。fp32 + eval（与 eval 口径一致）。"""
    ckpt = Path(ckpt)
    if not (ckpt / "model.safetensors").exists():
        raise FileNotFoundError(f"{ckpt} 里没有 model.safetensors")
    if (ckpt / "config.json").exists():
        from transformers import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(ckpt)
        kind = "hf"
    else:
        native = _load("src/models/native/model-example.py", "native_model_repl")
        cfg = _load("common/config-example.py", "config_example_repl").load("smoketest")
        model = native.build_native(cfg.MODEL, seed=None)
        from safetensors.torch import load_file
        sd = load_file(str(ckpt / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing  # tied 自动带上
        kind = "native"
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return model.to(dev).float().eval(), kind


@torch.no_grad()
def generate(model, kind: str, ids: list[int], max_new_tokens: int,
             temperature: float, top_k: int, max_context: int) -> list[int]:
    """给定前缀 token, 续到 EOS 或 max_new_tokens; 返回新生成部分（含 EOS）。"""
    dev = next(model.parameters()).device
    prefix = torch.tensor([ids], device=dev)
    if kind == "hf":
        window = prefix[:, -max_context:]
        out = model.generate(
            input_ids=window,
            attention_mask=torch.ones_like(window),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_k=top_k,
            eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS,
        )
        return out[0, window.shape[1]:].tolist()
    # native: 手写循环, 每步重跑窗口
    window = torch.tensor([ids[-max_context:]], device=dev)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(window).logits[0, -1].float()
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
        probs = torch.softmax(values / temperature, dim=-1)
        nxt = int(indices[torch.multinomial(probs, 1)])
        generated.append(nxt)
        if nxt == EOS:
            break
        window = torch.cat([window, torch.tensor([[nxt]], device=dev)], dim=1)[:, -max_context:]
    return generated


PRETRAIN_NEW = 48
CHAT_NEW = 64


def repl(ckpt, mode: str, temperature: float = 0.3, top_k: int = 50) -> None:
    model, kind = load_model(ckpt)
    tok = _load("common/tokenizer-example.py", "tokenizer_example_repl").Tok()
    max_context = model.cfg.context_length if kind == "native" else model.config.max_position_embeddings

    history: list[int] = [BOS]  # chat 模式的逐轮累积
    print(f"repl: {Path(ckpt).name} [{kind}] mode={mode} ctx={max_context} "
          f"temp={temperature} top_k={top_k}")
    print("命令: :q 退出  :m chat|pretrain  :c 清历史  :t <float>  :k <int>")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, arg = line.partition(" ")
            if cmd == ":q":
                break
            if cmd == ":m" and arg in ("chat", "pretrain"):
                mode = arg
                history = [BOS]
                print(f"mode -> {mode}（历史已清）")
                continue
            if cmd == ":c":
                history = [BOS]
                print("历史已清")
                continue
            if cmd == ":t":
                temperature = float(arg)
                print(f"temperature -> {temperature}")
                continue
            if cmd == ":k":
                top_k = int(arg)
                print(f"top_k -> {top_k}")
                continue
            if cmd == ":h":
                print("命令: :q 退出  :m chat|pretrain  :c 清历史  :t <float>  :k <int>")
                continue
            print(f"未知命令 {cmd}（:h 看帮助）")
            continue

        if mode == "chat":
            history += tok.encode(f"user\n{line}") + [EOS] + tok.encode("assistant\n")
            new = generate(model, kind, history[-max_context:], CHAT_NEW, temperature, top_k, max_context)
            history += new
            print("model>", tok.decode(new), flush=True)
        else:  # pretrain: 每轮独立续写
            ids = tok.encode(line, bos=True)
            new = generate(model, kind, ids, PRETRAIN_NEW, temperature, top_k, max_context)
            print("model>", tok.decode(new), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="chat", choices=["chat", "pretrain"])
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    repl(args.ckpt, args.mode, args.temperature, args.top_k)
