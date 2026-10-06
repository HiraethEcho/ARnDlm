"""scripts/collect_results.py -- 读取全部 summary.json + 跑四门槛报告 -> smoke RESULTS.md。

四门槛模型清单内置（AR-SFT 是门槛 1 的参照系）。用的采样: AR 格子 = generate/native 循环;
DLM 格子 = sample_dlm（双向块状揭示）。ASSISTANT eval: AR = 因果, DLM = 双向。

用法: .venv/bin/python scripts/collect_results.py [--out RESULTS-smoke.md]
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("c", "common/config-example.py")
chat = load("chat", "src/eval/chat-example.py")
mini = load("mini", "src/training/a2d_mini-example.py")
native_loop = load("nloop", "src/training/ar_native-example.py")
sft_x = load("sftx", "src/training/sft-example.py")
tok_mod = load("tok", "common/tokenizer-example.py")

PROMPTS = ["什么是深度学习？", "帮我写一句诗。", "今天天气怎么样？"]
# (ckpt 名, 是否 DLM, 家族) —— AR-SFT 在前（t1 的参照）
CELLS = [
    ("ar-3.2m-trans-sft-smoke", False, "ar"),
    ("ar-3.2m-native-sft-smoke", False, "ar"),
    ("ar-3.2m-trans-sft-mini_a2d-sft-smoke", True, "ar"),
    ("ar-3.2m-trans-sft-dllm_a2d-sft-smoke", True, "ar"),
    ("ar-3.2m-mini_a2d-sft-smoke", True, "ar"),
    ("ar-3.2m-dllm_a2d-sft-smoke", True, "ar"),
    ("dlm-3.2m-trans-sft-smoke", True, "dlm"),
    ("dlm-3.2m-dllm-sft-smoke", True, "dlm"),
]


def load_model(name, is_dlm):
    """ckpt 自动识别（与 repl 同规则）；DLM 时返回 (model, logits_fn, kind)。"""
    ck = ROOT / "checkpoints" / name
    cfg = cfg_mod.load("smoketest")
    if (ck / "config.json").exists():
        from transformers import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(ck)
        kind = "hf"
    else:
        ne = load("ne", "src/models/native/model-example.py")
        model = ne.build_native(cfg.MODEL, seed=None)
        from safetensors.torch import load_file
        sd = load_file(str(ck / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing
        kind = "native"
    dev = torch.device("cuda")
    return model.to(dev).float().eval(), kind


def make_sampler(model, kind, is_dlm):
    dev = next(model.parameters()).device
    tok = tok_mod.Tok()
    if is_dlm:
        logits_fn = (lambda x: model(x, is_causal=False).logits) if kind == "native" \
            else (lambda x: model(x, attention_mask=torch.zeros(
                x.shape[0], 1, x.shape[1], x.shape[1], device=dev, dtype=model.dtype)).logits)
        a2d = load("a2d", "common/a2d-example.py")

        def sample(q, max_new_tokens=64, **_):
            prefix = sft_x.encode_prompt(q)
            win = prefix[-128:]
            out = a2d.sample_dlm(logits_fn, torch.tensor([win], device=dev),
                                 min(max_new_tokens, 128 - len(win)), max_context=128)
            new = out[0, len(win):].tolist()
            return new[:-1] if new and new[-1] == 2 else new
        return sample
    if kind == "native":
        def sample(q, max_new_tokens=64, **_):
            ids = native_loop.sample(model, sft_x.encode_prompt(q), max_new_tokens=max_new_tokens)
            return ids[:-1] if ids and ids[-1] == 2 else ids
        return sample

    def sample(q, max_new_tokens=64, **_):  # HF generate
        was = model.training
        model.eval()
        prefix = torch.tensor([sft_x.encode_prompt(q)], device=dev)
        out = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                             max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3, top_k=50,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0)
        ids = out[0, prefix.shape[1]:].tolist()
        if was:
            model.train()
        return ids
    return sample


def main():
    out_name = "RESULTS-smoke.md"
    rows, reports = {}, {}
    ref_loss = None
    for name, is_dlm, family in CELLS:
        if not (ROOT / "checkpoints" / name).exists():
            print(f"skip {name}: 无 ckpt")
            continue
        print(f"== {name}")
        summary = json.loads((ROOT / "logs/run" / name / "summary.json").read_text())
        model, kind = load_model(name, is_dlm)
        sample = make_sampler(model, kind, is_dlm)
        tok = tok_mod.Tok()
        if family == "ar" and not is_dlm and ref_loss is None:
            ref_loss = chat.assistant_eval_loss(model, ROOT / "data/prepared/sft-mini-6400")
        rep = chat.report(model, ROOT / "data/prepared/sft-mini-6400", PROMPTS,
                          sample, tok.decode, ref_ar_sft_loss=ref_loss)
        reports[name] = rep
        (ROOT / "logs/run" / name / "chat_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2))
        checks = {c["name"]: c.get("pass") for c in rep["checks"]}
        rows[name] = {
            "eval": rep["assistant_eval_loss"], "rep4": rep["rep4"],
            "t1": checks["t1_loss_ratio"], "t2": checks["t2_rep4"],
            "t3": checks["t3_format"], "t4": checks["t4_no_echo"],
            "steps": summary.get("steps"), "family": family, "dlm": is_dlm,
        }
        print(f"   eval={rep['assistant_eval_loss']:.4f} rep4={rep['rep4']} "
              f"t2={checks['t2_rep4']} t3={checks['t3_format']} t4={checks['t4_no_echo']}")

    lines = ["# smoke RESULTS（3.2M, 全格子 200 步起步）", "",
             f"生成: collect_results.py 实测; t1 参照 = ar-3.2m-trans-sft-smoke "
             f"(assistant eval {ref_loss:.4f})" if ref_loss else "生成: collect_results.py", "",
             "| 格子 | steps | assistant eval | rep-4(各prompt) | t1 ≤1.5x | t2 rep-4<0.5 | t3 格式 | t4 不回声 |",
             "|---|---:|---:|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['steps']} | {r['eval']:.4f} | "
            f"{', '.join(f'{v:.2f}' for v in r['rep4'])} | "
            f"{'—' if r['t1'] is None else ('✅' if r['t1'] else '❌')} | "
            f"{'✅' if r['t2'] else '❌'} | {'✅' if r['t3'] else '❌'} | "
            f"{'✅' if r['t4'] else '❌'} |")
    lines.append("")
    lines.append("_t1 = assistant eval loss / AR-SFT 参照 ≤ 1.5；t2 = rep-4 < 0.5；"
                 "t3 = 回答非空且格式正确；t4 = 不回声。数据口径：sft-mini-6400 "
                 "（52 万 token），smoke 只验管线不评判质量，200 步起步不达标追 400/800。_")
    (ROOT / out_name).write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_name}")


if __name__ == "__main__":
    main()
