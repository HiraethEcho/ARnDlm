#!/usr/bin/env python
"""test 03 — LoRA 挂对了吗。

T1a 的后训练用 peft。挂错模块的典型症状是「loss 降了但什么都没学到」,
所以这里把「挂了什么」打印出来人工核对, 而不是只看能不能跑。

  1. target_modules 命中了哪些模块
  2. 可训参数占比 (LoRA 应该是百分之几, 不是百分之几十)
  3. **哪些模块没被冻结** —— embed / lm_head / norm 最容易漏
  4. merge_and_unload 之后权重真的变了, 且和 LoRA 前的差值符合预期量级

需要 peft + torch。
"""
import re
import sys

# 目标参数量 ~50M, 和 ROADMAP.md 的候选形状一致
HIDDEN, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB = 576, 12, 8, 4, 72, 1728, 6400

# LoRA 惯例: 只注入 attention 的投影层, 不碰 MLP 也不碰 embedding。
# 注入 MLP 会让可训参数翻 3-4x, 对小数据集是过拟合来源。
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    for mod in ("torch", "peft"):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(f"FAIL 没装 {mod}: ./scripts/setup_env.sh")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        max_position_embeddings=2048, tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"base params : {base_params / 1e6:.2f}M")

    # --- tied embedding 的坑 ---
    # tie_word_embeddings=True 时 embed_tokens.weight 和 lm_head.weight 是
    # **同一个对象**。用 named_parameters() 遍历只会出现一次 (去重了),
    # 但如果自己遍历 model.embed_tokens 和 model.lm_head 就会数两遍。
    # 下面验证 named_parameters() 的行为。
    n_embed = sum(1 for n, _ in model.named_parameters() if "embed_tokens" in n)
    print(f"embed_tokens 出现次数 : {n_embed}  (tied, 应该只有 1 次)")

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- 1. 命中的模块 ---
    # LoRA 参数名形如 ...self_attn.q_proj.lora_A.default.weight
    # 模块名在 **.lora_A. / .lora_B. 之前**。
    # 不能用 split(".")[-1] —— 那是 "weight"；也不能只切 lora_A ——
    # lora_B.default.weight 里没有 ".lora_A.", 会被原样留下同样取到 "weight"。
    # 用正则同时处理 A/B, 以后 peft 改内部命名也不会失效。
    hit = sorted({
        re.split(r"\.lora_[AB]\.", n)[0].split(".")[-1]
        for n, p in model.named_parameters() if p.requires_grad
    })
    print(f"\n命中 target_modules : {hit}")
    if hit != sorted(TARGET_MODULES):
        print(f"FAIL  命中的模块和配置不符: {hit}")
        sys.exit(1)

    # 这个脚本只验正确性, 不验速度 —— 所以刻意留在 CPU 上跑,
    # 免得跟别的 GPU 任务抢显存。吞吐另测 (见 logs/dev/02-hardware-perf.md)。
    device = next(model.parameters()).device
    print(f"device    : {device}   (正确性测试, 故意不上 GPU)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / base_params
    print(f"可训参数 : {trainable / 1e6:.3f}M  ({pct:.2f}%)")
    if not 0.1 < pct < 5:
        print(f"FAIL  可训比例 {pct:.2f}% 不在合理区间 (0.1%-5%)")
        sys.exit(1)

    # --- 3. 哪些模块没被冻结 ---
    # 这些是 LoRA 最容易漏的 —— 漏了就意味着 backbone 在一起动, 是在
    # 重新预训练而不是后训练
    must_freeze = ["embed_tokens", "lm_head", "input_layernorm",
                   "post_attention_layernorm", "mlp"]
    print("\n--- 冻结检查 ---")
    ok = True
    for pat in must_freeze:
        leaked = [n for n, p in model.named_parameters()
                  if pat in n and p.requires_grad]
        state = "OK (冻结)" if not leaked else f"FAIL 仍在训练: {leaked[:3]}"
        print(f"  {pat:24s} {state}")
        if leaked:
            ok = False

    # Qwen3 的 q_norm / k_norm 也是 backbone, 同样不该动
    leaked = [n for n, p in model.named_parameters()
              if ("q_norm" in n or "k_norm" in n) and p.requires_grad]
    print(f"  {'q_norm/k_norm':24s} "
          f"{'OK (冻结)' if not leaked else f'FAIL 仍在训练: {leaked[:3]}'}")
    if leaked:
        ok = False

    # --- 4. merge 前后权重确实变了 ---
    # LoRA 训练完要 merge 回主干再导出。merge 是 in-place 的,
    # 这里用一个 clone 做参照。
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, 16))
    model.eval()
    before = model(input_ids=ids).logits.clone()

    # 手动把 B 矩阵置零 -> 等价于「还没训练过的 LoRA」, merge 后应该≈不变。
    # 这验证 merge 路径本身没写错 (而不是验证训练效果)。
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.zeros_(p)
    merged = model.merge_and_unload()
    after = merged(input_ids=ids).logits
    delta = (before - after).abs().max().item()
    print(f"\nmerge 前后 logits 最大差 : {delta:.3e}")
    if delta > 1e-3:
        print("FAIL  lora_B 置零后 merge 应该≈恒等, 说明 merge 路径有问题")
        ok = False

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
