"""src/training/main-example.py -- src/training/main.py 的参考实现。

训练主干: 读 config -> impl 装载 -> 训练 -> eval -> 采样 -> 存盘 -> summary。
换 trans/native 不改这里; eval/采样/存盘/命名只有这一份, 两条路线不会漂移。

跑法:
  .venv/bin/python src/training/main-example.py --tier smoketest --impl native --steps 20
  .venv/bin/python src/training/main-example.py --tier smoketest --impl trans  --steps 20

命名 (SPEC 1.2): ar-<size>-<impl>[-smoke]; size 用解析参数量 (3.2m / 26.9m)。
smoke tier 自动加 -smoke 后缀; 产物: checkpoints/<name>/ + logs/run/<name>/。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = _load("common/config-example.py", "config_example")
tok_mod = _load("common/tokenizer-example.py", "tokenizer_example")
mx_mod = _load("common/metrics-example.py", "metrics_example")
data_mod = _load("common/data-example.py", "data_example_main")


def analytic_params(m) -> int:
    """Qwen3 参数量解析式 (与 test/06 的实测 3,213,056 互为校验)。tied 只算一次。"""
    per_layer = (m.hidden_size * m.num_heads * m.head_dim        # q_proj
                 + m.hidden_size * m.num_heads * m.head_dim      # o_proj
                 + 2 * m.hidden_size * m.num_kv_heads * m.head_dim  # k/v_proj
                 + 2 * m.head_dim                                # q/k norm
                 + 3 * m.hidden_size * m.intermediate_size       # gate/up/down
                 + 2 * m.hidden_size)                            # 两个 layernorm
    n = m.vocab_size * m.hidden_size + m.num_layers * per_layer + m.hidden_size
    return n


class Ctx:
    """impl 拿到的 IO 句柄: 计时点日志 / checkpoint / 采样 / eval。"""

    def __init__(self, cfg, name: str, data_dir: Path, run_dir: Path, ckpt_dir: Path):
        self.cfg, self.name = cfg, name
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.ckpt_dir = ckpt_dir
        self.tmp_dir = run_dir / "hf_tmp"
        self._csv = run_dir / "metrics.csv"
        self._t0 = time.monotonic()
        self._rows = 0
        self._csv_head()

    def lr_at(self, step: int) -> float:
        return cfg_mod.lr_at(self.cfg, step)

    def _csv_head(self):
        with open(self._csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr", "grad_norm", "train_tps", "wall_tps"])

    def log_step(self, impl: str, step: int, loss: float, lr: float,
                 grad_norm: float, tps: dict) -> None:
        with open(self._csv, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{loss:.4f}", f"{lr:.3e}", f"{grad_norm:.3f}",
                                    f"{tps['train_tps']:.0f}", f"{tps['wall_tps']:.0f}"])
        self._rows += 1
        print(f"[{self.name} #{step:5d}] loss={loss:.4f} lr={lr:.2e} "
              f"train={tps['train_tps']:.0f} tok/s wall={tps['wall_tps']:.0f} tok/s", flush=True)

    def save_ckpt(self, model, impl: str) -> Path:
        """HF 模型走 save_pretrained; native 纯 torch 模型手动存 safetensors。

        tied 权重 (lm_head.weight 共享 embed_tokens.weight 的存储) 会让
        safetensors 拒绝 —— 与 transformers 同一处理: 只存一份, 关系记进 metadata。
        """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(self.ckpt_dir, safe_serialization=True)
            return self.ckpt_dir
        from safetensors.torch import save_file
        sd = model.state_dict()
        seen: dict[int, str] = {}
        keep, shared = {}, []
        for k, v in sd.items():
            ptr = v.data_ptr()
            if ptr in seen:
                shared.append([k, seen[ptr]])
            else:
                seen[ptr] = k
                keep[k] = v.detach().cpu().contiguous()
        save_file(keep, str(self.ckpt_dir / "model.safetensors"),
                  metadata={"shared_tensors": __import__("json").dumps(shared),
                            "format": "pt"})
        return self.ckpt_dir

    @torch.no_grad()
    def eval_loss(self, model, eval_blocks: int = 32) -> float:
        """fp32 eval（对拍硬要求）; eval 秒数不进 train-only。"""
        was_training = model.training
        model.eval()
        corpus = data_mod.Corpus(self.data_dir, block_size=self.cfg.BLOCK_SIZE, vocab=self.cfg.VOCAB,
                        max_blocks=self.cfg.DATA.max_blocks)
        dev = next(model.parameters()).device
        n_eval = min(eval_blocks, corpus.n_blocks)
        total, count = 0.0, 0
        for i in range(corpus.n_blocks - n_eval, corpus.n_blocks):
            ids = torch.from_numpy(corpus.block(i)).long()[None].to(dev)
            logits = model(ids).logits.float()
            # 与 ar_native.ce_loss 同一 shift 语义
            tgt = ids[:, 1:]
            lo = logits[:, :-1]
            total += torch.nn.functional.cross_entropy(
                lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), reduction="sum").item()
            count += tgt.numel()
        if was_training:
            model.train()
        return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["smoketest", "tier1"])
    ap.add_argument("--impl", required=True, choices=["trans", "native"])
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "sft"])
    ap.add_argument("--base", default=None,
                    help="sft 阶段的起点权重目录（如 checkpoints/ar-3.2m-native-smoke）")
    ap.add_argument("--data", default=None,
                    help="prepared 目录覆盖（相对仓库根或绝对路径; 优先于 tier/stage 默认）")
    ap.add_argument("--steps", type=int, default=0, help="0 = config 的 train.steps")
    args = ap.parse_args()

    cfg = cfg_mod.load(args.tier)
    steps = args.steps or cfg.TRAIN.steps

    if args.data:
        data_dir = Path(args.data)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        if not (data_dir / "meta.json").exists():
            raise FileNotFoundError(f"--data 不是 prepared 目录（缺 meta.json）: {data_dir}")
    elif args.stage == "sft":
        # SFT 用 prepared SFT 目录（预切块 + assistant spans）; 不读 Corpus
        data_dir = ROOT / "data/prepared/sft-mini-6400"
    elif args.tier == "smoketest":
        # smoke 用最小语料 (SPEC 4.1): tinystories-6400 是最小就位语料
        data_dir = ROOT / "data/prepared/tinystories-6400"
        # max_blocks 来自 config 但那是 minimind 的量; smoke 覆盖为 4096 块上限不变
    else:
        data_dir = cfg.DATA_DIR

    if args.impl == "trans":
        impl = _load("src/training/ar_trans-example.py", "ar_trans_example")
    else:
        impl = _load("src/training/ar_native-example.py", "ar_native_example")

    params = analytic_params(cfg.MODEL)
    size = f"{params / 1e6:.1f}m"
    tag = "-smoke" if args.tier == "smoketest" else ""
    stage_seg = "" if args.stage == "pretrain" else "-sft"
    name = f"ar-{size}-{args.impl}{stage_seg}{tag}"

    run_dir = ROOT / "logs/run" / name
    ckpt_dir = ROOT / "checkpoints" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(cfg, name, data_dir, run_dir, ckpt_dir)

    print(f"== {name}: tier={args.tier} impl={args.impl} stage={args.stage} steps={steps} "
          f"params={params:,} data={data_dir} base={args.base}")
    t_start = time.monotonic()
    if args.stage == "sft":
        sft_mod = _load("src/training/sft-example.py", "sft_example_main")
        res = sft_mod.run_sft(cfg, steps, ctx, impl=args.impl, base_ckpt=args.base)
    else:
        res = impl.run(cfg, steps, ctx)
    wall_train = time.monotonic() - t_start

    # 收尾: eval (fp32) -> 采样 -> state -> summary (eval/采样不进 train-only 分母)
    if args.stage == "sft":
        chat_mod = _load("src/eval/chat-example.py", "chat_example_main")
        eval_loss = chat_mod.assistant_eval_loss(res["model"], data_dir)
    else:
        eval_loss = ctx.eval_loss(res["model"])
    tok = tok_mod.Tok()
    if args.stage == "sft":
        prompts = ["什么是深度学习？", "今天天气怎么样？"]
        prompt_ids = [sft_mod.encode_prompt(p) for p in prompts]
    else:
        prompts = ["Once upon a time", "小明放学回家"]
        prompt_ids = [tok.encode(p, bos=True) for p in prompts]
    sample_ids = [res["sample_fn"](res["model"], p) for p in prompt_ids]
    texts = [tok.decode(ids) for ids in sample_ids]
    rep4s = [mx_mod.rep4(ids) for ids in sample_ids]

    state_mod = _load("src/training/state-example.py", "state_example")
    state_mod.save_state(run_dir / "train_state.pt", step=res["step"],
                         optimizer=res["optimizer"], data_position=res["data_position"])

    summary = {
        "name": name, "tier": args.tier, "impl": args.impl, "stage": args.stage,
        "base": args.base, "params": params, "steps": steps,
        "data_dir": str(data_dir), "data_position": list(res["data_position"]),
        "throughput": res["meter"].report(),
        "loss": res["history"].summary(cfg.TRAIN.loss_window),
        "eval_loss": eval_loss,
        "sample": {"prompts": prompts, "texts": texts, "rep4": rep4s},
        "checkpoint": str(ckpt_dir),
        "loss_caveat": "trans 的 loss 是 Trainer 区间均值; native 是单步值" if args.impl == "trans" else "",
    }
    (run_dir / "summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== done: eval_loss={eval_loss:.4f} rep4={rep4s} "
          f"train={res['meter'].report()['train_tps']:.0f} tok/s "
          f"(wall {wall_train:.0f}s 总钟) -> {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
