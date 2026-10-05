"""test/10-chat.py -- chat-example 端到端: 真 SFT 200 步 -> 四门槛报告。

断言「报告完整且数字真实」，不断言门槛全过 —— 未训好的模型可能确实不能 chat,
门槛失败要如实出现在报告里（false-green 反模式）。门槛 1 需要 AR-SFT 参照,
本测试用同一模型的 eval loss 充当占位参照（比值=1, pass=None 不适用时手动跳）。

跑法: .venv/bin/python test/10-chat.py
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("config_example", "common/config-example.py")
main_mod = load("main_example", "src/training/main-example.py")
sft_mod = load("sft_example", "src/training/sft-example.py")
chat = load("chat_example", "src/eval/chat-example.py")
tok_mod = load("tokenizer_example", "common/tokenizer-example.py")
native_mod = load("native_loop", "src/training/ar_native-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


cfg = cfg_mod.load("smoketest")
SFT_DIR = ROOT / "data/prepared/sft-mini-6400"
tmp = Path(tempfile.mkdtemp(prefix="chat-"))
try:
    print("[1] SFT 真跑 200 步（native, 从零起步 —— smoke 只验管线）")
    run_dir = tmp / "run"
    run_dir.mkdir()
    ctx = main_mod.Ctx(cfg, "ar-3.2m-native-sft-smoke", SFT_DIR, run_dir, run_dir / "ckpt")
    res = sft_mod.run_sft(cfg, 200, ctx, impl="native")
    losses = res["history"].losses
    check("200 步 loss 记录", len(losses) == 200)
    check("loss 在降（首窗 > 末窗）",
          sum(losses[:100]) / 100 > sum(losses[-100:]) / 100,
          (losses[0], losses[-1]))

    print("[2] 四门槛报告")
    tok = tok_mod.Tok()
    prompts = ["什么是深度学习？", "帮我写一句诗。", "今天天气怎么样？"]
    model = res["model"]

    def sample_fn(q, max_new_tokens=64):
        return native_mod.sample(model, sft_mod.encode_prompt(q),
                                 max_new_tokens=max_new_tokens)

    rep = chat.report(model, SFT_DIR, prompts, sample_fn, tok.decode,
                      ref_ar_sft_loss=None)
    names = [c["name"] for c in rep["checks"]]
    check("四个门槛都在报告里", names == ["t1_loss_ratio", "t2_rep4", "t3_format", "t4_no_echo"], names)
    t1 = rep["checks"][0]
    check("门槛 1 是 pending 不是假 pass", t1["pass"] is None, t1)
    check("assistant eval loss 有限", rep["assistant_eval_loss"] != 0 and
          all(c is not None for c in [rep["assistant_eval_loss"]]))
    print(f"  eval_loss={rep['assistant_eval_loss']:.4f} rep4={rep['rep4']}")
    print(f"  t2_pass={rep['checks'][1]['pass']} t3_pass={rep['checks'][2]['pass']} "
          f"t4_pass={rep['checks'][3]['pass']}")
    for t in rep["texts"]:
        print(f"  sample: {t[:60]!r}")

    print("[3] 有参照时门槛 1 会真的算比值")
    rep2 = chat.check_thresholds(prompts, rep["texts"], rep["rep4"],
                                 rep["assistant_eval_loss"],
                                 ref_ar_sft_loss=rep["assistant_eval_loss"] / 1.2)
    t1b = rep2["checks"][0]
    check("比值算出且判 pass", t1b["ratio"] == 1.2 and t1b["pass"] is True, t1b)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("PASS test/10-chat")
