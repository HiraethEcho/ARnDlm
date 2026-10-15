"""test/01-config.py -- models/config.py 的验收（PLAN P1: yaml 浮点为 float、未知键报错）。

跑法: .venv/bin/python test/01-config.py   （从仓库根目录）
"""
import sys
import tempfile
from pathlib import Path

from _impl import load, MODE


cfg_mod = load("config_example", "common/config-example.py")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


print("[1] 两档 yaml 加载成功，数值类型正确")
c1 = cfg_mod.load("smoketest")
c2 = cfg_mod.load("tier1")
for c in (c1, c2):
    t = c.TRAIN
    check(f"{c.tier}: lr 是 float", isinstance(t.lr, float) and t.lr == 3.0e-4, repr(t.lr))
    check(f"{c.tier}: min_lr_ratio 是 float", isinstance(t.min_lr_ratio, float))
    check(f"{c.tier}: rms_norm_eps 是 float", isinstance(c.MODEL.rms_norm_eps, float))
    check(f"{c.tier}: steps 是 int", isinstance(t.steps, int) and not isinstance(t.steps, bool))
    check(f"{c.tier}: tie_word_embeddings 是 bool", isinstance(c.MODEL.tie_word_embeddings, bool))

print("[2] 常量与派生（单一来源）")
check("smoketest tokens", (c1.PAD, c1.BOS, c1.EOS, c1.MASK) == (0, 1, 2, 27), repr((c1.PAD, c1.BOS, c1.EOS, c1.MASK)))
check("VOCAB == vocab_size", c1.VOCAB == 6400 == c1.MODEL.vocab_size)
check("BLOCK_SIZE == context_length", c1.BLOCK_SIZE == 128 == c1.MODEL.context_length)
check("DTYPE 透传", c1.DTYPE == "bfloat16")
check("DATA_DIR 指向 prepared", str(c1.DATA_DIR).endswith("data/prepared/minimind3-6400"), str(c1.DATA_DIR))
check("GQA 整除", c1.MODEL.num_heads % c1.MODEL.num_kv_heads == 0)

print("[3] frozen dataclass")
try:
    c1.BLOCK_SIZE = 999
    check("赋值被拒", False, "frozen 失效")
except Exception:
    check("赋值被拒", True)

print("[4] 未知键一次列全")
bad_yaml = """
tier: bad
model:
  hidden_size: 256
  num_layers: 2
  num_heads: 2
  num_kv_heads: 1
  head_dim: 128
  intermediate_size: 768
  context_length: 128
  vocab_size: 6400
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-6
  tie_word_embeddings: true
  spint: 1          # 未知键
temperary: 2        # 未知顶层键
tokens: {pad: 0, bos: 1, eos: 2, mask: 27}
train:
  steps: 4
  batch_size: 8
  lr: 3.0e-4
  dlm_lr: 1.0e-4
  min_lr_ratio: 0.1
  warmup_steps: 2
  weight_decay: 0.1
  grad_clip: 1.0
  beta1: 0.9
  beta2: 0.95
  seed: 42
  log_every: 1
  loss_window: 2
data: {prepared_dir: x, max_blocks: 4}
runtime: {dtype: bfloat16}
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=cfg_mod.CONFIG_DIR) as f:
    f.write(bad_yaml)
    tmp = Path(f.name)
try:
    try:
        cfg_mod.load(tmp.stem)
        check("未知键报错", False, "竟然加载成功")
    except ValueError as e:
        msg = str(e)
        check("未知键报错", "spint" in msg and "temperary" in msg, msg)
        check("一次列全(不用改一次跑一次)", msg.count("未知") >= 2, msg)
finally:
    tmp.unlink()

print("[5] 浮点写成 3e-4 -> 显式报错而不是静默 str")
bad_yaml2 = bad_yaml.replace("3.0e-4", "3e-4").replace("  spint: 1", "").replace("temperary: 2", "")
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=cfg_mod.CONFIG_DIR) as f:
    f.write(bad_yaml2)
    tmp2 = Path(f.name)
try:
    try:
        cfg_mod.load(tmp2.stem)
        check("3e-4 报错", False, "静默通过了")
    except ValueError as e:
        check("3e-4 报错", "3e-4" in str(e), str(e))
finally:
    tmp2.unlink()

print("[6] 缺键报错")
bad_yaml3 = bad_yaml.replace("  loss_window: 2", "").replace("  spint: 1", "").replace("temperary: 2", "")
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=cfg_mod.CONFIG_DIR) as f:
    f.write(bad_yaml3)
    tmp3 = Path(f.name)
try:
    try:
        cfg_mod.load(tmp3.stem)
        check("缺键报错", False)
    except ValueError as e:
        check("缺键报错", "loss_window" in str(e), str(e))
finally:
    tmp3.unlink()

print("[7] lr 调度: warmup 线性 -> cosine, 两路同一公式（HF 语义: 峰在 warmup 整点）")
c = c1
check("step0 = 0", cfg_mod.lr_at(c, 0) == 0.0)
check("warmup-1 未达峰", abs(cfg_mod.lr_at(c, c.TRAIN.warmup_steps - 1)
                           - c.TRAIN.lr * (c.TRAIN.warmup_steps - 1) / c.TRAIN.warmup_steps) < 1e-12)
check("warmup 整点达峰", abs(cfg_mod.lr_at(c, c.TRAIN.warmup_steps) - c.TRAIN.lr) < 1e-12)
mid = cfg_mod.lr_at(c, (c.TRAIN.warmup_steps + c.TRAIN.steps) // 2)
floor = c.TRAIN.lr * c.TRAIN.min_lr_ratio
check("cosine 中段在 floor 与 lr 之间", floor < mid < c.TRAIN.lr, f"{mid}")
end = cfg_mod.lr_at(c, c.TRAIN.steps)
check("结尾贴 floor", abs(end - floor) < 1e-9, f"{end} vs {floor}")
check("超出 steps 贴 floor", abs(cfg_mod.lr_at(c, c.TRAIN.steps + 100) - floor) < 1e-12)

print()
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print(f"PASS test/01-config [IMPL={MODE}]")
