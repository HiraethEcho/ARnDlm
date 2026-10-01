#!/usr/bin/env bash
# 装 dllm 包: 从**上游仓库** clone 到固定 commit, 打本项目的 patch, 再 editable 装。
#
# 为什么不直接用本机的 ~/Public/dllm:
#   那样这个仓库就只能在你自己这台机器上复现。clone + pin + patch 的流程
#   换台机器跑一遍就行, 结果完全一致。
#
# 为什么需要 patch (patches/dllm-0001-a2d-config-class.patch):
#   dllm 的 pyproject pin 了 transformers==4.57.0, 但它的三个 A2D 模型类
#   (llama / qwen2 / qwen3) 都没定义 `config_class`。而 4.57 的
#   AutoModel.register (auto_factory.py:624) 新增了一致性校验:
#       model_class.config_class.__name__ != config_class.__name__  ->  ValueError
#   继承来的 config_class 是 LlamaConfig / Qwen2Config / Qwen3Config,
#   和 A2D*Config 对不上, 于是 `import dllm.pipelines.a2d` 直接炸。
#   patch 就是给三个类各补一行 `config_class = A2D*Config`。
#   已实测: 打上后 import 通过。
#
# 幂等: 已 clone 且 commit 匹配就复用, 否则重新 clone。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UPSTREAM="https://github.com/ZHZisZZ/dllm.git"
# 钉死 commit —— 用 main 分支名的话上游一提交, 你的环境就悄悄变了
DLLM_COMMIT="ca176752fbceec49c6b4777a2c18ae88e4eb10ed"
CACHE="${DLLM_CACHE:-$HOME/.cache/ARnDlm}"
SRC="$CACHE/dllm"

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR 没有 .venv, 先跑 ./scripts/setup_env.sh"
    exit 1
fi

# --- 1. clone / 更新到指定 commit ---
if [[ ! -d "$SRC/.git" ]]; then
    echo "==> clone dllm -> $SRC"
    mkdir -p "$CACHE"
    git clone -q "$UPSTREAM" "$SRC"
fi

# commit 已经在本地就**不要 fetch**。
# 网络时好时坏 (GitHub 经常 SSL_read unexpected eof), 而 set -e 下一个
# 无谓的 fetch 失败会让整个安装挂掉 —— 即使本地已经有那个 commit 了。
if git -C "$SRC" cat-file -e "$DLLM_COMMIT^{commit}" 2>/dev/null; then
    echo "==> commit 已在本地, 跳过 fetch"
else
    echo "==> fetch 指定 commit"
    git -C "$SRC" fetch -q origin
fi

git -C "$SRC" checkout -q "$DLLM_COMMIT"
echo "    commit: $(git -C "$SRC" rev-parse --short HEAD)  $(git -C "$SRC" log -1 --format=%s | cut -c1-50)"

# --- 2. 打 patch ---
# 用 git apply --check 先试: 已经打过就跳过, 没打过就打上。
# 用 --3way 让 patch 在上游有小改动时也能尽量套上。
if git -C "$SRC" apply --check -R "$ROOT/patches/dllm-0001-a2d-config-class.patch" 2>/dev/null; then
    echo "==> patch 已经打过, 跳过"
else
    echo "==> 打 patch (A2D config_class)"
    git -C "$SRC" apply --3way "$ROOT/patches/dllm-0001-a2d-config-class.patch"
    # 注意: --3way 是把改动放进**暂存区**, 所以在那个 clone 里 `git diff`
    # 看着是空的 —— 要 `git diff --cached` 才看得到。幂等检测走的是
    # `git apply --check -R`, 不受影响, 但别拿 `git diff` 判断 patch 打没打上。
fi

# --- 3. 装 ---
echo "==> editable 安装 (--no-deps)"
uv pip install --python .venv/bin/python --no-deps -e "$SRC"

# --- 4. 补 dllm 急切 import 的运行时依赖 ---
# dllm/__init__.py 顶层 `from . import core, data, pipelines, utils`, 而
# pipelines/__init__.py 又 `from . import a2d, bert, dream, editflow,
# fastdllm, llada, llada2, llada21, rl` —— 一个包把整棵树拉进来。
# 所以下面这些包**本项目一个都不用**, 但不装 dllm 就 import 不了:
#
#   lm_eval  被 dllm/core/eval/base.py 拉入 → 拖 30 个包 (scipy/sklearn/
#           pandas/nltk/lxml ...), +400MB。本项目永远不跑它的评测。
#   trl     被 dllm/pipelines/rl/grpo/trainer.py 拉入 (GRPO 强化学习)
#   其余     tyro/omegaconf/rich/torchmetrics 是 a2d + core 的直接依赖
#
# 已经确认: 装上面这些包会把 torch / transformers / numpy 锁在原版本,
# 但 **datasets 会被顶上去** (trl 的依赖解析把它带到 5.x)。dllm 自己 pin 的
# 是 datasets==4.2.0, 所以装完必须压回去 —— 这一步不能省。
echo "==> 装 dllm 的急切 import 依赖"
uv pip install --python .venv/bin/python \
    lm_eval tyro omegaconf rich torchmetrics trl

echo "==> 回钉被依赖解析顶上去的包"
uv pip install --python .venv/bin/python "datasets==4.2.0"

# --- 5. 验证 ---
echo
echo "==> 验证"
.venv/bin/python - <<'EOF'
import sys
ok = True

try:
    import dllm
    print("OK   import dllm")
except Exception as e:
    print("FAIL import dllm ->", type(e).__name__, str(e)[:120]); ok = False

try:
    from dllm.core.schedulers import LinearAlphaScheduler
    import torch
    w = LinearAlphaScheduler().weight(torch.tensor(0.5)).item()
    print(f"OK   scheduler  weight(t=0.5) = {w:.4f}  (期望 {1/0.500001:.4f})")
except Exception as e:
    print("FAIL scheduler ->", type(e).__name__, str(e)[:120]); ok = False

# 这条最关键 —— 它会拉起三个被 patch 过的 modeling 文件
try:
    import dllm.pipelines.a2d as a2d
    names = sorted(n for n in dir(a2d) if n.startswith("A2D"))
    print(f"OK   import dllm.pipelines.a2d   {names}")
except Exception as e:
    print("FAIL a2d ->", type(e).__name__, str(e)[:120]); ok = False

# 版本不能被依赖解析悄悄改掉。四个都要查 —— 只查 torch/transformers/numpy
# 会漏掉 datasets, 而 trl 恰好会顶它。
import torch, transformers, numpy, datasets
for mod, want in [(torch, "2.12.0+rocm10.0.0"),
                  (transformers, "4.57.0"),
                  (datasets, "4.2.0")]:
    got = mod.__version__
    if got != want:
        print(f"FAIL {mod.__name__} = {got}, 期望 {want}"); ok = False
print(f"OK   版本未被动: torch={torch.__version__} transformers={transformers.__version__} "
      f"datasets={datasets.__version__} numpy={numpy.__version__}")

sys.exit(0 if ok else 1)
EOF

cat <<'EOF'

下一步:
  .venv/bin/python test/04-dllm-loss-check.py     # Part B 现在应该真跑, 不再 SKIP
  .venv/bin/python test/03-lora-inject.py

dllm 的 A2D 只实现了三个架构:
  ~/.cache/ARnDlm/dllm/dllm/pipelines/a2d/models/{llama,qwen2,qwen3}/
EOF
