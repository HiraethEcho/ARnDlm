#!/usr/bin/env bash
# 装 dllm 包 (~/Public/dllm), editable + --no-deps。
#
# 为什么 --no-deps:
#   dllm/pyproject.toml pin 了 deepspeed==0.18.0 / accelerate==1.11.0 / wandb,
#   本项目是单卡小模型 —— ZeRO 切片的收益是 0 (没有多卡可切),
#   代价是 deepspeed 在 ROCm + Python 3.14 上大概率编译失败。
#   所以只装 dllm 本体, 缺什么补什么。
#
# 为什么 editable (-e):
#   dllm/pipelines/a2d/models/{llama,qwen2,qwen3}/ 是**拷贝 transformers 源码
#   改出来的**。读它的 modeling 文件是理解 A2D 到底改了三处的最快路径,
#   装了之后可以直接在 .venv 里 import 进来对着看。
#
# 幂等: 已装则重装, 不会报错。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DLLM="${DLLM_PATH:-$HOME/Public/dllm}"

if [[ ! -d "$DLLM" ]]; then
    echo "ERROR 找不到 dllm 仓库: $DLLM"
    echo "       用 DLLM_PATH=/path/to/dllm 覆盖"
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR 没有 .venv, 先跑 ./scripts/setup_env.sh"
    exit 1
fi

echo "==> 装 dllm (editable, --no-deps) from $DLLM"
uv pip install --python .venv/bin/python --no-deps -e "$DLLM"

echo
echo "==> 验证"
.venv/bin/python - <<'EOF'
import importlib
for mod in ["dllm", "dllm.core", "dllm.pipelines.a2d"]:
    try:
        importlib.import_module(mod)
        print("OK  ", mod)
    except Exception as e:
        print("FAIL", mod, "->", type(e).__name__, str(e)[:100])
EOF

cat <<'EOF'

如果 a2d / trainers 报缺包, 按需单独补 (不要一次装全):
  uv pip install --python .venv/bin/python omegaconf tyro rich torchmetrics

dllm 的 A2D 只实现了 llama / qwen2 / qwen3 三个架构:
  ~/Public/dllm/dllm/pipelines/a2d/models/
EOF
