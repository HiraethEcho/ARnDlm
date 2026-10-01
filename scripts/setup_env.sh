#!/usr/bin/env bash
# 创建 .venv 并装好依赖。
#
# torch 版本对应 AMD 官方文档 (ROCm 10.0.0 / PyTorch 2.12.0 / gfx1150):
#   https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
#
# 官方给的是 venv + pip, 这里换成 uv —— 命令一一对应, 只是快很多。
#
#   官方                              uv (本脚本)
#   ──────────────────────────────    ──────────────────────────────────
#   python3.14 -m venv .venv           uv venv .venv --python 3.14
#   pip install --index-url ...        uv pip install --index-url ...
#       torch[device-gfx1150]==...          torch[device-gfx1150]==...
#   pip install transformers ...       uv pip install transformers ...
#
# 幂等: 已有 .venv 就跳过创建, 直接装包。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROCM_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
PYTHON_VERSION="3.14"

# ---------------------------------------------------------------------------
# --index-strategy unsafe-best-match 是**必需的**, 不是可选优化。
#
# ROCm 索引只提供 torch, PyPI 只提供 HF 栈。uv 默认只从「第一个含有该包的
# 索引」取版本, 不跨索引 —— 所以不加这个 flag, uv 会在 PyPI 上找不到
# torch==2.12.0+rocm10.0.0 然后直接报 unsatisfiable。
# ---------------------------------------------------------------------------
INDEX_FLAGS=(--index-url "$ROCM_INDEX" --extra-index-url https://pypi.org/simple
             --index-strategy unsafe-best-match)

if [[ ! -d .venv ]]; then
    echo "==> 创建 .venv (Python $PYTHON_VERSION)"
    uv venv .venv --python "$PYTHON_VERSION"
else
    echo "==> .venv 已存在, 跳过创建"
fi

echo "==> 装 ROCm torch (几个 GB, 慢, 约 10-20 分钟)"
uv pip install --python .venv/bin/python "${INDEX_FLAGS[@]}" \
    "torch[device-gfx1150]==2.12.0+rocm10.0.0"

# transformers 钉 4.57.0 是 dllm 的 pin。
# 它在 PyPI 上是 yank 状态 (原因: "Error in the setup causing installation
# issues"), wheel 本身可用 —— 已实测能装能跑。不要因为看到 yank 就升版本。
# 5.x 改了 create_causal_mask 签名, dllm/pipelines/a2d/models/qwen3/ 接不上。
echo "==> 装 HF 栈"
# accelerate 看着像是「分布式训练用不上」的东西, 但 **peft 硬依赖它**
# (peft 0.17.1: accelerate>=0.21.0), 装 peft 就会把它拉进来。
# 它本身是纯 Python, 不带 deepspeed, 装了没有代价 —— 所以直接显式装上,
# 别等 peft 隐式解析时才发现。
uv pip install --python .venv/bin/python \
    "transformers==4.57.0" \
    "peft==0.17.1" \
    "accelerate" \
    "datasets==4.2.0" \
    tokenizers \
    safetensors \
    numpy \
    pyyaml \
    tqdm \
    matplotlib

# deepspeed / wandb / torchvision / torchaudio 全部**不装**。
# 理由见 requirements.txt。

echo
echo "==> 验证"
.venv/bin/python - <<'EOF'
import sys, torch, transformers, tokenizers
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__)
print("hip          :", torch.version.hip)
print("gpu          :", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers :", transformers.__version__)
print("tokenizers   :", tokenizers.__version__)
try:
    import peft, datasets
    print("peft         :", peft.__version__)
    print("datasets     :", datasets.__version__)
    import accelerate
    print("accelerate   :", accelerate.__version__)
except Exception as e:
    print("peft/datasets:", type(e).__name__, str(e)[:60])

# dllm 是可选的, 没装不算错
try:
    import dllm
    print("dllm         : 已安装 (./scripts/install_dllm.sh)")
except ImportError:
    print("dllm         : 未安装 (./scripts/install_dllm.sh)")
EOF

cat <<'EOF'

下一步:
  ./scripts/install_dllm.sh     # dllm 包 (--no-deps)
  ./scripts/setup_symlinks.sh   # data/ checkpoints/ -> /mnt/extra
  .venv/bin/python test/01-tokenizer-mask.py
EOF
