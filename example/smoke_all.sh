#!/usr/bin/env bash
# Run all ten cells of SPEC 2.3 in dependency order, then every cross-check.
#
#   .venv/bin/example/smoke_all.sh          # full run (~25 min)
#   .venv/bin/example/smoke_all.sh --quick  # 20 steps, small corpus (~8 min)
#
# Cells are ordered so each one's input already exists: the two AR baselines
# come first, then the three methods on row 1, then row 2, then row 3.
#
# `--quick` shrinks steps and corpus but keeps every code path, so it is the
# honest smoke test; the full run is what RESULTS.md quotes.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
QUICK=""
STEPS_FLAG=""
CELLS_ONLY=""
for arg in "$@"; do
  case "$arg" in
    --quick)      QUICK="--quick"; STEPS_FLAG="--steps 20 --max-blocks 1024" ;;
    --cells-only) CELLS_ONLY="1" ;;
  esac
done

FAILED=()

run() {                      # run <label> <command...>
  local label="$1"; shift
  echo
  echo "=============================================================="
  echo "  $label"
  echo "=============================================================="
  if "$@"; then
    echo "  -> OK"
  else
    echo "  -> FAILED: $label"
    FAILED+=("$label")
  fi
}

echo "A-version smoke: 10 cells, Tier 0, ${QUICK:-full}"

# --- unit tests first: a broken base makes every later number meaningless ---
if [[ -n "$CELLS_ONLY" ]]; then
  echo "  (--cells-only: unit tests skipped, cells + cross-checks only)"
else
for t in test_common test_hf_ar test_native_c1 test_a2d test_dllm_a2d \
         test_direct_dlm test_native_dlm test_lora_0p6b \
         test_sft_flatten test_sft_loss test_sft_cells; do
  run "test/$t" "$PY" "example/tests/$t.py"
done
fi

# --- row 1: AR baseline, then the three DLM methods ---
run "cell 1  ar-3.2m-hf" \
    "$PY" example/base/hf/train_ar.py $STEPS_FLAG
run "cell 2  ar-3.2m-native" \
    "$PY" example/base/native/train_ar.py $STEPS_FLAG --compare-hf
run "cell 3  ar-3.2m-hf-mini_a2d" \
    "$PY" example/base/hf/train_a2d_minimind.py $STEPS_FLAG
run "cell 4  ar-3.2m-hf-dllm_a2d" \
    "$PY" example/base/hf/train_a2d_dllm.py $STEPS_FLAG
run "cell 5  dlm-3.2m-hf" \
    "$PY" example/base/hf/train_direct_dlm.py $STEPS_FLAG

# --- row 2: the same three on the native torch base ---
run "cell 6  ar-3.2m-native-dllm_a2d" \
    "$PY" example/base/native/train_a2d_dllm.py $STEPS_FLAG
run "cell 7  ar-3.2m-native-mini_a2d" \
    "$PY" example/base/native/train_a2d_minimind.py $STEPS_FLAG
run "cell 8  dlm-3.2m-dllm" \
    "$PY" example/base/hf/train_direct_dlm_dllm.py $STEPS_FLAG

# --- row 3: real 0.6B with LoRA (slow; --quick still runs 20 steps) ---
run "cell 9  qwen3-0.6b-dllm_a2d" \
    "$PY" example/base/qwen3_0p6b/train_lora_dllm.py --n-docs 400
run "cell 10 qwen3-0.6b-mini_a2d" \
    "$PY" example/base/qwen3_0p6b/train_lora_minimind.py --n-docs 400

# --- cross-checks ---
run "check_ab c1..c6" "$PY" example/check_ab.py all
run "decode benchmark" "$PY" example/check_ab.py decode-bench

echo
echo "=============================================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "  ALL GREEN"
else
  echo "  ${#FAILED[@]} FAILED:"
  printf '   - %s\n' "${FAILED[@]}"
fi
echo "=============================================================="
[[ ${#FAILED[@]} -eq 0 ]]