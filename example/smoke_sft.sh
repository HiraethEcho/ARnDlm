#!/usr/bin/env bash
# The SFT / DPO half of SPEC 2.3, in dependency order.
#
#   bash example/smoke_sft.sh              # full run
#   bash example/smoke_sft.sh --cells-only # same, without the unit tests
#
# smoke_all.sh runs the other half (pretraining, A2D, Direct-DLM, LoRA). Between
# them the two scripts cover every cell in the matrix, so "reproduce the whole
# experimental version" is two commands.
#
# The unit tests are CPU-heavy on purpose -- one trains a 20-step model on CPU,
# another re-tokenises the real 1.7 GB SFT corpus -- so `--cells-only` is the mode
# to use on a shared box. It leaves coverage behind; run the tests separately when
# the machine is free.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
CELLS_ONLY=""
[[ "${1:-}" == "--cells-only" ]] && CELLS_ONLY="1"

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

if [[ -z "$CELLS_ONLY" ]]; then
  for t in test_sft_flatten test_sft_loss test_sft_cells test_dpo test_dlm_sft; do
    run "test/$t" "$PY" "example/tests/$t.py"
  done
fi

# --- 0. the corpus. Rebuilt every run so meta.json can never be stale.
run "prep sft-mini-6400" "$PY" example/base/hf/make_sft_data.py --max-blocks 4096

# --- 1. the AR-SFT control (G3): nothing downstream means anything without it
run "ar-3.2m-hf-sft"     "$PY" example/base/hf/train_sft.py
run "ar-3.2m-native-sft" "$PY" example/base/native/train_sft.py

# --- 2. G3 new route: pretrain -> A2D -> SFT
for c in ar-3.2m-hf-mini_a2d-sft ar-3.2m-hf-dllm_a2d-sft \
         ar-3.2m-native-mini_a2d-sft ar-3.2m-native-dllm_a2d-sft; do
  run "$c" "$PY" example/base/hf/train_dlm_sft.py --cell "$c"
done

# --- 3. G3 original route: pretrain -> SFT -> A2D, then the trailing DLM-SFT
for c in ar-3.2m-hf-sft-mini_a2d ar-3.2m-hf-sft-dllm_a2d \
         ar-3.2m-native-sft-mini_a2d ar-3.2m-native-sft-dllm_a2d; do
  run "$c" "$PY" example/base/hf/train_a2d_from_sft.py --cell "$c"
done
for c in ar-3.2m-hf-sft-mini_a2d-sft ar-3.2m-hf-sft-dllm_a2d-sft \
         ar-3.2m-native-sft-mini_a2d-sft ar-3.2m-native-sft-dllm_a2d-sft; do
  run "$c" "$PY" example/base/hf/train_dlm_sft.py --cell "$c"
done

# --- 4. G2: a DLM pretrained from scratch, then SFT'd
for c in dlm-3.2m-hf-sft dlm-3.2m-dllm-sft; do
  run "$c" "$PY" example/base/hf/train_dlm_sft.py --cell "$c"
done

# --- 5. G4: the real 600M model under LoRA, SFT'd (slowest cell)
run "qwen3-0.6b-mini_a2d-sft" \
    "$PY" example/base/qwen3_0p6b/train_lora_sft.py --n-docs 1500

# --- 6. DPO on the AR rows. The dlm row has no sequence logprob (OPEN-10), so
#        there is deliberately no `dlm-*-dpo` cell here.
for c in ar-3.2m-hf-sft-dpo ar-3.2m-native-sft-dpo; do
  run "$c" "$PY" example/base/hf/train_dpo.py --cell "$c" --max-pairs 4000
done

# --- 7. cross-checks: C7 isolates the SFT path from the upstream weights
run "check_ab c7" "$PY" example/check_ab.py c7

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