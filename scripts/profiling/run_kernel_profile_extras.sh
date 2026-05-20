#!/bin/bash
# Track-A round 2: profile step 0 (warmup) and step 26 (boundary swap)
# separately, to explain the ~36 s of "non-recurring overhead" in Y4-sync's
# 613 s diffusion loop that the steady-state pie chart doesn't account for.
#
# Two runs back-to-back so the second benefits from the warm checkpoint
# cache (first cold load ~12 min; second load <2 min).
set -uo pipefail
source /workspace/activate.sh
cd /workspace/Wan2.2

CKPT="${CKPT:-/workspace/Wan2.2-T2V-A14B}"
OUT_BASE="${OUT_BASE:-/workspace/profile_outputs/kernel_profile_extras}"
PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_one() {
  local TARGET=$1
  local OUT="$OUT_BASE/$TARGET"
  local TRACE_DIR="$OUT/trace"
  rm -rf "$TRACE_DIR"
  mkdir -p "$TRACE_DIR"

  export WAN_PROFILE_KERNELS=1
  export WAN_PROFILE_KERNELS_DIR="$TRACE_DIR"
  export WAN_PROFILE_KERNELS_TARGET="$TARGET"
  export WAN_PROFILE_DIR="$OUT/span_trace"

  echo
  echo "############################################################"
  echo "### TARGET=$TARGET"
  echo "###   trace out:  $TRACE_DIR"
  echo "############################################################"
  T0=$(date +%s.%N)
  python generate.py --task t2v-A14B --ckpt_dir "$CKPT" \
    --size '1280*720' --frame_num 81 --sample_steps 40 \
    --base_seed "$SEED" --prompt "$PROMPT" \
    --convert_model_dtype --offload_model True
  RC=$?
  T1=$(date +%s.%N)
  printf "  exit=%d wall=%.1fs\n" "$RC" "$(awk "BEGIN {print $T1-$T0}")"

  echo "trace files in $TRACE_DIR:"
  ls -la "$TRACE_DIR"
}

# Step 0 first — minimal compute, captures the warmup state.
run_one warmup

# Step 26 next — needs to run through steps 0-24 before profiling boundary.
# Model is in page cache now, so model load is fast (<2 min on a 2nd run).
run_one boundary

echo
echo "=== DONE ==="
echo "trace dirs:"
ls "$OUT_BASE"/*/trace/*.json* 2>/dev/null
