#!/bin/bash
# Track-A per-kernel profiling: capture 2 steady-state diffusion steps with
# torch.profiler at full kernel-level detail, on 1× H100.
#
# Goal: validate the analytical decomposition of the DiT forward
# (attention ~65 %, matmul ~12 %, LN/residual ~8 %, overhead ~13 %).
# Produces a Chrome / Perfetto trace that scripts/profiling/analyze_kernel_trace.py
# can parse into a per-category pie chart.
#
# Workload is identical to Z1 (1-GPU baseline) — same DiT forward kernels
# as Y4-sync, just running cond and uncond serially on one GPU instead of
# split across two ranks. Per-kernel mix is the same.
set -uo pipefail
source /workspace/activate.sh
cd /workspace/Wan2.2

CKPT="${CKPT:-/workspace/Wan2.2-T2V-A14B}"
OUT="${OUT:-/workspace/profile_outputs/kernel_profile_y4_sync}"
PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
SEED="${SEED:-42}"

mkdir -p "$OUT"
TRACE_DIR="$OUT/trace"
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR"

# Force single-GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Enable the torch.profiler hook in wan/text2video.py
export WAN_PROFILE_KERNELS=1
export WAN_PROFILE_KERNELS_DIR="$TRACE_DIR"

# Useful but optional: keep the existing per-span CSV / Chrome trace too,
# for cross-referencing kernel-level timings against the named-span timings.
export WAN_PROFILE_DIR="$OUT/span_trace"

echo "=== Y4-sync-equivalent 1-GPU kernel-profile run ==="
echo "  ckpt:      $CKPT"
echo "  trace out: $TRACE_DIR"
echo "  span out:  $WAN_PROFILE_DIR"
echo "  expected wall: ~5-10 min if model is already in page cache, ~20-25 min cold."
echo

T0=$(date +%s.%N)
python generate.py --task t2v-A14B --ckpt_dir "$CKPT" \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed "$SEED" --prompt "$PROMPT" \
  --convert_model_dtype --offload_model True
RC=$?
T1=$(date +%s.%N)
printf "  exit=%d wall=%.1fs\n" "$RC" "$(awk "BEGIN {print $T1-$T0}")"

echo
echo "=== trace files in $TRACE_DIR ==="
ls -la "$TRACE_DIR"
echo
TRACE_FILE=$(ls -t "$TRACE_DIR"/*.json* 2>/dev/null | head -1)
if [ -n "$TRACE_FILE" ]; then
  echo "Primary trace: $TRACE_FILE"
  echo "Size: $(du -h "$TRACE_FILE" | awk '{print $1}')"
  echo
  echo "To analyze on the Mac, scp it down and run:"
  echo "  python scripts/profiling/analyze_kernel_trace.py $TRACE_FILE"
fi
