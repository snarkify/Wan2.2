#!/bin/bash
# Overnight chain runner for the Phase 3 lossless extensions.
# Designed to be launched via nohup; logs everything to /workspace/profile_outputs/overnight.log.
#
# Sequence:
#   1. Background: try to build FA3 (flash-attention hopper variant). May fail; fine.
#   2. Y5 × 5 variance hardening (~85 min).
#   3. compile mode tuning: reduce-overhead, max-autotune (~104 min).
#   4. If FA3 build succeeded, Y4/Y5 with FA3 (~104 min).
#   5. PSNR/SSIM check on each compile-mode output vs Y1 reference.
#   6. analyze_sweep.py over each new sweep dir.
#   7. Final summary line written to overnight_done.flag (used by Monitor).

set -uo pipefail
exec > /workspace/profile_outputs/overnight.log 2>&1

WAN_DIR=/workspace/Wan2.2
VENV_ACTIVATE=/workspace/activate.sh
DONE_FLAG=/workspace/profile_outputs/overnight_done.flag

source $VENV_ACTIVATE
cd $WAN_DIR

echo "=========================================="
echo "OVERNIGHT CHAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

# Step 1: Background FA3 build. Don't gate the chain on this.
(
  echo "----- FA3 build start $(date -u +%H:%M:%S) -----"
  cd /tmp
  rm -rf flash-attention
  if git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>&1; then
    cd flash-attention/hopper
    # Use limited parallelism to avoid OOM on cicc
    MAX_JOBS=4 /workspace/venv/bin/pip install -v --no-build-isolation . 2>&1 | tail -10
    /workspace/venv/bin/python -c "import flash_attn_interface; print('FA3 install OK', flash_attn_interface.__name__)" 2>&1
  fi
  echo "----- FA3 build end $(date -u +%H:%M:%S) -----"
) &
FA3_PID=$!

# Step 2: Y5 variance hardening
echo
echo "=========================================="
echo "STEP 2: Y5 variance × 5 — start $(date -u +%H:%M:%S)"
echo "=========================================="
python3 scripts/profiling/sweep.py scripts/profiling/configs/phase3_y5_variance.yaml

# Step 3: compile mode tuning
echo
echo "=========================================="
echo "STEP 3: compile mode tuning — start $(date -u +%H:%M:%S)"
echo "=========================================="
python3 scripts/profiling/sweep.py scripts/profiling/configs/phase3_compile_modes.yaml

# Step 4: FA3 — wait for build, then run if succeeded
echo
echo "=========================================="
echo "STEP 4: waiting for FA3 build to finish — $(date -u +%H:%M:%S)"
echo "=========================================="
wait $FA3_PID || true
if /workspace/venv/bin/python -c "import flash_attn_interface" 2>/dev/null; then
  echo "FA3 available — running phase3_fa3.yaml"
  python3 scripts/profiling/sweep.py scripts/profiling/configs/phase3_fa3.yaml
else
  echo "FA3 not available — skipping phase3_fa3.yaml"
fi

# Step 5: PSNR/SSIM on the new compile-mode outputs vs Y1 reference
echo
echo "=========================================="
echo "STEP 5: PSNR/SSIM compile modes vs Y1 — $(date -u +%H:%M:%S)"
echo "=========================================="
Y1_REF="$WAN_DIR/t2v-A14B_1280*720_2_Two_anthropomorphic_cats_in_comfy_boxing_gear_and__20260505_100026.mp4"
for label in Y6 Y7; do
  # Find latest mp4 from the relevant phase3_compile_modes run dirs.
  # Configs name them Y6_*/Y7_* via the sweep harness, but mp4s land in $WAN_DIR
  # alongside everything else. Use the run start time recorded in the sweep log
  # as a hint via the latest mp4 newer than our reference.
  echo "--- $label vs Y1 ---"
  # The mp4 with the latest mtime in $WAN_DIR after the config name appears in the log.
  # Fallback heuristic: the 2 most recent mp4s correspond to Y6 and Y7 in order.
done

LATEST_MP4S=$(ls -1t "$WAN_DIR"/*.mp4 2>/dev/null | head -2)
echo "Most recent 2 mp4s (presumed Y7 then Y6):"
echo "$LATEST_MP4S"
i=0
for mp4 in $LATEST_MP4S; do
  name=$([ $i -eq 0 ] && echo "Y7_max_autotune" || echo "Y6_reduce_overhead")
  echo
  echo "=== $name vs Y1 ==="
  md5sum "$mp4" "$Y1_REF" 2>&1
  ffmpeg -hide_banner -i "$Y1_REF" -i "$mp4" -filter_complex '[0:v][1:v]psnr' -f null - 2>&1 | grep -E 'PSNR' | head -1
  ffmpeg -hide_banner -i "$Y1_REF" -i "$mp4" -filter_complex '[0:v][1:v]ssim' -f null - 2>&1 | grep -E 'SSIM' | head -1
  i=$((i+1))
done

# Step 6: Per-sweep analysis
echo
echo "=========================================="
echo "STEP 6: analyze_sweep on each new dir — $(date -u +%H:%M:%S)"
echo "=========================================="
for d in /workspace/profile_outputs/phase3_y5_variance \
         /workspace/profile_outputs/phase3_compile_modes \
         /workspace/profile_outputs/phase3_fa3; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null | grep _run)" ]; then
    echo
    echo "=== $d ==="
    python3 scripts/profiling/analyze_sweep.py "$d" 2>&1 | head -80 || true
  fi
done

# Step 7: Done flag
echo
echo "=========================================="
echo "OVERNIGHT CHAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
date -u +%Y-%m-%dT%H:%M:%SZ > $DONE_FLAG
