#!/bin/bash
# xDiT sweep — runs Wan2.2 T2V-A14B at 720p via xfuser with 3 parallelism configs:
#   X1: ulysses=2, ring=1                    (seq-parallel via Ulysses)
#   X2: ulysses=1, ring=2                    (Ring attention)
#   X3: ulysses=1, ring=1, use_cfg_parallel  (CFG-parallel, comparable to our Y4)
#
# Each config × 3 runs, skip_first → 2 measured. Total ~3 hr wall.
#
# Output: /workspace/profile_outputs/phase2_xdit/<config>_run<i>/timing_rank0.csv

set -uo pipefail

WAN_DIR=/workspace/Wan2.2
ACTIVATE=/workspace/activate.sh
MODEL=/workspace/Wan2.2-T2V-A14B-Diffusers
OUT_ROOT=/workspace/profile_outputs/phase2_xdit
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
HEIGHT=720
WIDTH=1280
FRAMES=81
STEPS=40
SEED=42
GUIDE=4.0       # high_noise (transformer)
GUIDE_2=3.0     # low_noise  (transformer_2)
RUNS=3
SKIP_FIRST=1

source "$ACTIVATE"
cd "$WAN_DIR"

mkdir -p "$OUT_ROOT"

run_one() {
  local NAME=$1
  shift
  local PARALLEL_ARGS="$@"
  local PORT=29500

  echo
  echo "##########################################"
  echo "##### Sweep config: $NAME"
  echo "##########################################"

  for i in $(seq 0 $((RUNS - 1))); do
    local RUN_DIR="$OUT_ROOT/${NAME}_run${i}"
    rm -rf "$RUN_DIR"
    echo
    echo "=== Run $((i+1))/$RUNS: ${NAME}_run${i} ==="
    local T0=$(date +%s.%N)
    torchrun --nproc_per_node=2 --master_port=$PORT \
      scripts/profiling/run_xdit.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --negative_prompt "$NEG_PROMPT" \
      --height $HEIGHT --width $WIDTH \
      --num_frames $FRAMES --num_inference_steps $STEPS \
      --seed $SEED \
      --guidance_scale $GUIDE \
      --guidance_scale_2 $GUIDE_2 \
      --profile_dir "$RUN_DIR" \
      $PARALLEL_ARGS
    local T1=$(date +%s.%N)
    local WALL=$(awk "BEGIN {print $T1 - $T0}")
    printf "  exit=%d wall=%.1fs\n" $? $WALL
  done
}

# X1: Ulysses-2 (sequence-parallel only)
run_one "X1_ulysses2" --ulysses_degree 2 --ring_degree 1

# X2: Ring-2 (Ring attention only)
run_one "X2_ring2" --ulysses_degree 1 --ring_degree 2

# X3: CFG-parallel-2 (cond/uncond split across rank groups)
run_one "X3_cfg2" --ulysses_degree 1 --ring_degree 1 --use_cfg_parallel

echo
echo "##########################################"
echo "##### xDiT sweep DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "##########################################"
