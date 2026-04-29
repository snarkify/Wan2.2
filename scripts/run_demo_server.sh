#!/bin/bash
# Launch the Wan2.2 TI2V-5B demo API server via torchrun.
#
# Required env vars:
#   WAN_DEMO_TOKEN      shared bearer token clients must present
#   WAN_DEMO_CKPT_DIR   path to the Wan2.2-TI2V-5B checkpoint directory
#   WAN_DEMO_OUTPUT_DIR directory where jobs.db and per-job videos land
#
# Optional:
#   WAN_DEMO_PORT                    default 8000
#   WAN_DEMO_HOST                    default 0.0.0.0
#   WAN_DEMO_MAX_QUEUE               default 3
#   WAN_DEMO_ALLOW_PRIVATE_CALLBACK  default 0 (set to 1 for localhost webhooks)
#   WAN_DEMO_PUBLIC_URL              default http://$WAN_DEMO_HOST:$WAN_DEMO_PORT
#                                    (used in callback payload's video_url)
#   WAN_DEMO_GPUS                    default 1 (nproc_per_node) — see note below
#   WAN_DEMO_MASTER_PORT             default 29600
#   WAN_DEMO_QUANT                   default fp8 — set to "bf16" for the slower
#                                    full-precision path. fp8 uses
#                                    torch.float8_e4m3fn + _scaled_mm and gives
#                                    ~2.25x wall-clock speedup vs the previous
#                                    4-GPU bf16 path (288 s/job vs 649 s/job
#                                    for an 81-frame 1280*704 job on RTX 4090).
#
# Why GPUS=1 by default:
#   The 4-GPU FSDP+Ulysses path runs at 649 s/job and is bottlenecked by
#   FSDP wrap/unflatten + cross-rank communication. The 1-GPU warm path
#   with fp8 lands at 288 s/job (warm; first job pays an extra 86 s for
#   the pipeline constructor). See docs/demo-server.md for the bench
#   data and bench_artifacts/stats_warm_fp8_v2.json on gpu6 for raw
#   numbers. Override with WAN_DEMO_GPUS=4 to fall back to the FSDP
#   path if needed.
set -euo pipefail

: "${WAN_DEMO_TOKEN:?WAN_DEMO_TOKEN is required}"
: "${WAN_DEMO_CKPT_DIR:?WAN_DEMO_CKPT_DIR is required}"
: "${WAN_DEMO_OUTPUT_DIR:?WAN_DEMO_OUTPUT_DIR is required}"

GPUS="${WAN_DEMO_GPUS:-1}"
MASTER_PORT="${WAN_DEMO_MASTER_PORT:-29600}"

# fp8 quantization for transformer-block Linears; bf16 to disable.
export WAN_DEMO_QUANT="${WAN_DEMO_QUANT:-fp8}"

# The expandable_segments allocator keeps us safe around the 24 GB VAE
# decode workspace on 4090s (see docs/profiling-results.md).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Disable NCCL async error handling — the demo sits idle in
# broadcast_object_list between jobs and the watchdog would otherwise
# abort after 10 min.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"

# Disable profiling CSV/trace output by default — the server is long-running
# and the profiling layer isn't designed for multi-request persistence yet.
unset WAN_PROFILE_DIR || true

exec torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m server.main
