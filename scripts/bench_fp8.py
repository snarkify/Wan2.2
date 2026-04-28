"""Standalone benchmark runner for the demo server's _build_and_run path.

Launches with the same `torchrun --nproc_per_node=4` semantics as the
demo server, runs ONE WanTI2V generation, prints wall-clock + memory
stats, exits.

Usage (from gpu6):
    cd /home/ubuntu/boyu/Wan2.2 && \
        WAN_DEMO_QUANT=bf16 \
        torchrun --nproc_per_node=4 --master_port=29610 \
        scripts/bench_fp8.py \
            --ckpt /home/ubuntu/boyu/Wan2.2/Wan2.2-TI2V-5B \
            --out /tmp/bench_bf16.mp4 \
            --prompt "A cat walking in a sunlit meadow" \
            --frames 81 --width 1280 --height 704 --seed 42 --steps 50

To toggle fp8: set WAN_DEMO_QUANT=fp8.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _init_distributed() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _peak_mem_mb(local_rank: int) -> dict:
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated(local_rank) // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(local_rank) // (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--stats-out", default="", help="Where to write JSON stats (rank 0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("bench")

    rank, world, local_rank = _init_distributed()
    quant = os.environ.get("WAN_DEMO_QUANT", "bf16")
    log.info(
        "bench start rank=%d world=%d local_rank=%d quant=%s",
        rank, world, local_rank, quant,
    )

    # Reset peak so we get this run's high-water mark.
    torch.cuda.reset_peak_memory_stats(local_rank)

    # Import after dist init / CUDA visible.
    from server.worker import _build_and_run

    kwargs = {
        "input_prompt": args.prompt,
        "size": (args.width, args.height),
        "frame_num": args.frames,
        "seed": args.seed,
        "sampling_steps": args.steps,
        "offload_model": True,
    }

    save_path = args.out if rank == 0 else None
    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.perf_counter()
    try:
        result_path = _build_and_run(args.ckpt, kwargs, save_path)
    except Exception:
        log.exception("bench: _build_and_run raised on rank=%d", rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    mem = _peak_mem_mb(local_rank)
    log.info(
        "bench done rank=%d elapsed=%.2fs peak_alloc=%dMB peak_reserved=%dMB result=%s",
        rank, elapsed, mem["peak_alloc_mb"], mem["peak_reserved_mb"], result_path,
    )

    # Gather peak alloc from every rank to rank 0 for the stats file.
    peak_per_rank = [0] * world
    peak_per_rank[rank] = mem["peak_alloc_mb"]
    peak_tensor = torch.tensor(peak_per_rank, dtype=torch.long, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_per_rank = peak_tensor.tolist()

    if rank == 0 and args.stats_out:
        stats = {
            "quant": quant,
            "world_size": world,
            "frames": args.frames,
            "size": [args.width, args.height],
            "seed": args.seed,
            "steps": args.steps,
            "elapsed_s": round(elapsed, 3),
            "peak_alloc_mb_per_rank": peak_per_rank,
            "result_path": result_path,
        }
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("bench: stats written to %s", args.stats_out)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
