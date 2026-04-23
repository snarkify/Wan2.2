#!/usr/bin/env python3
"""Run a profiling config N times and aggregate wall/peak/diffusion stats.

Usage:
    python scripts/profiling/variance_run.py \\
        --name fsdp_uly_81f --runs 3 \\
        --out-root /workspace/profile_outputs \\
        -- \\
        torchrun --nproc_per_node=4 --master_port=29500 generate.py \\
          --task ti2v-5B --size '1280*704' --frame_num 81 \\
          --ckpt_dir /workspace/Wan2.2-TI2V-5B --offload_model True \\
          --dit_fsdp --ulysses_size 4 --prompt 'repro'

The driver appends `--profile_dir <out-root>/<name>_run<i>` to each
invocation. After all runs it reads timing_rank0.csv from each and
prints mean ± std for peak_mb, total_s, diffusion_s.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time


def read_run(profile_dir: str) -> dict | None:
    csv_path = os.path.join(profile_dir, "timing_rank0.csv")
    if not os.path.exists(csv_path):
        return None
    peak = 0.0
    init_ts = None
    end_ts = None
    dit_on = None
    diff_end = None
    with open(csv_path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) < 7:
                continue
            name = parts[2]
            try:
                gpu_ms = float(parts[5])
                ts = float(parts[6])
            except ValueError:
                continue
            if name.startswith("memory/") and not name.startswith("memory_frag"):
                if gpu_ms > peak:
                    peak = gpu_ms
            if name == "memory/init_start":
                init_ts = ts
            elif name == "memory/pipeline_end":
                end_ts = ts
            elif name == "memory/after_dit_to_gpu":
                dit_on = ts
            elif name == "memory/after_diffusion_loop":
                diff_end = ts
    return {
        "peak_mb": peak,
        "total_s": (end_ts - init_ts) if init_ts and end_ts else None,
        "diffusion_s": (diff_end - dit_on) if dit_on and diff_end else None,
    }


def summarize(values: list[float]) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f} (n={len(xs)})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Run name prefix.")
    parser.add_argument("--runs", type=int, default=3, help="Repeat count.")
    parser.add_argument("--out-root", required=True,
                        help="Directory for per-run profile_dir subdirs.")
    parser.add_argument("--skip-first", action="store_true",
                        help="Discard run 0 (warmup) from summary.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Command to run, preceded by '--'.")
    args = parser.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        parser.error("Use '--' before the command to run")
    cmd_tail = args.cmd[1:]

    results = []
    for i in range(args.runs):
        run_name = f"{args.name}_run{i}"
        out = os.path.join(args.out_root, run_name)
        subprocess.run(["rm", "-rf", out], check=False)
        full_cmd = cmd_tail + ["--profile_dir", out]
        print(f"\n=== Run {i + 1}/{args.runs}: {run_name} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(full_cmd)
        wall = time.time() - t0
        print(f"  exit={rc} wall={wall:.1f}s", flush=True)
        stats = read_run(out) if rc == 0 else None
        results.append(stats)

    samples = results[1:] if args.skip_first else results
    print("\n=== Aggregate ===")
    print(f"  peak_mb     : {summarize([r['peak_mb'] for r in samples if r])}")
    print(f"  total_s     : {summarize([r['total_s'] for r in samples if r])}")
    print(f"  diffusion_s : {summarize([r['diffusion_s'] for r in samples if r])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
