#!/usr/bin/env python3
"""Generate a human-readable profiling summary from timing CSVs.

Usage:
    python scripts/profiling/summarize.py /path/to/profile_dir

Produces a markdown report with:
  - Total generation time
  - Per-phase breakdown (percentage of total)
  - Per-step statistics
  - Peak memory usage
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict


def load_csvs(profile_dir):
    """Load all timing CSVs from a profile directory."""
    pattern = os.path.join(profile_dir, "timing_rank*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No timing CSV files found in {profile_dir}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for path in sorted(files):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def summarize(rows):
    """Build summary from timing rows."""
    # Separate timing and memory rows
    timing_rows = []
    memory_rows = []
    for row in rows:
        name = row["name"]
        if name.startswith("memory/"):
            memory_rows.append(row)
        else:
            timing_rows.append(row)

    # Top-level spans (step=-1, excluding per-step data)
    top_level = defaultdict(list)
    step_data = defaultdict(list)

    for row in timing_rows:
        name = row["name"]
        step = int(row["step"])
        wall_ms = float(row["wall_ms"])
        gpu_ms = float(row["gpu_ms"])

        if step == -1:
            top_level[name].append({"wall_ms": wall_ms, "gpu_ms": gpu_ms})
        else:
            step_data[name].append({
                "step": step, "wall_ms": wall_ms, "gpu_ms": gpu_ms
            })

    # Memory peaks
    peak_mem = 0.0
    for row in memory_rows:
        gpu_ms = float(row["gpu_ms"])  # peak stored in gpu_ms column
        if gpu_ms > peak_mem:
            peak_mem = gpu_ms

    return top_level, step_data, peak_mem


def _median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def report(top_level, step_data, peak_mem):
    """Generate markdown report."""
    lines = []
    lines.append("# Profiling Summary\n")

    # Total generation time
    gen_times = top_level.get("pipeline_generate", [])
    if gen_times:
        total_wall = gen_times[0]["wall_ms"]
        total_gpu = gen_times[0]["gpu_ms"]
        lines.append(f"**Total generation time**: "
                     f"{total_wall:.1f}ms wall / {total_gpu:.1f}ms GPU\n")
    else:
        total_wall = None

    # Phase breakdown
    lines.append("## Phase Breakdown\n")
    lines.append(f"| {'Phase':<30} | {'Wall (ms)':>12} | {'GPU (ms)':>12} |"
                 + (" % of Total |" if total_wall else ""))
    lines.append(f"|{'-'*32}|{'-'*14}|{'-'*14}|"
                 + (f"{'-'*13}|" if total_wall else ""))

    phase_order = [
        "diffusion_loop", "text_encoding", "vae_encode", "vae_decode",
    ]
    seen = set()
    for phase in phase_order:
        if phase in top_level:
            data = top_level[phase]
            wall = data[0]["wall_ms"]
            gpu = data[0]["gpu_ms"]
            pct = f"{wall / total_wall * 100:>10.1f}%" if total_wall else ""
            lines.append(f"| {phase:<30} | {wall:>11.1f}ms | "
                         f"{gpu:>11.1f}ms |{pct:>12} |"
                         if total_wall else
                         f"| {phase:<30} | {wall:>11.1f}ms | "
                         f"{gpu:>11.1f}ms |")
            seen.add(phase)

    for phase, data in sorted(top_level.items()):
        if phase in seen or phase == "pipeline_generate":
            continue
        wall = data[0]["wall_ms"]
        gpu = data[0]["gpu_ms"]
        pct = f"{wall / total_wall * 100:>10.1f}%" if total_wall else ""
        lines.append(f"| {phase:<30} | {wall:>11.1f}ms | "
                     f"{gpu:>11.1f}ms |{pct:>12} |"
                     if total_wall else
                     f"| {phase:<30} | {wall:>11.1f}ms | "
                     f"{gpu:>11.1f}ms |")

    # Per-step breakdown
    if step_data:
        lines.append("\n## Per-Step Statistics (excluding step 0)\n")
        lines.append(f"| {'Span':<25} | {'N':>5} | {'Med Wall':>10} | "
                     f"{'Med GPU':>10} |")
        lines.append(f"|{'-'*27}|{'-'*7}|{'-'*12}|{'-'*12}|")

        for name in ["step", "model_forward_cond", "model_forward_uncond",
                      "guidance_merge", "scheduler_step", "model_prepare"]:
            if name not in step_data:
                continue
            data = step_data[name]
            # Skip step 0
            filtered = [d for d in data if d["step"] > 0]
            if not filtered:
                continue
            n = len(filtered)
            med_wall = _median([d["wall_ms"] for d in filtered])
            med_gpu = _median([d["gpu_ms"] for d in filtered])
            lines.append(f"| {name:<25} | {n:>5} | {med_wall:>9.1f}ms | "
                         f"{med_gpu:>9.1f}ms |")

    # Memory
    if peak_mem > 0:
        lines.append(f"\n## Memory\n")
        lines.append(f"**Peak GPU memory**: {peak_mem:.1f} MB\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", help="Directory with timing CSV files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: stdout)")
    args = parser.parse_args()

    rows = load_csvs(args.profile_dir)
    top_level, step_data, peak_mem = summarize(rows)
    text = report(top_level, step_data, peak_mem)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Summary written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
