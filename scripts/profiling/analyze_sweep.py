#!/usr/bin/env python3
"""Cross-config sweep analyzer — extracts WHY one config wins, not just by how much.

Walks a sweep out_root, groups run dirs by config name, and produces a
markdown report with:
  1. Per-span median timing across runs (with skip-first dropped)
  2. Collective bandwidth aggregates: bytes shipped, ms spent, effective GB/s
  3. NVML GPU utilization stats: mean, max, dips
  4. Per-step diffusion median + p95

Usage:
    python scripts/profiling/analyze_sweep.py /workspace/profile_outputs/phase1_720p
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

# Reuse stat helpers from aggregate_csv.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_csv import _median, _percentile, _mean, _stdev


def discover_runs(sweep_root: str) -> dict[str, list[str]]:
    """Group run dirs by config name. Convention: <name>_run<i>."""
    by_config: dict[str, list[str]] = defaultdict(list)
    for entry in sorted(os.listdir(sweep_root)):
        full = os.path.join(sweep_root, entry)
        if not os.path.isdir(full):
            continue
        if "_run" not in entry:
            continue
        # Split on the last "_run<digits>" suffix.
        idx = entry.rfind("_run")
        config_name = entry[:idx]
        try:
            run_idx = int(entry[idx + 4:])
        except ValueError:
            continue
        by_config[config_name].append((run_idx, full))
    # Sort each list by run index.
    return {k: [p for _, p in sorted(v)] for k, v in by_config.items()}


def read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def per_span_medians(run_dirs: list[str], skip_first_run: bool = True) -> dict[str, dict]:
    """For each non-memory span name, median wall_ms / gpu_ms across runs.

    Within each run we take the median across steps (excluding warmup step 0).
    Then across runs we take the median.
    """
    if skip_first_run and len(run_dirs) > 1:
        run_dirs = run_dirs[1:]

    per_run: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for run in run_dirs:
        rank0 = os.path.join(run, "timing_rank0.csv")
        rows = read_csv_rows(rank0)
        # Group by name, collect wall/gpu lists per run
        per_step: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            name = row.get("name", "")
            if name.startswith(("memory/", "memory_frag/", "collective/", "collective_total/", "gpu/", "env/")):
                continue
            try:
                step = int(row["step"])
                wall = float(row["wall_ms"])
                gpu = float(row["gpu_ms"])
            except (ValueError, KeyError):
                continue
            if step == 0 and len(per_step.get(name, [])) > 0:
                # Skip the warmup step for per-step spans
                continue
            if gpu < 0:
                # Profiler dropped this span (compile region etc.)
                pass  # keep wall, but exclude gpu in that span only
            per_step[name].append((wall, gpu))
        for name, pairs in per_step.items():
            walls = [w for w, _ in pairs]
            gpus = [g for _, g in pairs if g >= 0]
            if walls:
                per_run[name].append((
                    _median(walls),
                    _median(gpus) if gpus else -1.0,
                ))

    results: dict[str, dict] = {}
    for name, samples in per_run.items():
        walls = [w for w, _ in samples]
        gpus = [g for _, g in samples if g >= 0]
        results[name] = {
            "n_runs": len(samples),
            "wall_median_ms": _median(walls),
            "wall_p95_ms": _percentile(walls, 95),
            "gpu_median_ms": _median(gpus) if gpus else None,
        }
    return results


def collective_aggregates(run_dirs: list[str], skip_first_run: bool = True) -> dict:
    """Sum collective bytes/ms across the measured runs.

    Pulls 'collective_total/<op>' (final aggregate ms) and
    'collective_total/<op>_bytes' (final aggregate byte count) rows.
    """
    if skip_first_run and len(run_dirs) > 1:
        run_dirs = run_dirs[1:]

    totals: dict[str, dict] = defaultdict(lambda: {"wall_ms": [], "gpu_ms": [], "bytes": []})
    for run in run_dirs:
        rows = read_csv_rows(os.path.join(run, "timing_rank0.csv"))
        for row in rows:
            name = row.get("name", "")
            if not name.startswith("collective_total/"):
                continue
            stem = name[len("collective_total/"):]
            try:
                wall = float(row["wall_ms"])
                gpu = float(row["gpu_ms"])
            except (ValueError, KeyError):
                continue
            if stem.endswith("_bytes"):
                op = stem[:-len("_bytes")]
                totals[op]["bytes"].append(wall)  # bytes are stashed in wall_ms column
            else:
                totals[stem]["wall_ms"].append(wall)
                if gpu >= 0:
                    totals[stem]["gpu_ms"].append(gpu)
    out: dict[str, dict] = {}
    for op, d in totals.items():
        out[op] = {
            "wall_ms_median": _median(d["wall_ms"]) if d["wall_ms"] else 0.0,
            "gpu_ms_median": _median(d["gpu_ms"]) if d["gpu_ms"] else None,
            "bytes_median": _median(d["bytes"]) if d["bytes"] else 0.0,
        }
        if out[op]["bytes_median"] and out[op]["wall_ms_median"]:
            out[op]["effective_gbps"] = (
                out[op]["bytes_median"] / 1e9 / (out[op]["wall_ms_median"] / 1000.0)
            )
        else:
            out[op]["effective_gbps"] = None
    return out


def gpu_util_stats(run_dirs: list[str], skip_first_run: bool = True) -> dict:
    """Pull NVML 'gpu/util' samples (% over rolling ~1s window).

    Filter to during-diffusion only (between memory/pipeline_init and
    memory/pipeline_end, with a buffer for VAE decode tail).
    """
    if skip_first_run and len(run_dirs) > 1:
        run_dirs = run_dirs[1:]

    util_samples: list[float] = []
    mem_samples: list[float] = []
    power_samples: list[float] = []
    for run in run_dirs:
        rows = read_csv_rows(os.path.join(run, "timing_rank0.csv"))
        for row in rows:
            name = row.get("name", "")
            if name == "gpu/util":
                try:
                    util_samples.append(float(row["wall_ms"]))
                except (ValueError, KeyError):
                    pass
            elif name == "gpu/mem_used_mb":
                try:
                    mem_samples.append(float(row["wall_ms"]))
                except (ValueError, KeyError):
                    pass
            elif name == "gpu/power_w":
                try:
                    power_samples.append(float(row["wall_ms"]))
                except (ValueError, KeyError):
                    pass

    out = {}
    if util_samples:
        out["util_mean_pct"] = _mean(util_samples)
        out["util_p10_pct"] = _percentile(util_samples, 10)
        out["util_max_pct"] = max(util_samples)
        out["util_n_samples"] = len(util_samples)
    if mem_samples:
        out["mem_max_mb"] = max(mem_samples)
    if power_samples:
        out["power_max_w"] = max(power_samples)
        out["power_mean_w"] = _mean(power_samples)
    return out


def fmt(x, spec=".1f"):
    if x is None:
        return "—"
    return f"{x:{spec}}"


def emit_report(by_config: dict, sweep_root: str, baseline: str | None) -> str:
    lines = []
    lines.append(f"# Cross-config analysis: {sweep_root}\n")

    # Section 1: per-span timing
    lines.append("## Per-span median timing (wall_ms, ranked by total contribution)")
    lines.append("")
    span_names = sorted(
        {s for cfg in by_config.values() for s in cfg.get("spans", {})},
        key=lambda n: -max(
            cfg["spans"].get(n, {}).get("wall_median_ms", 0)
            for cfg in by_config.values()
        ),
    )
    cols = list(by_config.keys())
    header = "| Span | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for span in span_names:
        row_cells = [span]
        for cfg in cols:
            data = by_config[cfg].get("spans", {}).get(span)
            if data is None:
                row_cells.append("—")
            else:
                row_cells.append(f"{fmt(data['wall_median_ms'], '.1f')}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Section 2: collective aggregates
    lines.append("## Collective totals per generation (median across runs)")
    lines.append("")
    lines.append("| Config | op | wall_ms | gpu_ms | bytes | eff GB/s |")
    lines.append("|---|---|---|---|---|---|")
    for cfg, info in by_config.items():
        for op, agg in info.get("collectives", {}).items():
            lines.append(
                f"| {cfg} | {op} | {fmt(agg['wall_ms_median'], '.0f')} "
                f"| {fmt(agg['gpu_ms_median'], '.0f')} "
                f"| {fmt(agg['bytes_median'] / 1e9, '.2f')} GB "
                f"| {fmt(agg.get('effective_gbps'), '.2f')} |"
            )
    lines.append("")

    # Section 3: GPU utilization
    lines.append("## GPU utilization & power (NVML samples, rank 0)")
    lines.append("")
    lines.append("| Config | util_mean | util_p10 | util_max | power_mean_W | power_max_W | mem_max_MB |")
    lines.append("|---|---|---|---|---|---|---|")
    for cfg, info in by_config.items():
        u = info.get("gpu", {})
        lines.append(
            f"| {cfg} | {fmt(u.get('util_mean_pct'))}% "
            f"| {fmt(u.get('util_p10_pct'))}% "
            f"| {fmt(u.get('util_max_pct'))}% "
            f"| {fmt(u.get('power_mean_w'))} "
            f"| {fmt(u.get('power_max_w'))} "
            f"| {fmt(u.get('mem_max_mb'), '.0f')} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_root")
    p.add_argument("--baseline", default=None,
                   help="Config name to use as baseline column (default: first).")
    p.add_argument("--no-skip-first", action="store_true",
                   help="Include first run in aggregation (default: skip).")
    p.add_argument("-o", "--output", default=None,
                   help="Output markdown path (default: <sweep_root>/analysis.md).")
    args = p.parse_args()

    skip_first = not args.no_skip_first
    runs_by_config = discover_runs(args.sweep_root)
    if not runs_by_config:
        print(f"No <name>_run<i> directories under {args.sweep_root}",
              file=sys.stderr)
        return 1

    by_config = {}
    for cfg, run_dirs in runs_by_config.items():
        by_config[cfg] = {
            "spans": per_span_medians(run_dirs, skip_first_run=skip_first),
            "collectives": collective_aggregates(run_dirs, skip_first_run=skip_first),
            "gpu": gpu_util_stats(run_dirs, skip_first_run=skip_first),
            "n_runs_total": len(run_dirs),
        }

    md = emit_report(by_config, args.sweep_root, args.baseline)
    out = args.output or os.path.join(args.sweep_root, "analysis.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"\n(written to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
