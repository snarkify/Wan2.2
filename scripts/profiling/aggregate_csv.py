#!/usr/bin/env python3
"""Aggregate profiling CSV files: warmup filtering, median/stats per span.

Usage:
    python scripts/profiling/aggregate_csv.py /path/to/profile_dir

Reads timing_rank*.csv files, filters warmup (step 0), computes
median/mean/std/p95 per span name, and outputs summary.csv + table.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path


def _percentile(values, pct):
    """Compute percentile without numpy dependency."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (pct / 100.0) * (len(sorted_v) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_v) - 1)
    frac = idx - lower
    return sorted_v[lower] * (1 - frac) + sorted_v[upper] * frac


def _median(values):
    return _percentile(values, 50)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _stdev(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


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


def aggregate(rows, skip_step_0=True):
    """Group by (name, rank) and compute stats.

    Args:
        rows: List of CSV row dicts.
        skip_step_0: If True, filter out step=0 rows (warmup).
    """
    groups = defaultdict(lambda: {"wall_ms": [], "gpu_ms": []})

    for row in rows:
        name = row["name"]
        step = int(row["step"])
        rank = int(row["rank"])
        wall_ms = float(row["wall_ms"])
        gpu_ms = float(row["gpu_ms"])

        # Skip warmup (step 0)
        if skip_step_0 and step == 0:
            continue

        # Skip memory records (different schema)
        if name.startswith("memory/"):
            continue

        # Skip failed spans
        if gpu_ms < 0:
            continue

        key = (name, rank)
        groups[key]["wall_ms"].append(wall_ms)
        groups[key]["gpu_ms"].append(gpu_ms)

    results = []
    for (name, rank), data in sorted(groups.items()):
        wall = data["wall_ms"]
        gpu = data["gpu_ms"]
        if not wall:
            continue
        results.append({
            "name": name,
            "rank": rank,
            "n": len(wall),
            "wall_median_ms": round(_median(wall), 3),
            "wall_mean_ms": round(_mean(wall), 3),
            "wall_std_ms": round(_stdev(wall), 3),
            "wall_p95_ms": round(_percentile(wall, 95), 3),
            "gpu_median_ms": round(_median(gpu), 3),
            "gpu_mean_ms": round(_mean(gpu), 3),
            "gpu_std_ms": round(_stdev(gpu), 3),
            "gpu_p95_ms": round(_percentile(gpu, 95), 3),
            "wall_total_ms": round(sum(wall), 3),
            "gpu_total_ms": round(sum(gpu), 3),
        })
    return results


def write_summary(results, output_path):
    """Write aggregated results to CSV."""
    if not results:
        print("No data to aggregate.", file=sys.stderr)
        return

    fields = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def print_table(results):
    """Print human-readable summary table."""
    if not results:
        return

    print(f"\n{'Span Name':<35} {'Rank':>4} {'N':>5} "
          f"{'Wall Med':>10} {'Wall P95':>10} "
          f"{'GPU Med':>10} {'GPU P95':>10} "
          f"{'Wall Tot':>10}")
    print("-" * 110)

    for r in results:
        print(f"{r['name']:<35} {r['rank']:>4} {r['n']:>5} "
              f"{r['wall_median_ms']:>9.1f}ms {r['wall_p95_ms']:>9.1f}ms "
              f"{r['gpu_median_ms']:>9.1f}ms {r['gpu_p95_ms']:>9.1f}ms "
              f"{r['wall_total_ms']:>9.1f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", help="Directory with timing CSV files")
    parser.add_argument("--no-skip-warmup", action="store_true",
                        help="Don't skip step 0 (warmup)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <dir>/summary.csv)")
    args = parser.parse_args()

    rows = load_csvs(args.profile_dir)
    results = aggregate(rows, skip_step_0=not args.no_skip_warmup)

    output = args.output or os.path.join(args.profile_dir, "summary.csv")
    write_summary(results, output)
    print(f"Summary written to {output}")
    print_table(results)


if __name__ == "__main__":
    main()
