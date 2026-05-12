#!/usr/bin/env python3
"""Run a sweep of profiling configs (matrix of generate.py runs) and emit a comparison table.

Each entry in the YAML `configs` list is one variance_run, repeated `runs` times.
After all configs complete, prints a markdown comparison table and writes
`sweep_results.csv` to `defaults.out_root`.

Usage:
    python scripts/profiling/sweep.py scripts/profiling/configs/phase1_2gpu.yaml
    python scripts/profiling/sweep.py --dry-run scripts/profiling/configs/phase0_smoke_5b.yaml

Reuses scripts/profiling/variance_run.py for per-config N-run aggregation
(shells out so the existing CLI surface stays unchanged for manual use).
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
from variance_run import read_run  # noqa: E402

DEFAULT_KEYS = (
    "task", "ckpt_dir", "size", "frame_num", "sample_steps",
    "prompt", "base_seed", "sample_solver", "sample_shift",
    "sample_guide_scale", "image", "save_file",
)


def parse_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict) or "configs" not in spec:
        sys.exit(f"{path}: top-level mapping with 'configs' list required")
    return spec


def build_command(defaults: dict, cfg: dict, master_port: int) -> list[str]:
    nproc = int(cfg.get("nproc", 1))
    parts: list[str] = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        str(REPO_ROOT / "generate.py"),
    ]
    for key in DEFAULT_KEYS:
        if defaults.get(key) is not None:
            parts.extend([f"--{key}", str(defaults[key])])
    parts.extend(str(a) for a in cfg.get("args", []))
    return parts


def build_env(env_defaults: dict | None, cfg_env: dict | None) -> dict:
    env = os.environ.copy()
    for k, v in (env_defaults or {}).items():
        env[k] = str(v)
    for k, v in (cfg_env or {}).items():
        env[k] = str(v)
    return env


def run_variance(name: str, runs: int, out_root: str, skip_first: bool,
                 cmd_tail: list[str], env: dict, dry_run: bool) -> int:
    variance_script = SCRIPT_DIR / "variance_run.py"
    var_cmd = [
        sys.executable, str(variance_script),
        "--name", name,
        "--runs", str(runs),
        "--out-root", out_root,
    ]
    if skip_first:
        var_cmd.append("--skip-first")
    var_cmd.append("--")
    var_cmd.extend(cmd_tail)
    print(f"\n##### Sweep config: {name} #####", flush=True)
    print(f"$ {' '.join(var_cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(var_cmd, env=env)


def aggregate_extras(profile_dir: str) -> dict:
    csv_path = os.path.join(profile_dir, "timing_rank0.csv")
    out = {"a2a_ms": None, "ag_ms": None, "attn": ""}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) < 7:
                continue
            name = parts[2]
            try:
                wall_ms = float(parts[4])
            except ValueError:
                wall_ms = None
            if name == "collective_total/all_to_all" and wall_ms is not None:
                out["a2a_ms"] = wall_ms
            elif name == "collective_total/all_gather" and wall_ms is not None:
                out["ag_ms"] = wall_ms
            elif name.startswith("env/attn_backend/"):
                out["attn"] = name[len("env/attn_backend/"):]
    return out


def aggregate_runs(name: str, runs: int, out_root: str, skip_first: bool) -> dict:
    per_run = []
    for i in range(runs):
        run_dir = os.path.join(out_root, f"{name}_run{i}")
        base = read_run(run_dir)
        if base is None:
            per_run.append(None)
            continue
        extras = aggregate_extras(run_dir)
        per_run.append({**base, **extras})
    samples = per_run[1:] if skip_first else per_run
    samples = [s for s in samples if s is not None]
    out: dict[str, Any] = {"name": name, "n": len(samples)}
    if not samples:
        return out

    def mean_std(key):
        xs = [s[key] for s in samples if s.get(key) is not None]
        if not xs:
            return (None, None)
        if len(xs) == 1:
            return (xs[0], 0.0)
        return (statistics.mean(xs), statistics.stdev(xs))

    for k in ("total_s", "diffusion_s", "peak_mb", "a2a_ms", "ag_ms"):
        m, s = mean_std(k)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    out["attn"] = next((s["attn"] for s in samples if s.get("attn")), "")
    return out


def fmt(m, s, spec=".1f"):
    if m is None:
        return "n/a"
    if s is None or s == 0.0:
        return f"{m:{spec}}"
    return f"{m:{spec}} ± {s:{spec}}"


def emit_markdown(rows: list[dict], baseline: str | None) -> str:
    headers = ["Config", "n", "total_s", "diffusion_s", "peak_mb", "a2a_ms", "attn"]
    if baseline is not None:
        headers.append(f"Δ vs {baseline}")
    base_row = next((r for r in rows if r["name"] == baseline), None) if baseline else None
    base_diff = base_row.get("diffusion_s_mean") if base_row else None

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [
            r["name"],
            str(r.get("n", 0)),
            fmt(r.get("total_s_mean"), r.get("total_s_std")),
            fmt(r.get("diffusion_s_mean"), r.get("diffusion_s_std")),
            fmt(r.get("peak_mb_mean"), r.get("peak_mb_std"), ".0f"),
            fmt(r.get("a2a_ms_mean"), r.get("a2a_ms_std"), ".0f"),
            r.get("attn", ""),
        ]
        if baseline is not None:
            d = r.get("diffusion_s_mean")
            if r["name"] == baseline:
                cells.append("—")
            elif base_diff and d is not None:
                cells.append(f"{(d - base_diff) / base_diff * 100:+.1f}%")
            else:
                cells.append("n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config_yaml", help="Path to sweep YAML.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--baseline", default=None,
                   help="Config name for Δ column (default: first config).")
    p.add_argument("--port-base", type=int, default=29500,
                   help="torchrun master_port base; +i per config.")
    args = p.parse_args()

    spec = parse_yaml(args.config_yaml)
    defaults = spec.get("defaults", {}) or {}
    env_defaults = spec.get("env_defaults", {}) or {}
    configs = spec["configs"]

    out_root = defaults.get("out_root")
    if not out_root:
        sys.exit("defaults.out_root is required")
    runs = int(defaults.get("runs", 3))
    skip_first = bool(defaults.get("skip_first", True))
    os.makedirs(out_root, exist_ok=True)

    rc_total = 0
    for i, cfg in enumerate(configs):
        cmd_tail = build_command(defaults, cfg, args.port_base + i)
        env = build_env(env_defaults, cfg.get("env"))
        rc = run_variance(cfg["name"], runs, out_root, skip_first,
                          cmd_tail, env, args.dry_run)
        if rc != 0:
            print(f"  !! variance_run rc={rc} for {cfg['name']}",
                  file=sys.stderr)
            rc_total = rc

    if args.dry_run:
        return 0

    rows = [aggregate_runs(c["name"], runs, out_root, skip_first)
            for c in configs]
    baseline = args.baseline or (configs[0]["name"] if configs else None)
    md = emit_markdown(rows, baseline)
    print("\n##### Sweep results #####\n")
    print(md)
    sweep_csv = os.path.join(out_root, "sweep_results.csv")
    write_csv(rows, sweep_csv)
    print(f"\n(written to {sweep_csv})")
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
