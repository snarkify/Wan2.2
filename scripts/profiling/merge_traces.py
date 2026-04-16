#!/usr/bin/env python3
"""Merge per-rank Chrome Trace JSON files into a single Perfetto trace.

Usage:
    python scripts/profiling/merge_traces.py /path/to/profile_dir

Merges trace_rank*.json (Layer 2) and optionally torch_chrome_trace_rank*.json
(Layer 3) into merged_trace.json, viewable in https://ui.perfetto.dev.

Timestamp alignment uses __trace_sync__ events (wall-clock anchors).
"""

import argparse
import glob
import json
import os
import sys


def _find_l3_offset(l3_events, l2_events, profile_dir, phase_name=None):
    """Compute timestamp offset to align L3 events to L2 timeline.

    Args:
        phase_name: If provided, matches this name directly against L2
            spans (e.g. "text_encoding", "step_2", "vae_decode").
    """
    # Find first span event in l3_events
    l3_first_ts = None
    for ev in l3_events:
        if ev.get("ph") in ("X", "B") and "ts" in ev:
            l3_first_ts = ev["ts"]
            break
    if l3_first_ts is None:
        return 0

    # Build L2 anchor map: span name → first begin timestamp
    l2_begin = {}
    for ev in l2_events:
        if ev.get("ph") == "B" and ev.get("pid") == 0:
            name = ev.get("name", "")
            if name not in l2_begin:
                l2_begin[name] = ev["ts"]

    # Direct match by phase name
    if phase_name and phase_name in l2_begin:
        return l2_begin[phase_name] - l3_first_ts

    # ProfilerStep# events (combined trace / old schedule API)
    for ev in l3_events:
        name = ev.get("name", "")
        if name.startswith("ProfilerStep#"):
            step_num = int(name.split("#")[1])
            l2_anchor = l2_begin.get(f"step_{step_num}")
            if l2_anchor is not None:
                return l2_anchor - ev.get("ts", 0)

    return 0



def _find_sync_event(events):
    """Find the __trace_sync__ event to extract wall-clock anchor."""
    for ev in events:
        if ev.get("name") == "__trace_sync__":
            tracer_ts = ev.get("ts", 0)
            wall_us = ev.get("args", {}).get("wall_us", 0)
            return tracer_ts, wall_us
    return 0, 0


def _shift_events(events, offset_us):
    """Shift all event timestamps by offset_us."""
    for ev in events:
        if "ts" in ev:
            ev["ts"] = ev["ts"] + offset_us
        if "dur" in ev:
            pass  # Duration is relative, no shift needed
    return events


def load_trace_file(path):
    """Load a Chrome Trace JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    elif isinstance(data, list):
        return data
    return []


def merge(profile_dir, include_torch=True):
    """Merge all trace files into a single event list."""
    # Layer 2 traces
    l2_pattern = os.path.join(profile_dir, "trace_rank*.json")
    l2_files = sorted(glob.glob(l2_pattern))

    # Layer 3 torch profiler traces (optional)
    l3_pattern = os.path.join(profile_dir, "torch_chrome_trace_rank*.json")
    l3_files = sorted(glob.glob(l3_pattern)) if include_torch else []

    if not l2_files and not l3_files:
        print(f"No trace files found in {profile_dir}", file=sys.stderr)
        sys.exit(1)

    all_events = []

    # Process Layer 2 files — use sync events for alignment
    reference_wall_us = None
    for path in l2_files:
        events = load_trace_file(path)
        tracer_ts, wall_us = _find_sync_event(events)

        if reference_wall_us is None:
            reference_wall_us = wall_us

        # Offset to align to a common wall-clock base
        offset = (wall_us - tracer_ts) - (reference_wall_us - 0)
        events = _shift_events(events, offset)

        # Filter out sync events from output (diagnostic only)
        events = [e for e in events if e.get("name") != "__trace_sync__"]
        all_events.extend(events)

    # Process Layer 3: load per-phase trace files individually for
    # per-phase timestamp alignment (avoids clock drift between phases).
    # Fall back to combined chrome trace if no per-phase files exist.
    l3_phase_files = []
    for td in glob.glob(os.path.join(profile_dir, "torch_trace_rank*")):
        l3_phase_files.extend(
            sorted(glob.glob(os.path.join(td, "*.pt.trace.json"))))

    if l3_phase_files:
        l3_files_to_process = l3_phase_files
    else:
        l3_files_to_process = l3_files  # fall back to combined

    for path in l3_files_to_process:
        events = load_trace_file(path)
        # Extract phase name from filename for per-phase alignment
        phase_name = os.path.basename(path).replace(".pt.trace.json", "").replace(".json", "")
        ts_offset = _find_l3_offset(events, all_events, profile_dir, phase_name=phase_name)

        # Identify CPU pid in L3
        cpu_pid = None
        for ev in events:
            if ev.get("ph") == "M" and ev.get("name") == "process_labels":
                if ev.get("args", {}).get("labels") == "CPU":
                    cpu_pid = ev.get("pid")
                    break

        # Filter: keep CPU + GPU 0, drop unused GPU 1-15, overhead, string pids
        used_gpu_pids = {0}
        filtered = []
        for ev in events:
            pid = ev.get("pid")
            if pid == -1:
                continue
            if isinstance(pid, int) and pid not in used_gpu_pids and pid != cpu_pid:
                if pid > 0:
                    continue
            if isinstance(pid, str):
                continue
            filtered.append(ev)
        events = filtered

        # Remap pids: L3 CPU → 100, L3 GPU 0 → 101
        pid_remap = {}
        if cpu_pid is not None:
            pid_remap[cpu_pid] = 100
        pid_remap[0] = 101

        for ev in events:
            if "ts" in ev:
                ev["ts"] = int(ev["ts"] + ts_offset)
            if "dur" in ev:
                ev["dur"] = int(ev["dur"]) if ev["dur"] >= 1 else 1
            pid = ev.get("pid")
            if pid in pid_remap:
                ev["pid"] = pid_remap[pid]

        # Filter: drop tiny CPU ops, drop events with negative timestamps
        kept = []
        for ev in events:
            ph = ev.get("ph", "")
            ts = ev.get("ts", 0)
            if ts < 0:
                continue
            if ph == "M":
                kept.append(ev)
            elif ev.get("pid") == pid_remap.get(0):
                kept.append(ev)
            elif ph == "X" and ev.get("dur", 0) < 100:
                continue
            else:
                kept.append(ev)
        all_events.extend(kept)

    return all_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", help="Directory with trace JSON files")
    parser.add_argument("--no-torch", action="store_true",
                        help="Exclude torch profiler traces")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: <dir>/merged_trace.json)")
    args = parser.parse_args()

    events = merge(args.profile_dir, include_torch=not args.no_torch)

    output = args.output or os.path.join(
        args.profile_dir, "merged_trace.json"
    )
    with open(output, "w") as f:
        json.dump({"traceEvents": events}, f)

    n_files = len(glob.glob(os.path.join(args.profile_dir, "trace_rank*.json")))
    n_torch = len(glob.glob(
        os.path.join(args.profile_dir, "torch_chrome_trace_rank*.json")
    ))
    print(f"Merged {n_files} L2 traces + {n_torch} L3 traces "
          f"→ {output} ({len(events)} events)")


if __name__ == "__main__":
    main()
