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

    # Process Layer 3 files — assign to separate tid range
    for path in l3_files:
        events = load_trace_file(path)
        # Assign torch profiler events to high tid range to separate tracks
        for ev in events:
            if "tid" in ev:
                ev["tid"] = ev.get("tid", 0) + 1000
        all_events.extend(events)

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
