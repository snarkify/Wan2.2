"""Parse a Wan bench log for `[N/M]` Dynamo compile-id markers.

Each Dynamo subgraph emits a `[frame_id/recompile_count]` tag in its
log lines. The frame_id increments per traced subgraph; the
recompile_count increments every time that subgraph re-traces under
new guards. "No recompiles after gen 2" means the maximum
recompile_count seen during gens 3+ does not exceed the maximum seen
through gen 2.

Usage:
    python scripts/check_dynamo_recompiles.py /tmp/bench-p3-cat.log

The bench log alternates "=== gen N/M ===" markers with body. We find
the [N/M] markers in each gen-window and report:
- per-gen max recompile_count
- new (frame_id, max_recompile_count) tuples introduced in gens 3+
- verdict: PASS if no recompile_count strictly increases after gen 2
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict


# Two patterns:
#   1. Dynamo log line:   "W0430 09:22:27.609000 ... [0/0] failed during ..."
#   2. Inductor log line: "[0/0] ..." (less common but seen)
# We match `[<int>/<int>]` anywhere on the line.
_TAG_RE = re.compile(r"\[(\d+)/(\d+)\]")
_GEN_RE = re.compile(r"=== gen (\d+)/\d+ ===")


def parse_log(path: str) -> dict[int, dict[int, int]]:
    """Return {gen_idx: {frame_id: max_recompile_count}}."""
    by_gen: dict[int, dict[int, int]] = defaultdict(dict)
    cur_gen = 0  # before "=== gen 1/N ===" we're in setup
    with open(path) as f:
        for line in f:
            mg = _GEN_RE.search(line)
            if mg:
                cur_gen = int(mg.group(1))
                continue
            for fid_s, rc_s in _TAG_RE.findall(line):
                fid, rc = int(fid_s), int(rc_s)
                prev = by_gen[cur_gen].get(fid, -1)
                if rc > prev:
                    by_gen[cur_gen][fid] = rc
    return by_gen


def report(by_gen: dict[int, dict[int, int]]) -> int:
    """Print a per-gen summary; return exit code (0 = clean)."""
    if not by_gen:
        print("ERROR: no [N/M] tags found in log; either compile is "
              "off or the log is missing dynamo output.")
        return 2

    gens = sorted(by_gen.keys())
    print(f"Gens with compile activity: {gens}")
    for g in gens:
        frames = by_gen[g]
        if not frames:
            print(f"  gen {g}: (no [N/M] tags)")
            continue
        max_fid = max(frames)
        max_rc = max(frames.values())
        n_frames = len(frames)
        rc_dist = sorted(frames.values())
        print(
            f"  gen {g}: {n_frames} frame_ids, max_frame_id={max_fid}, "
            f"max_recompile_count={max_rc}, rc_dist_top5={rc_dist[-5:]}"
        )

    # Build cumulative max recompile_count per frame_id through gen 2.
    cum_through_g2: dict[int, int] = {}
    for g in gens:
        if g > 2:
            break
        for fid, rc in by_gen[g].items():
            cum_through_g2[fid] = max(cum_through_g2.get(fid, -1), rc)

    # Check: any frame_id in gen 3+ with rc strictly greater?
    new_recompiles: list[tuple[int, int, int, int]] = []  # (gen, fid, rc, prev)
    new_frames: list[tuple[int, int, int]] = []  # (gen, fid, rc)
    for g in gens:
        if g <= 2:
            continue
        for fid, rc in by_gen[g].items():
            if fid not in cum_through_g2:
                # New subgraph appeared — that's a recompile (a new
                # branch the compiler hadn't seen).
                new_frames.append((g, fid, rc))
            elif rc > cum_through_g2[fid]:
                new_recompiles.append((g, fid, rc, cum_through_g2[fid]))

    print()
    if new_frames:
        print(f"NEW SUBGRAPHS in gen 3+ (total {len(new_frames)}):")
        for g, fid, rc in new_frames[:20]:
            print(f"  gen {g} frame_id={fid} rc={rc}")
        if len(new_frames) > 20:
            print(f"  ... and {len(new_frames) - 20} more")
    else:
        print("NEW SUBGRAPHS in gen 3+: none")

    if new_recompiles:
        print(f"INCREMENTED recompile_counts in gen 3+ (total {len(new_recompiles)}):")
        for g, fid, rc, prev in new_recompiles[:20]:
            print(f"  gen {g} frame_id={fid} rc={prev} -> {rc}")
        if len(new_recompiles) > 20:
            print(f"  ... and {len(new_recompiles) - 20} more")
    else:
        print("INCREMENTED recompile_counts in gen 3+: none")

    print()
    if not new_recompiles and not new_frames:
        print("VERDICT: PASS — no recompiles after gen 2.")
        return 0
    else:
        print("VERDICT: FAIL — recompiles fired in gen 3+; compile cache "
              "is shape-sensitive and needs investigation.")
        return 1


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    by_gen = parse_log(sys.argv[1])
    return report(by_gen)


if __name__ == "__main__":
    sys.exit(main())
