"""Collective (NCCL) instrumentation — wraps all_to_all / all_gather.

For each call we emit a trace span (so it appears in Perfetto alongside
model spans) plus a CSV row with bytes moved. A global accumulator
tracks totals so flush() can emit an aggregate row — useful for answering
'how much of Ulysses's wall time was eaten by all-to-all?'.

Zero-cost when profiling is disabled.
"""

import threading
import time


class _Accumulator:
    __slots__ = ("lock", "count", "bytes", "wall_ms", "gpu_ms")

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.bytes = 0
        self.wall_ms = 0.0
        self.gpu_ms = 0.0


_ACC: dict[str, _Accumulator] = {}


def _get_accumulator(name: str) -> _Accumulator:
    acc = _ACC.get(name)
    if acc is None:
        acc = _Accumulator()
        _ACC[name] = acc
    return acc


def record_collective(op: str, nbytes: int, wall_ms: float, gpu_ms: float) -> None:
    """Called from util.py wrappers after each collective."""
    from wan.profiling import get_config
    config = get_config()
    if not config.enabled:
        return
    acc = _get_accumulator(op)
    with acc.lock:
        acc.count += 1
        acc.bytes += nbytes
        acc.wall_ms += wall_ms
        acc.gpu_ms += gpu_ms

    # Per-call CSV row: name=collective/<op>, wall_ms=wall, gpu_ms=gpu.
    # Metadata in trace goes via the span wrapping the call (see util.py).
    from wan.profiling import _stopwatch
    if _stopwatch is not None:
        _stopwatch.record(
            f"collective/{op}", int(nbytes), wall_ms, gpu_ms
        )


def flush_totals() -> None:
    """Emit one aggregate row per op: collective_total/<op>.

    wall_ms = total wall, gpu_ms = total gpu, step = count, "nbytes"
    packed into the step column (int) for post-hoc analysis.
    """
    from wan.profiling import _stopwatch
    if _stopwatch is None:
        return
    for op, acc in _ACC.items():
        with acc.lock:
            _stopwatch.record(
                f"collective_total/{op}", acc.count,
                acc.wall_ms, acc.gpu_ms
            )
            # Second row packs bytes as wall_ms so awk can read it.
            _stopwatch.record(
                f"collective_total/{op}_bytes", acc.count,
                float(acc.bytes), -1.0
            )
