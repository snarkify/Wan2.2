"""GPU memory snapshot utilities."""

import torch


def record_memory_snapshot(label, stopwatch, tracer, config):
    """Record current GPU memory stats as CSV row + Chrome Trace counter.

    Metrics:
      - memory_allocated_mb: currently allocated by tensors
      - memory_reserved_mb: total reserved by the caching allocator
      - max_memory_allocated_mb: peak since last reset
    """
    if not torch.cuda.is_available():
        return

    try:
        device = config.local_rank
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except RuntimeError:
        return

    values = {
        "memory_allocated_mb": round(allocated, 1),
        "memory_reserved_mb": round(reserved, 1),
        "max_memory_allocated_mb": round(peak, 1),
    }

    # Memory counters disabled in trace (not readable in Perfetto).
    # Still recorded in CSV via stopwatch below.

    if stopwatch is not None:
        name = f"memory/{label}" if label else "memory"
        stopwatch.record(name, -1, allocated, peak)
