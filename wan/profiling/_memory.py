"""GPU memory snapshot utilities."""

import torch


def record_memory_snapshot(label, stopwatch, tracer, config, reset_peak=False):
    """Record current GPU memory stats as CSV row + Chrome Trace counter.

    CSV row uses wall_ms=allocated_mb, gpu_ms=peak_mb as a packing hack.

    Args:
        reset_peak: If True, reset the peak memory stats after recording, so
            the next snapshot's peak captures only the intervening section.
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

    if stopwatch is not None:
        name = f"memory/{label}" if label else "memory"
        stopwatch.record(name, -1, allocated, peak)
        # Fragmentation = reserved - allocated. Large values mean the caching
        # allocator is holding onto freed blocks that can't be reused for the
        # next alloc (usually due to size mismatch). wall_ms carries MB.
        stopwatch.record(
            f"memory_frag/{label}" if label else "memory_frag",
            -1, max(0.0, reserved - allocated), reserved,
        )

    if reset_peak:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            pass
