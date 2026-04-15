"""Global per-rank CUDA event buffer with deferred batch resolution."""

import threading
import time
from dataclasses import dataclass, field

import torch


@dataclass
class PendingSpan:
    name: str
    step: int
    wall_ms: float
    cuda_start: torch.cuda.Event
    cuda_end: torch.cuda.Event


class CudaEventBuffer:
    """Collects deferred CUDA event pairs from loop-level spans.

    All CudaTimedSpan instances in deferred mode push here.
    Resolution (one torch.cuda.synchronize + batch elapsed_time) happens at:
      - profiled_loop().__exit__
      - flush()
    """

    def __init__(self):
        self._pending: list[PendingSpan] = []
        self._lock = threading.Lock()

    def push(self, span: PendingSpan) -> None:
        with self._lock:
            self._pending.append(span)

    def resolve_all(self, stopwatch, tracer) -> None:
        """Synchronize once, then resolve every pending event pair."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        with self._lock:
            for span in self._pending:
                try:
                    gpu_ms = span.cuda_start.elapsed_time(span.cuda_end)
                except RuntimeError:
                    gpu_ms = -1.0
                if stopwatch is not None:
                    stopwatch.record(
                        span.name, span.step, span.wall_ms, gpu_ms
                    )
            self._pending.clear()

    def discard_all(self) -> None:
        with self._lock:
            self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending)


# Module-level singleton
_BUFFER = CudaEventBuffer()


def get_buffer() -> CudaEventBuffer:
    return _BUFFER
