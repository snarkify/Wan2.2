"""Layer 1: CSV timing records with wall-clock + CUDA event GPU timing."""

import csv
import os
import threading
import time

import torch

from wan.profiling._config import ProfileConfig
from wan.profiling._event_buffer import PendingSpan, get_buffer


class StopwatchRecorder:
    """Append-only CSV writer for timing records. Thread-safe."""

    def __init__(self, config: ProfileConfig):
        self._config = config
        self._lock = threading.Lock()
        self._buffer: list[list] = []
        self._path = os.path.join(
            config.output_dir, f"timing_rank{config.rank}.csv"
        )
        self._file = open(self._path, "a", newline="")
        self._writer = csv.writer(self._file)
        # Write header if file is new/empty
        if self._file.tell() == 0:
            self._writer.writerow(
                ["run_id", "rank", "name", "step", "wall_ms", "gpu_ms",
                 "timestamp"]
            )
            self._file.flush()

    def record(
        self, name: str, step: int, wall_ms: float, gpu_ms: float
    ) -> None:
        row = [
            self._config.run_id,
            self._config.rank,
            name,
            step,
            f"{wall_ms:.3f}",
            f"{gpu_ms:.3f}",
            f"{time.time():.6f}",
        ]
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self._config.flush_interval:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._file.close()


class CudaTimedSpan:
    """Context manager that records wall-clock and CUDA event timing.

    Two modes:
      - deferred=False (default for macro spans): sync in __exit__, write immediately
      - deferred=True (for loop-interior spans): push to global buffer, resolved later
    """

    __slots__ = (
        "name", "step", "metadata", "_stopwatch", "_tracer", "_config",
        "_deferred", "_wall_start", "_cuda_start", "_cuda_end",
    )

    def __init__(self, name, step, metadata, stopwatch, tracer, config,
                 deferred=False):
        self.name = name
        self.step = step
        self.metadata = metadata
        self._stopwatch = stopwatch
        self._tracer = tracer
        self._config = config
        self._deferred = deferred

    def __enter__(self):
        if self._config.sync_before_timing and not self._deferred:
            torch.cuda.synchronize()
        self._cuda_start = torch.cuda.Event(enable_timing=True)
        self._cuda_end = torch.cuda.Event(enable_timing=True)
        self._cuda_start.record()
        self._wall_start = time.perf_counter()
        if self._tracer:
            self._tracer.begin(self.name, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # ALWAYS record end event, even on exception (I-2)
            self._cuda_end.record()
            wall_ms = (time.perf_counter() - self._wall_start) * 1000.0

            if exc_type is not None:
                # Exception path: do NOT push to buffer — elapsed_time
                # may return garbage. Write wall-clock only with gpu_ms=-1.
                if self._tracer:
                    self._tracer.end(
                        self.name,
                        {"status": "failed", "error": str(exc_val)[:200]},
                    )
                if self._stopwatch:
                    self._stopwatch.record(self.name, self.step, wall_ms, -1.0)
            elif self._deferred:
                # Deferred path: push to global buffer
                pending = PendingSpan(
                    name=self.name,
                    step=self.step,
                    wall_ms=wall_ms,
                    cuda_start=self._cuda_start,
                    cuda_end=self._cuda_end,
                )
                get_buffer().push(pending)
                if self._tracer:
                    self._tracer.end(
                        self.name, {"wall_ms": f"{wall_ms:.3f}"}
                    )
            else:
                # Immediate path: sync and resolve now
                if self._config.sync_before_timing:
                    torch.cuda.synchronize()
                    gpu_ms = self._cuda_start.elapsed_time(self._cuda_end)
                else:
                    torch.cuda.synchronize()
                    gpu_ms = self._cuda_start.elapsed_time(self._cuda_end)
                if self._stopwatch:
                    self._stopwatch.record(
                        self.name, self.step, wall_ms, gpu_ms
                    )
                if self._tracer:
                    self._tracer.end(
                        self.name,
                        {"wall_ms": f"{wall_ms:.3f}",
                         "gpu_ms": f"{gpu_ms:.3f}"},
                    )
        except Exception:
            pass  # Profiling must never crash the pipeline
        return False  # Never suppress the original exception
