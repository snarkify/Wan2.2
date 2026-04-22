"""Layer 2: Chrome Trace Format streaming JSON writer for Perfetto."""

import json
import os
import threading
import time

from wan.profiling._config import ProfileConfig


class TraceWriter:
    """Streaming Chrome Trace Format writer.

    Writes events incrementally to avoid unbounded memory usage with
    model detail enabled (~380K events). Format:
      - Init: write '{"traceEvents":[\n'
      - Each event: append ',\n{...}' (first event omits leading comma)
      - Flush/close: append '\n]}'

    Crash produces truncated but recoverable JSON.
    """

    def __init__(self, config: ProfileConfig):
        self._config = config
        self._lock = threading.Lock()
        self._path = os.path.join(
            config.output_dir, f"trace_rank{config.rank}.json"
        )
        self._pid = config.rank
        self._tid = 0  # Main thread
        self._start_ts = time.perf_counter()
        self._event_count = 0
        self._closed = False

        self._file = open(self._path, "w")
        self._file.write('{"traceEvents":[\n')
        # Write process metadata
        self._write_event(
            {
                "ph": "M", "pid": self._pid, "tid": self._tid,
                "name": "process_name",
                "args": {"name": f"Rank {config.rank}"},
            }
        )
        self._write_event(
            {
                "ph": "M", "pid": self._pid, "tid": self._tid,
                "name": "thread_name",
                "args": {"name": "main"},
            }
        )
        # Sync event for trace merging (I-10)
        self._write_event(
            {
                "ph": "i", "pid": self._pid, "tid": self._tid,
                "ts": self._ts_us(), "name": "__trace_sync__", "s": "p",
                "args": {"wall_us": int(time.time() * 1_000_000)},
            }
        )

    def _ts_us(self) -> int:
        return int((time.perf_counter() - self._start_ts) * 1_000_000)

    def _write_event(self, event: dict) -> None:
        prefix = "" if self._event_count == 0 else ","
        self._file.write(prefix + json.dumps(event, separators=(",", ":"))
                         + "\n")
        self._event_count += 1

    def begin(self, name: str, metadata: dict | None = None) -> None:
        event = {
            "ph": "B", "pid": self._pid, "tid": self._tid,
            "ts": self._ts_us(), "name": name,
        }
        if metadata:
            event["args"] = metadata
        with self._lock:
            self._write_event(event)

    def end(self, name: str, metadata: dict | None = None) -> None:
        event = {
            "ph": "E", "pid": self._pid, "tid": self._tid,
            "ts": self._ts_us(), "name": name,
        }
        if metadata:
            event["args"] = metadata
        with self._lock:
            self._write_event(event)

    def instant(self, name: str, category: str = "",
                metadata: dict | None = None) -> None:
        event = {
            "ph": "i", "pid": self._pid, "tid": self._tid,
            "ts": self._ts_us(), "name": name, "s": "p",
        }
        if category:
            event["cat"] = category
        if metadata:
            event["args"] = metadata
        with self._lock:
            self._write_event(event)

    def counter(self, name: str, values: dict) -> None:
        event = {
            "ph": "C", "pid": self._pid, "tid": self._tid,
            "ts": self._ts_us(), "name": name,
            "args": values,
        }
        with self._lock:
            if self._closed:
                return  # swallow late writes (e.g. background sampler)
            self._write_event(event)

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.write("\n]}\n")
                self._file.flush()
                self._file.close()
                self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
