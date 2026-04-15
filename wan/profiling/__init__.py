"""Wan2.2 Profiling & Tracing Framework.

Two primary layers + one optional:
  Layer 1: CSV timing records (wall-clock + CUDA event GPU timing)
  Layer 2: Chrome Trace Format timeline for Perfetto visualization
  Layer 3: torch.profiler wrapper (opt-in, kernel-level CUPTI tracing)

Zero-cost when disabled — gated by --profile_dir / WAN_PROFILE_DIR.

Usage in pipeline code:
    from wan.profiling import profiled_loop, trace_span

    with profiled_loop() as loop:
        for step_idx, t in enumerate(tqdm(timesteps)):
            with loop.step(step_idx) as spans:
                with spans.span("model_forward_cond"):
                    pred = model(x, **args)[0]
                with spans.span("scheduler_step"):
                    x0 = scheduler.step(...)
"""

from contextlib import contextmanager

from wan.profiling._config import ProfileConfig, get_config, init_from_args
from wan.profiling._hooks import setup_profiling

__all__ = [
    "trace_span",
    "trace_counter",
    "record_memory",
    "setup_profiling",
    "profiled_loop",
    "flush",
    "init_from_args",
]

# ---------------------------------------------------------------------------
# Lazy-initialized singletons
# ---------------------------------------------------------------------------
_stopwatch = None
_tracer = None
_torch_profiler = None
_initialized = False
_in_loop = False  # Set by profiled_loop(), used by hooks for deferred mode


def _ensure_initialized():
    global _stopwatch, _tracer, _torch_profiler, _initialized
    if _initialized:
        return
    config = get_config()
    if not config.enabled:
        _initialized = True
        return
    import os
    os.makedirs(config.output_dir, exist_ok=True)

    from wan.profiling._stopwatch import StopwatchRecorder
    _stopwatch = StopwatchRecorder(config)

    if config.trace_enabled:
        from wan.profiling._tracer import TraceWriter
        _tracer = TraceWriter(config)

    if config.torch_profiler_enabled:
        try:
            from wan.profiling._torch_profiler import TorchProfilerWrapper
            _torch_profiler = TorchProfilerWrapper(config)
        except Exception:
            pass  # torch.profiler may not be available

    _initialized = True


# ---------------------------------------------------------------------------
# No-op singletons for disabled path
# ---------------------------------------------------------------------------
class _NoopContextManager:
    """Zero-cost stand-in when profiling is disabled."""
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *args): return False


class _NoopStepSpans:
    """No-op StepSpans — span() returns the global no-op."""
    __slots__ = ()
    def span(self, name):
        return _NOOP


class _NoopStepContext:
    """No-op step context manager — yields no-op StepSpans."""
    __slots__ = ()
    def __enter__(self): return _NOOP_SPANS
    def __exit__(self, *args): return False


class _NoopStepProfiler:
    """No-op StepProfiler — step() returns no-op context."""
    __slots__ = ()
    def step(self, step_idx):
        return _NOOP_STEP_CTX


_NOOP = _NoopContextManager()
_NOOP_SPANS = _NoopStepSpans()
_NOOP_STEP_CTX = _NoopStepContext()
_NOOP_PROFILER = _NoopStepProfiler()


# ---------------------------------------------------------------------------
# Active StepProfiler
# ---------------------------------------------------------------------------
class _StepSpans:
    """Sub-step span accessor. Flexible API — no fixed method set (A-2)."""

    __slots__ = ("_step",)

    def __init__(self, step_idx):
        self._step = step_idx

    def span(self, name):
        """Return a deferred trace_span context manager for this sub-step."""
        return _make_deferred_span(name, self._step)


class _StepContext:
    """Context manager for one diffusion step."""

    __slots__ = ("_step", "_span")

    def __init__(self, step_idx):
        self._step = step_idx
        self._span = None

    def __enter__(self):
        self._span = _make_deferred_span("step", self._step)
        self._span.__enter__()
        return _StepSpans(self._step)

    def __exit__(self, *exc):
        result = False
        if self._span is not None:
            result = self._span.__exit__(*exc)
        # Memory snapshot per step
        _record_step_memory(self._step)
        return result


class _StepProfiler:
    """Active StepProfiler — returned by profiled_loop() when enabled."""

    __slots__ = ()

    def step(self, step_idx):
        return _StepContext(step_idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def trace_span(name: str, step: int = -1, metadata: dict | None = None):
    """Context manager for timing a code region.

    Returns a no-op singleton when profiling is disabled (~50-100ns).
    """
    config = get_config()
    if not config.enabled:
        return _NOOP
    _ensure_initialized()
    from wan.profiling._stopwatch import CudaTimedSpan
    return CudaTimedSpan(
        name, step, metadata, _stopwatch, _tracer, config, deferred=False
    )


def _make_deferred_span(name: str, step: int):
    """Create a deferred (non-syncing) span for use inside diffusion loops."""
    config = get_config()
    if not config.enabled:
        return _NOOP
    _ensure_initialized()
    from wan.profiling._stopwatch import CudaTimedSpan
    return CudaTimedSpan(
        name, step, None, _stopwatch, _tracer, config, deferred=True
    )


@contextmanager
def profiled_loop():
    """Context manager wrapping a diffusion loop.

    Yields a StepProfiler for per-step sub-spans. On exit, resolves
    all deferred CUDA events in one batch synchronization.
    """
    config = get_config()
    if not config.enabled:
        yield _NOOP_PROFILER
        return

    _ensure_initialized()

    # Start torch.profiler if enabled
    if _torch_profiler is not None:
        _torch_profiler.start()

    global _in_loop
    from wan.profiling._stopwatch import CudaTimedSpan
    span = CudaTimedSpan(
        "diffusion_loop", -1, None, _stopwatch, _tracer, config, deferred=False
    )
    span.__enter__()
    _in_loop = True
    try:
        yield _StepProfiler()
    finally:
        _in_loop = False
        span.__exit__(None, None, None)
        # Resolve all deferred CUDA events from loop-interior spans
        from wan.profiling._event_buffer import get_buffer
        get_buffer().resolve_all(_stopwatch, _tracer)
        # Stop torch.profiler
        if _torch_profiler is not None:
            _torch_profiler.stop()


def trace_counter(name: str, values: dict) -> None:
    """Emit a Chrome Trace counter event (e.g., GPU memory)."""
    config = get_config()
    if not config.enabled:
        return
    _ensure_initialized()
    if _tracer is not None:
        _tracer.counter(name, values)


def record_memory(label: str = "") -> None:
    """Record GPU memory snapshot as CSV row + Chrome Trace counter."""
    config = get_config()
    if not config.enabled:
        return
    _ensure_initialized()
    from wan.profiling._memory import record_memory_snapshot
    record_memory_snapshot(label, _stopwatch, _tracer, config)


def _record_step_memory(step_idx: int) -> None:
    """Internal: record memory at end of each diffusion step."""
    config = get_config()
    if not config.enabled:
        return
    if _tracer is not None:
        from wan.profiling._memory import record_memory_snapshot
        record_memory_snapshot(f"step_{step_idx}", None, _tracer, config)


def flush() -> None:
    """Synchronize CUDA, resolve deferred events, flush all buffers to disk."""
    config = get_config()
    if not config.enabled:
        return
    _ensure_initialized()
    # Resolve any remaining deferred events
    from wan.profiling._event_buffer import get_buffer
    get_buffer().resolve_all(_stopwatch, _tracer)
    # Flush writers
    if _stopwatch is not None:
        _stopwatch.flush()
    if _tracer is not None:
        _tracer.flush()
