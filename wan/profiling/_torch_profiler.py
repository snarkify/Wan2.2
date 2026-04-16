"""Layer 3: Optional torch.profiler wrapper for kernel-level CUPTI tracing."""

import logging
import os

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from wan.profiling._config import ProfileConfig


class TorchProfilerWrapper:
    """Wraps torch.profiler for opt-in kernel-level GPU tracing.

    Activated by WAN_PROFILE_TORCH=1. Profiles specific phases of the
    pipeline (T5 encoding, one diffusion step, VAE decode) to keep
    trace size manageable.

    Usage: call ``phase(name)`` to start profiling a phase, and
    ``phase_end()`` to stop and export.  Inside the diffusion loop,
    ``step()`` tracks the step index; only the step matching
    ``torch_profiler_start`` is profiled.
    """

    def __init__(self, config: ProfileConfig):
        self._config = config
        self._trace_dir = os.path.join(
            config.output_dir, f"torch_trace_rank{config.rank}"
        )
        os.makedirs(self._trace_dir, exist_ok=True)
        self._chrome_trace_path = os.path.join(
            config.output_dir,
            f"torch_chrome_trace_rank{config.rank}.json",
        )
        self._profiler = None
        self._phase_name = None
        self._step_idx = -1
        self._target_step = config.torch_profiler_start
        self._pending_exports = []  # list of (name, profiler) deferred until stop()
        self._phase_traces = []  # list of exported file paths for merging

    def phase(self, name: str) -> None:
        """Start profiling a named phase."""
        if self._profiler is not None:
            return  # already profiling
        self._phase_name = name
        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self._profiler.__enter__()

    def phase_end(self) -> None:
        """Stop profiling the current phase. Export is deferred to stop()."""
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        # Defer export to avoid blocking the pipeline (33MB+ writes)
        self._pending_exports.append((self._phase_name, self._profiler))
        self._profiler = None
        self._phase_name = None

    def step(self) -> None:
        """Called per diffusion step. Profiles only the target step."""
        self._step_idx += 1
        if self._step_idx == self._target_step:
            self.phase(f"step_{self._step_idx}")
        elif self._step_idx == self._target_step + 1 and self._profiler is not None:
            self.phase_end()

    def start(self) -> None:
        """Legacy: called at diffusion loop start. No-op in new design."""
        self._step_idx = -1

    def stop(self) -> None:
        """Called at diffusion loop end. Exports all deferred phases."""
        if self._profiler is not None:
            self.phase_end()
        # Export all deferred phases
        for name, profiler in self._pending_exports:
            phase_path = os.path.join(
                self._trace_dir, f"{name}.pt.trace.json"
            )
            try:
                profiler.export_chrome_trace(phase_path)
                self._phase_traces.append(phase_path)
                logging.info(f"Torch profiler: exported {name} → {phase_path}")
            except Exception as e:
                logging.warning(f"Torch profiler export failed for {name}: {e}")
        self._pending_exports.clear()
        # Merge all phase traces into one chrome trace
        self._merge_phase_traces()

    def _merge_phase_traces(self):
        """Combine per-phase trace files into a single chrome trace."""
        import json
        all_events = []
        for path in self._phase_traces:
            try:
                with open(path) as f:
                    data = json.load(f)
                events = data.get("traceEvents", data) if isinstance(data, dict) else data
                all_events.extend(events)
            except Exception:
                pass
        if all_events:
            with open(self._chrome_trace_path, "w") as f:
                json.dump({"traceEvents": all_events}, f)
            logging.info(
                f"Torch profiler: merged {len(self._phase_traces)} phases "
                f"→ {self._chrome_trace_path} ({len(all_events)} events)"
            )
