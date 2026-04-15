"""Layer 3: Optional torch.profiler wrapper for kernel-level CUPTI tracing."""

import os

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from wan.profiling._config import ProfileConfig


class TorchProfilerWrapper:
    """Wraps torch.profiler for opt-in kernel-level GPU tracing.

    Activated by WAN_PROFILE_TORCH=1. Runs on a configurable window
    of diffusion steps. Exports Chrome Trace Format (mergeable with
    Layer 2 via post-processing) + TensorBoard traces.
    """

    def __init__(self, config: ProfileConfig):
        self._config = config
        trace_dir = os.path.join(
            config.output_dir, f"torch_trace_rank{config.rank}"
        )
        os.makedirs(trace_dir, exist_ok=True)

        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=max(0, config.torch_profiler_start - 1),
                warmup=1,
                active=config.torch_profiler_end - config.torch_profiler_start,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self._chrome_trace_path = os.path.join(
            config.output_dir,
            f"torch_chrome_trace_rank{config.rank}.json",
        )
        self._started = False

    def start(self) -> None:
        self._profiler.__enter__()
        self._started = True

    def step(self) -> None:
        if self._started:
            self._profiler.step()

    def stop(self) -> None:
        if self._started:
            self._profiler.__exit__(None, None, None)
            self._started = False
            # Export Chrome Trace for merging with Layer 2
            try:
                self._profiler.export_chrome_trace(self._chrome_trace_path)
            except Exception:
                pass
