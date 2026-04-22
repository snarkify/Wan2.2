"""Background GPU sampler — polls nvml for utilization, clocks, power, memory.

Runs in a dedicated thread at a fixed interval (default 100 ms) and
emits each sample to both the CSV stopwatch and the Chrome Trace
counter stream, so utilization shows up as a line in Perfetto.

Zero-cost when disabled (no thread spawned). Graceful if pynvml is
missing — just logs once and exits.

CSV schema for sampler rows (all rows share the schema of the rest of
the file: run_id,rank,name,step,wall_ms,gpu_ms,timestamp):
  - name column carries the metric name (e.g. gpu/util, gpu/power_w)
  - wall_ms column carries the metric value
  - gpu_ms column is always -1 for sampler rows
  - step is always -1

Caveats documented in docs/profiling-results.md:
  - nvmlDeviceGetUtilizationRates returns a rolling ~1 s average,
    not an instantaneous reading (NVIDIA NVML behaviour, known).
  - gpu/mem_bw_pct is memory-BANDWIDTH utilization (% of time any
    read/write was in flight), NOT memory capacity used. For capacity
    see gpu/mem_used_mb.
  - throttle_reasons is the raw NVML bitmask; 0 = none.
"""

import os
import threading
import time


# NVML clock-throttle-reason bitmask — kept as an int in the CSV so we
# can decode later with a grep+awk. 0 means "no throttling".
_THROTTLE_BIT_NAMES = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}


def _resolve_nvml_handle(pynvml, local_rank: int):
    """Resolve the nvml handle for the GPU torch's local_rank maps to.

    pynvml.nvmlDeviceGetHandleByIndex uses PHYSICAL device indices and
    does NOT honour CUDA_VISIBLE_DEVICES. Under torchrun with
    CUDA_VISIBLE_DEVICES set (common on multi-tenant GPU hosts), using
    local_rank as the nvml index samples the wrong GPU silently.

    We instead ask torch for the UUID of local_rank, then look up the
    nvml handle by UUID. Fall back to index only if the UUID path fails.
    """
    try:
        import torch
        if torch.cuda.is_available():
            uuid = str(torch.cuda.get_device_properties(local_rank).uuid)
            # NVML wants "GPU-<uuid>" or the raw uuid with hyphens —
            # both forms accepted on recent drivers; try prefixed first.
            for candidate in (f"GPU-{uuid}", uuid):
                try:
                    return pynvml.nvmlDeviceGetHandleByUUID(
                        candidate.encode()
                        if isinstance(candidate, str) else candidate
                    )
                except Exception:
                    continue
    except Exception:
        pass
    # Fallback: physical-index lookup. Correct when CUDA_VISIBLE_DEVICES
    # is unset or equals the identity permutation.
    return pynvml.nvmlDeviceGetHandleByIndex(local_rank)


def _should_sample_this_rank(config) -> bool:
    """Env WAN_PROFILE_GPU_SAMPLER_RANKS controls which ranks sample.

    Values:
      "0"   — rank 0 only (default; avoids 4x redundant sample volume)
      "all" — every rank samples its own GPU
      "0,2" — comma-separated rank whitelist
    """
    spec = os.environ.get("WAN_PROFILE_GPU_SAMPLER_RANKS", "0").strip()
    if spec == "all":
        return True
    try:
        wanted = {int(x) for x in spec.split(",") if x.strip()}
    except ValueError:
        return config.rank == 0
    return config.rank in wanted


class GpuSampler:
    """Poll nvml in a loop. Stop cleanly on close()."""

    __slots__ = (
        "_config", "_stopwatch", "_tracer", "_interval_s",
        "_thread", "_stop", "_stopped", "_nvml", "_handle",
    )

    def __init__(self, config, stopwatch, tracer, interval_ms: int = 100):
        self._config = config
        self._stopwatch = stopwatch
        self._tracer = tracer
        self._interval_s = max(0.01, interval_ms / 1000.0)
        self._thread = None
        self._stop = threading.Event()
        self._stopped = False  # set after close(), silences any late emits
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        if not _should_sample_this_rank(self._config):
            return
        try:
            import pynvml
        except ImportError:
            return  # silently skip
        try:
            pynvml.nvmlInit()
            self._handle = _resolve_nvml_handle(
                pynvml, self._config.local_rank
            )
            self._nvml = pynvml
        except Exception:
            return

        self._thread = threading.Thread(
            target=self._run, name="gpu-sampler", daemon=True
        )
        self._thread.start()

    # -- emit helpers ---------------------------------------------------
    def _record(self, name: str, value: float) -> None:
        """Write one sampler metric as its own CSV row.

        wall_ms carries the value; gpu_ms is -1 (sampler rows never time
        a code region). _stopped gate prevents writes after close().
        """
        if self._stopped or self._stopwatch is None:
            return
        self._stopwatch.record(name, -1, float(value), -1.0)

    def _emit_counter(self, name: str, values: dict) -> None:
        if self._stopped or self._tracer is None:
            return
        try:
            self._tracer.counter(name, values)
        except Exception:
            pass

    # -- main loop ------------------------------------------------------
    def _run(self) -> None:
        nvml = self._nvml
        handle = self._handle
        while not self._stop.is_set():
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_pct = int(util.gpu)
                mem_bw_pct = int(util.memory)
            except Exception:
                gpu_pct = -1
                mem_bw_pct = -1

            try:
                power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power_w = -1.0

            try:
                sm_clock = nvml.nvmlDeviceGetClockInfo(
                    handle, nvml.NVML_CLOCK_SM
                )
            except Exception:
                sm_clock = -1

            try:
                temp = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = -1

            try:
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = meminfo.used / (1024.0 * 1024.0)
            except Exception:
                mem_used_mb = -1.0

            try:
                throttle = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                    handle
                )
            except Exception:
                throttle = -1

            # CSV: one row per metric — unambiguous for grep/awk.
            self._record("gpu/util", gpu_pct)
            self._record("gpu/mem_bw_pct", mem_bw_pct)
            self._record("gpu/power_w", power_w)
            self._record("gpu/sm_mhz", sm_clock)
            self._record("gpu/temp_c", temp)
            self._record("gpu/mem_used_mb", mem_used_mb)
            self._record("gpu/throttle_reasons", throttle)

            # Chrome counter — everything in one event so Perfetto
            # draws a single multi-series counter lane.
            self._emit_counter(
                "gpu_sampler",
                {
                    "gpu_pct": gpu_pct,
                    "mem_bw_pct": mem_bw_pct,
                    "power_w": round(power_w, 1) if power_w >= 0 else -1,
                    "sm_mhz": sm_clock,
                    "temp_c": temp,
                    "mem_used_mb": int(mem_used_mb) if mem_used_mb >= 0 else -1,
                    "throttle": throttle,
                },
            )

            self._stop.wait(self._interval_s)

    def close(self) -> None:
        if self._thread is None:
            # Either disabled by rank filter or nvml unavailable.
            self._stopped = True
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        # Gate any late emits (e.g. if join timed out) BEFORE writers close.
        self._stopped = True
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
