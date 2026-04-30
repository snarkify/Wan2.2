"""Runtime configuration from environment variables."""

import os
from dataclasses import dataclass


# Timing formula for the warm 1-GPU + fp8 path on a single RTX 4090.
# Calibrated against bench_artifacts/perf-path-b_20260430-030223_qfp8_aflash_c0_f81_s50_pathb-baseline-fp8.json
# (N=81 -> 291 s warm mean across gen 2 + gen 3; first-job-of-process
# pays an extra ~86 s for the pipeline constructor — surfaced separately
# as _T_COLD_START_BONUS).
#
# Per-frame slope: at world=1 with fp8, the diffusion-step cost scales
# linearly in frame count (sequence length). We have a single measured
# point (N=81 -> 291 s) and hold the constant overhead at 191 s (T5
# encode + VAE decode + scheduler init), leaving (291 - 191) / 81 = 1.235
# s/frame for the 50-step diffusion loop. If we ever land a second N,
# refit both constants.
_T_OFFSET = 191.0
_T_PER_FRAME = 1.235

# Extra time the first job of a fresh server process pays. With
# WAN_DEMO_COMPILE=1 (Phase 3 default), this includes BOTH the WanTI2V
# pipeline constructor (T5 + VAE + DiT load + fp8 quant ≈ 86 s) AND
# the first-forward torch.compile cost (~270 s of Dynamo/Inductor work
# captured the first time the DiT is invoked under compile).
#
# 360 s = 86 s constructor + ~270 s first-forward compile, calibrated
# against Phase 3 acceptance budget (docs/path-b-acceptance.md §3-2:
# first-job-of-process must be ≤ 480 s, i.e. ≤ ~270 s on top of the
# steady-state ≤ 215 s warm wall). After the first job completes, the
# singleton is both built and compiled; every subsequent job is warm.
#
# If `WAN_DEMO_COMPILE=0` is ever set in production, this constant
# overestimates the cold bonus by ~270 s — that just yields a more
# pessimistic ETA, never an under-estimate, so we accept the conservatism
# rather than complicate the formula.
_T_COLD_START_BONUS = 360.0

# Frame-count ceiling measured on 4x RTX 4090 (see plan / profiling-results).
# 141 is the last successful value before VAE decode OOMs at 1280x704.
# Holds at world=1 too: VAE decode is rank-0-only, so the OOM ceiling
# is identical between the 4-GPU and 1-GPU paths.
MAX_FRAMES = 141
MIN_FRAMES = 5


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    token: str
    ckpt_dir: str
    output_dir: str
    max_queue: int
    allow_private_callback: bool
    # Fixed demo defaults
    size: str = "1280*704"
    sampling_steps: int = 50
    # Path B Phase 3: torch.compile on the warm DiT singleton. Default
    # True — compile is the largest remaining lever after sage attention
    # and is numerically near-identical to eager (gating PSNR ≥ 40 dB
    # vs Phase 2). Set `WAN_DEMO_COMPILE=0` to disable for debugging.
    compile_enabled: bool = True
    # Compile mode passed through to torch.compile. `default` is safest
    # with fp8 _scaled_mm; `reduce-overhead` trades ~1 GB extra for
    # CUDA-graph capture but is risky on a 22.2 GB peak / 24 GB 4090.
    compile_mode: str = "default"

    def eta_seconds(
        self,
        queue_position: int,
        frame_num: int,
        pipeline_warm: bool = False,
    ) -> float:
        """Rough ETA including jobs ahead in the queue.

        queue_position is 0-based (0 = runs next). Each ahead-job is
        assumed to take the same per-job time as this one.

        pipeline_warm: if False (the typical case at server-startup
        before the first job has finished), add the one-time pipeline
        constructor cost to the front of the queue. The first job pays
        ~86 s of model-load before its first diffusion step; every job
        thereafter is warm.
        """
        per_job = _T_OFFSET + _T_PER_FRAME * frame_num
        cold_bonus = 0.0 if pipeline_warm else _T_COLD_START_BONUS
        return cold_bonus + (queue_position + 1) * per_job


def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"required env var {var} is not set")
    return val


def _parse_bool(env_name: str, default: bool) -> bool:
    """Parse a boolean env var. Accepts 0/1/true/false/yes/no (any case).
    Empty / unset → default. Anything else → ValueError so misconfig
    surfaces at startup."""
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{env_name} must be 0/1/true/false/yes/no, got {raw!r}"
    )


def load() -> Config:
    # Lazy import: compile_util imports torch, which we don't want to
    # force on the cold path of `from server.config import Config` from
    # tests that just want to look at the dataclass shape.
    from wan.distributed.compile_util import resolve_compile_mode

    return Config(
        host=os.environ.get("WAN_DEMO_HOST", "0.0.0.0"),
        port=int(os.environ.get("WAN_DEMO_PORT", "8000")),
        token=_require("WAN_DEMO_TOKEN"),
        ckpt_dir=_require("WAN_DEMO_CKPT_DIR"),
        output_dir=_require("WAN_DEMO_OUTPUT_DIR"),
        max_queue=int(os.environ.get("WAN_DEMO_MAX_QUEUE", "3")),
        allow_private_callback=os.environ.get(
            "WAN_DEMO_ALLOW_PRIVATE_CALLBACK", "0"
        ) == "1",
        compile_enabled=_parse_bool("WAN_DEMO_COMPILE", default=True),
        compile_mode=resolve_compile_mode(
            os.environ.get("WAN_DEMO_COMPILE_MODE")
        ),
    )
