"""Runtime configuration from environment variables."""

import os
from dataclasses import dataclass


# Timing formula from docs/profiling-results.md:
#   T(seconds) ≈ 443 + 2.68 * N  for N-frame jobs on 4x RTX 4090
_T_OFFSET = 443.0
_T_PER_FRAME = 2.68

# Frame-count ceiling measured on 4x RTX 4090 (see plan / profiling-results).
# 141 is the last successful value before VAE decode OOMs at 1280x704.
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

    def eta_seconds(self, queue_position: int, frame_num: int) -> float:
        """Rough ETA including jobs ahead in the queue.

        queue_position is 0-based (0 = runs next). Each ahead-job is
        assumed to take the same ~(443 + 2.68*N) as the current one.
        """
        per_job = _T_OFFSET + _T_PER_FRAME * frame_num
        return (queue_position + 1) * per_job


def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"required env var {var} is not set")
    return val


def load() -> Config:
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
    )
