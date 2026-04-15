# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Profiling configuration — lazy singleton from env vars + argparse."""

import os
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool
    output_dir: str
    run_id: str
    rank: int
    local_rank: int
    trace_enabled: bool
    torch_profiler_enabled: bool
    torch_profiler_start: int
    torch_profiler_end: int
    sync_before_timing: bool
    model_detail: bool
    flush_interval: int


_CONFIG: ProfileConfig | None = None


def _load_config() -> ProfileConfig:
    output_dir = os.environ.get("WAN_PROFILE_DIR", "")
    enabled = bool(output_dir)
    return ProfileConfig(
        enabled=enabled,
        output_dir=output_dir,
        run_id=os.environ.get("WAN_PROFILE_RUN_ID", uuid.uuid4().hex[:12]),
        rank=int(os.environ.get("RANK", 0)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        trace_enabled=enabled
        and os.environ.get("WAN_PROFILE_TRACE", "1") == "1",
        torch_profiler_enabled=enabled
        and os.environ.get("WAN_PROFILE_TORCH", "0") == "1",
        torch_profiler_start=int(
            os.environ.get("WAN_PROFILE_TORCH_START", "2")
        ),
        torch_profiler_end=int(os.environ.get("WAN_PROFILE_TORCH_END", "5")),
        sync_before_timing=enabled
        and os.environ.get("WAN_PROFILE_SYNC", "1") == "1",
        model_detail=enabled
        and os.environ.get("WAN_PROFILE_MODEL_DETAIL", "0") == "1",
        flush_interval=int(
            os.environ.get("WAN_PROFILE_FLUSH_INTERVAL", "500")
        ),
    )


def get_config() -> ProfileConfig:
    """Lazy config — deferred until first use so RANK is available."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def init_from_args(profile_dir: str | None) -> None:
    """Called from generate.py after argparse. Sets env var before lazy init."""
    global _CONFIG
    if profile_dir:
        os.environ["WAN_PROFILE_DIR"] = profile_dir
        _CONFIG = None  # Force re-init on next get_config()
