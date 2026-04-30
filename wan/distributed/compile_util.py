"""torch.compile glue for Wan DiT pipelines.

A single helper, `compile_dit`, walks a pipeline object and replaces
each DiT-style attribute with its torch.compile()'d counterpart. The
standalone CLI (`generate.py`) and the demo server (`server/worker.py`)
both call this — duplication between the two entrypoints would be a
guaranteed source of divergence as compile modes / candidate attribute
names evolve.

Why a function rather than a class wrapper:
  - torch.compile returns a wrapped module that quacks like the
    original. Re-binding `setattr(pipeline, attr, compiled)` keeps the
    pipeline shape unchanged; downstream code (`pipeline.generate(...)`,
    `pipeline.model.parameters()`, etc.) continues to work.
  - The compile happens once on the first forward pass after this call
    returns. We do NOT eagerly trace here — the pipeline's offload
    machinery moves the model on/off GPU, and tracing in the wrong
    location would just thrash.

Why we toggle `wan.profiling._compile_active`:
  - Several profiling spans in `wan/profiling/` use CUDA events to time
    sub-step regions. CUDA events inside a compiled graph either get
    folded out of the captured graph (silent zero timings) or trigger
    "graph break under stream-capture" errors. The flag tells the
    profiling layer to fall back to wall-clock timing for those spans
    when compile is active.

Candidate attribute names:
  - `model`           — single-DiT pipelines (TI2V-5B, our demo path)
  - `noise_model`     — used by some Wan variants
  - `low_noise_model`, `high_noise_model` — two-DiT MoE-style pipelines
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch


# Attribute names we recognize as DiT slots on a Wan pipeline. Order is
# stable across the call so logs / diagnostics are reproducible.
_DIT_ATTR_CANDIDATES: tuple[str, ...] = (
    "model",
    "noise_model",
    "low_noise_model",
    "high_noise_model",
)


def compile_dit(
    pipeline: Any,
    *,
    mode: str = "default",
    enabled: bool = True,
) -> list[str]:
    """torch.compile every DiT attribute on `pipeline`, in place.

    Args:
        pipeline: any object with one or more of the DiT-shaped
            attributes listed in `_DIT_ATTR_CANDIDATES`. Usually a
            `WanTI2V` instance.
        mode: torch.compile `mode` kwarg. "default" is the safest with
            fp8 `_scaled_mm` ops (less aggressive fusion, lower chance
            of graph breaks at the custom Linear forward). "reduce-
            overhead" is faster on short sequences but uses more memory
            (CUDA graph capture); only flip to it when memory headroom
            is confirmed.
        enabled: convenience flag so callers can route the compile
            toggle through this function without conditional wrapping.
            When False, we no-op and return [] — useful for tests and
            the `WAN_DEMO_COMPILE=0` path.

    Returns:
        List of attribute names that were actually compiled (subset of
        `_DIT_ATTR_CANDIDATES`). Empty list if nothing matched or if
        `enabled=False`. Caller can use this to gate downstream
        behavior (e.g. cold-start ETA bonus).

    Side effects:
        - `pipeline.<attr>` is replaced with `torch.compile(<attr>)` for
          each matched attribute.
        - `wan.profiling._compile_active` is set to True iff at least
          one attribute was compiled. Never set back to False here —
          compile is a one-way switch within a process.
        - One INFO log line per compiled attribute and one summary line.
    """
    if not enabled:
        return []

    compiled_attrs: list[str] = []
    for attr in _DIT_ATTR_CANDIDATES:
        if not hasattr(pipeline, attr):
            continue
        m = getattr(pipeline, attr)
        if m is None:
            continue
        compiled = torch.compile(m, mode=mode)
        setattr(pipeline, attr, compiled)
        compiled_attrs.append(attr)
        logging.info(f"[WAN_DEMO_COMPILE] compiled attr={attr} mode={mode}")

    if compiled_attrs:
        # Tell the profiling framework to skip CUDA events for spans
        # that wrap compiled forward calls. Importing lazily because
        # this module is also imported from `generate.py` before
        # profiling has been initialized.
        import wan.profiling as _prof
        _prof._compile_active = True
        logging.info(
            f"[WAN_DEMO_COMPILE] applied to {len(compiled_attrs)} attr(s): "
            f"{compiled_attrs} mode={mode}"
        )
    else:
        logging.info(
            "[WAN_DEMO_COMPILE] no DiT attributes found on pipeline; "
            "compile no-op"
        )

    return compiled_attrs


def resolve_compile_mode(env_value: Optional[str]) -> str:
    """Map an env-var string to a torch.compile mode.

    Accepts either a recognized mode literal (`default`, `reduce-overhead`,
    `max-autotune`) or `None`/empty → `default`. Raises ValueError on
    anything else so misconfiguration surfaces at startup instead of
    silently using an unintended mode.
    """
    if env_value is None or env_value.strip() == "":
        return "default"
    v = env_value.strip().lower()
    if v not in ("default", "reduce-overhead", "max-autotune"):
        raise ValueError(
            f"WAN_DEMO_COMPILE_MODE must be one of "
            f"'default', 'reduce-overhead', 'max-autotune', got {env_value!r}"
        )
    return v
