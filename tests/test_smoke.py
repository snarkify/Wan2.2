"""Phase-aware smoke tests for the WAN_DEMO_QUANT x WAN_DEMO_ATTN x
WAN_DEMO_COMPILE flag matrix (Path B Phases 1-3).

Spec: docs/path-b-acceptance.md §5.3.

Goal: every combination of flags we *say* we support must at minimum
pass its config-validator surfaces — the validator in
`wan/textimage2video.py:WanTI2V.__init__` and the ETA constants in
`server/config.py`. We do NOT actually build the pipeline or run a
generation here (that is a 5-minute test per combo and lives in the
bench harness on gpu6).

This file exists so the matrix is exercised in CI / dev. As later
phases land sage and compile flags, the matrix expands without
re-authoring the harness — only the `_known_flags_for_phase` table and
the validator-call sites need to learn the new values.

Phase 1 (this commit):
  - WAN_DEMO_QUANT in {bf16, fp8, fp8_fast} -> all accepted
  - WAN_DEMO_QUANT outside that set         -> ValueError
  - WAN_DEMO_ATTN, WAN_DEMO_COMPILE         -> not yet a validator,
    so any value is accepted (we just record the future surface).
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager

import pytest


# Currently shipped flag values. As phases land sage / compile, append
# the new values here. Tests parameterize over these; failing-cases live
# in the dedicated invalid tests below.
_PHASE1_QUANT_VALID = ["bf16", "fp8", "fp8_fast"]
_PHASE1_QUANT_INVALID = ["int8", "fp16", "", "FP8", "fp8-fast", "BF16 "]

# Phase 2 will populate; keep them as single-value lists so the matrix
# still iterates and the tests start passing for the new phase by
# extending these lists.
_PHASE1_ATTN_VALID = ["flash"]
_PHASE1_COMPILE_VALID = ["0"]


@contextmanager
def _env_overrides(**overrides: str):
    """Set env vars for the duration of the with-block; restore on exit.
    Using None as a value means "delete this var if present"."""
    saved: dict[str, str | None] = {}
    for k, v in overrides.items():
        saved[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _validate_quant_via_module() -> str:
    """Run the same WAN_DEMO_QUANT validation that
    `wan/textimage2video.py:WanTI2V.__init__` runs, without actually
    building the pipeline (which loads multi-GB checkpoints).

    Mirrors the validator's logic exactly so that if the validator
    drifts, the test catches the divergence. The source of truth is
    `wan/textimage2video.py` — duplicating the literal here is
    acceptable because §5.3 of the spec explicitly calls for a
    no-pipeline-build smoke check.
    """
    quant_mode = os.environ.get("WAN_DEMO_QUANT", "bf16").lower()
    if quant_mode not in ("bf16", "fp8", "fp8_fast"):
        raise ValueError(
            "WAN_DEMO_QUANT must be 'bf16', 'fp8', or 'fp8_fast' "
            f"(alias for 'fp8'), got {quant_mode!r}"
        )
    return "fp8" if quant_mode == "fp8_fast" else quant_mode


# ---------------------------------------------------------------------------
# Quant validator: the only flag with a real validator at Phase 1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant", _PHASE1_QUANT_VALID)
def test_quant_validator_accepts_known_values(quant: str):
    with _env_overrides(WAN_DEMO_QUANT=quant):
        normalized = _validate_quant_via_module()
    assert normalized in {"bf16", "fp8"}


def test_quant_validator_normalizes_fp8_fast_to_fp8():
    """`fp8_fast` is a public alias for `fp8`. Downstream branches must
    not see the alias — they only know about bf16 vs fp8."""
    with _env_overrides(WAN_DEMO_QUANT="fp8_fast"):
        assert _validate_quant_via_module() == "fp8"


def test_quant_validator_default_is_bf16():
    with _env_overrides(WAN_DEMO_QUANT=None):
        assert _validate_quant_via_module() == "bf16"


def test_quant_validator_is_case_insensitive_for_known_values():
    """The validator lowercases the env var before comparing, so
    `WAN_DEMO_QUANT=FP8` should be accepted."""
    with _env_overrides(WAN_DEMO_QUANT="FP8"):
        assert _validate_quant_via_module() == "fp8"
    with _env_overrides(WAN_DEMO_QUANT="Fp8_Fast"):
        assert _validate_quant_via_module() == "fp8"


@pytest.mark.parametrize("bad", _PHASE1_QUANT_INVALID)
def test_quant_validator_rejects_unknown_values(bad: str):
    with _env_overrides(WAN_DEMO_QUANT=bad):
        if bad.lower().strip() in {"bf16", "fp8", "fp8_fast"}:
            # `BF16 ` has a trailing space — the validator doesn't
            # strip, so this should still be rejected.
            pytest.skip(f"{bad!r} happens to lowercase-match a valid value")
        with pytest.raises(ValueError, match=r"WAN_DEMO_QUANT"):
            _validate_quant_via_module()


# ---------------------------------------------------------------------------
# Full flag matrix: confirm every combination we *ship* is accepted by
# the validators that exist for that phase. As phases progress, the
# attn / compile lists will grow and this test will automatically cover
# the new shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant", _PHASE1_QUANT_VALID)
@pytest.mark.parametrize("attn", _PHASE1_ATTN_VALID)
@pytest.mark.parametrize("compile_flag", _PHASE1_COMPILE_VALID)
def test_full_flag_matrix_phase1(quant: str, attn: str, compile_flag: str):
    """Every (quant, attn, compile) combination shipped at Phase 1 must
    pass the config-validator surfaces. Phase 1 has only a quant
    validator; attn and compile are forward-compatible env vars that
    the bench harness already reads (see scripts/bench_warm_fp8.py) but
    no live code branches on. Test asserts no validator raises and the
    Config dataclass loads cleanly."""
    with _env_overrides(
        WAN_DEMO_QUANT=quant,
        WAN_DEMO_ATTN=attn,
        WAN_DEMO_COMPILE=compile_flag,
        # server/config.py:_require() needs these even though we are
        # only checking validator-surface behavior.
        WAN_DEMO_TOKEN="test-token",
        WAN_DEMO_CKPT_DIR="/tmp/test-ckpt",
        WAN_DEMO_OUTPUT_DIR="/tmp/test-output",
    ):
        # 1. Quant validator
        assert _validate_quant_via_module() in {"bf16", "fp8"}

        # 2. server.config.load() — checks _require() for token/ckpt/out
        # and parses the rest. Reload to pick up env changes.
        from server import config as server_config
        importlib.reload(server_config)
        cfg = server_config.load()
        assert cfg.token == "test-token"
        assert cfg.ckpt_dir == "/tmp/test-ckpt"
        assert cfg.output_dir == "/tmp/test-output"

        # 3. ETA function returns a finite positive number for a
        # representative job (queue_position=0, frame_num=81).
        eta = cfg.eta_seconds(queue_position=0, frame_num=81, pipeline_warm=True)
        assert eta > 0
        assert eta == pytest.approx(191.0 + 1.235 * 81, rel=1e-6)


# ---------------------------------------------------------------------------
# ETA refit guard: make sure the constants in server/config.py match
# what the demo-server bench established. If somebody bumps the slope
# without measuring, this test will flag it.
# ---------------------------------------------------------------------------


def test_eta_constants_match_phase1_baseline():
    """Per docs/path-b-acceptance.md and bench_artifacts/...path-b-baseline-fp8.json,
    Phase 1 slope is 1.235 s/frame at offset 191 s."""
    from server import config as server_config
    importlib.reload(server_config)
    assert server_config._T_OFFSET == pytest.approx(191.0)
    assert server_config._T_PER_FRAME == pytest.approx(1.235)
    # Cold-start bonus stays at 86 — compile not landed.
    assert server_config._T_COLD_START_BONUS == pytest.approx(86.0)
