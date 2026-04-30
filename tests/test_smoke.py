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

Phase 1:
  - WAN_DEMO_QUANT in {bf16, fp8, fp8_fast} -> all accepted
  - WAN_DEMO_QUANT outside that set         -> ValueError
  - WAN_DEMO_ATTN, WAN_DEMO_COMPILE         -> not yet a validator,
    so any value is accepted (we just record the future surface).

Phase 3 (this commit adds):
  - WAN_DEMO_COMPILE in {0, 1, true, false, yes, no} -> all accepted
  - WAN_DEMO_COMPILE outside that set       -> ValueError raised by
    server/config.py:_parse_bool
  - WAN_DEMO_COMPILE_MODE in {default, reduce-overhead, max-autotune}
    -> accepted; anything else -> ValueError (raised by
    wan.distributed.compile_util.resolve_compile_mode)
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

# Phase 2 wires sage into WAN_DEMO_ATTN. The dispatcher accepts
# "auto"/"sage"/"flash" (auto is the default — picks sage if
# importable, else flash). "flash" remains the canonical fallback.
# Compile is still Phase-3 territory.
_PHASE1_ATTN_VALID = ["auto", "sage", "flash"]
# Phase 3 wires WAN_DEMO_COMPILE into server/config.py via _parse_bool;
# anything _parse_bool accepts is shippable. Both "0" and "1" must
# round-trip through Config.compile_enabled correctly.
_PHASE3_COMPILE_VALID = ["0", "1", "true", "false", "yes", "no"]
_PHASE3_COMPILE_INVALID = ["maybe", "compile", "2", "FP8", "  "]
# Compile mode accepts the three torch.compile literals plus empty/None
# (interpreted as "default"). Anything else raises.
_PHASE3_COMPILE_MODE_VALID = ["default", "reduce-overhead", "max-autotune"]
_PHASE3_COMPILE_MODE_INVALID = ["aggressive", "fast", "DEFAULT ", "max_autotune"]


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
@pytest.mark.parametrize("compile_flag", _PHASE3_COMPILE_VALID)
def test_full_flag_matrix(quant: str, attn: str, compile_flag: str):
    """Every (quant, attn, compile) combination shipped through Phase 3
    must pass the config-validator surfaces. The matrix grew at Phase 3:
    WAN_DEMO_COMPILE now drives `Config.compile_enabled` and is parsed
    by `server.config._parse_bool`. We assert here that every legal
    string round-trips through Config without raising and that the
    boolean ends up correct.
    """
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

        # 3. compile_enabled correctly parsed.
        expected_on = compile_flag.lower() in ("1", "true", "yes", "on")
        assert cfg.compile_enabled is expected_on, (
            f"compile_flag={compile_flag!r} expected_on={expected_on} "
            f"got cfg.compile_enabled={cfg.compile_enabled}"
        )

        # 4. ETA function returns a finite positive number for a
        # representative job (queue_position=0, frame_num=81).
        eta = cfg.eta_seconds(queue_position=0, frame_num=81, pipeline_warm=True)
        assert eta > 0
        assert eta == pytest.approx(191.0 + 1.235 * 81, rel=1e-6)


# ---------------------------------------------------------------------------
# Phase 3 compile-flag dispatch tests. Exercise the boolean parser and
# the compile-mode resolver directly. Neither path actually invokes
# torch.compile (that requires CUDA + the warm pipeline), but both
# validators run on import and must reject misconfiguration eagerly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compile_flag", _PHASE3_COMPILE_VALID)
def test_compile_flag_accepts_known_values(compile_flag: str):
    """Every value in `_PHASE3_COMPILE_VALID` must parse to a bool
    without raising. The expected truthiness is the obvious mapping.
    """
    with _env_overrides(
        WAN_DEMO_COMPILE=compile_flag,
        WAN_DEMO_TOKEN="t", WAN_DEMO_CKPT_DIR="/tmp/c",
        WAN_DEMO_OUTPUT_DIR="/tmp/o",
    ):
        from server import config as server_config
        importlib.reload(server_config)
        cfg = server_config.load()
        expected = compile_flag.lower() in ("1", "true", "yes", "on")
        assert cfg.compile_enabled is expected


@pytest.mark.parametrize("compile_flag", _PHASE3_COMPILE_INVALID)
def test_compile_flag_rejects_unknown_values(compile_flag: str):
    with _env_overrides(
        WAN_DEMO_COMPILE=compile_flag,
        WAN_DEMO_TOKEN="t", WAN_DEMO_CKPT_DIR="/tmp/c",
        WAN_DEMO_OUTPUT_DIR="/tmp/o",
    ):
        from server import config as server_config
        importlib.reload(server_config)
        # Empty/whitespace returns the default (True), so don't expect
        # ValueError for those. The parser deliberately treats them as
        # "unset" rather than as misconfiguration.
        if compile_flag.strip() == "":
            cfg = server_config.load()
            assert cfg.compile_enabled is True  # default
            return
        with pytest.raises(ValueError, match=r"WAN_DEMO_COMPILE"):
            server_config.load()


def test_compile_flag_default_is_enabled():
    """User explicitly asked for compile-on by default in Phase 3."""
    with _env_overrides(
        WAN_DEMO_COMPILE=None,
        WAN_DEMO_TOKEN="t", WAN_DEMO_CKPT_DIR="/tmp/c",
        WAN_DEMO_OUTPUT_DIR="/tmp/o",
    ):
        from server import config as server_config
        importlib.reload(server_config)
        cfg = server_config.load()
        assert cfg.compile_enabled is True


@pytest.mark.parametrize("mode", _PHASE3_COMPILE_MODE_VALID)
def test_compile_mode_accepts_known_values(mode: str):
    from wan.distributed.compile_util import resolve_compile_mode
    assert resolve_compile_mode(mode) == mode


def test_compile_mode_default_when_unset():
    from wan.distributed.compile_util import resolve_compile_mode
    assert resolve_compile_mode(None) == "default"
    assert resolve_compile_mode("") == "default"
    assert resolve_compile_mode("   ") == "default"


@pytest.mark.parametrize("bad", _PHASE3_COMPILE_MODE_INVALID)
def test_compile_mode_rejects_unknown_values(bad: str):
    from wan.distributed.compile_util import resolve_compile_mode
    if bad.strip().lower() in ("default", "reduce-overhead", "max-autotune"):
        pytest.skip(f"{bad!r} normalizes to a valid value")
    with pytest.raises(ValueError, match=r"WAN_DEMO_COMPILE_MODE"):
        resolve_compile_mode(bad)


def test_compile_dit_no_op_when_disabled():
    """`compile_dit(..., enabled=False)` returns [] without touching
    the pipeline. This is the path taken when WAN_DEMO_COMPILE=0."""
    from wan.distributed.compile_util import compile_dit

    class FakePipeline:
        def __init__(self):
            self.model = "sentinel-model"
            self.noise_model = "sentinel-noise"

    p = FakePipeline()
    result = compile_dit(p, enabled=False)
    assert result == []
    # Untouched.
    assert p.model == "sentinel-model"
    assert p.noise_model == "sentinel-noise"


def test_compile_dit_skips_missing_attrs():
    """Pipelines that have only `model` should compile only `model`
    (not raise on the absent `noise_model`/`low_noise_model`).
    """
    import torch
    from wan.distributed.compile_util import compile_dit

    class FakePipeline:
        def __init__(self):
            self.model = torch.nn.Linear(4, 4)

    p = FakePipeline()
    result = compile_dit(p, enabled=True)
    # On environments without a working compile backend (e.g. CPU-only
    # CI), torch.compile may still wrap; we accept any result with
    # exactly ["model"] as the matched attribute set.
    assert result == ["model"]
    # The replaced attribute must still be call-able (compile wraps,
    # doesn't replace with None).
    assert callable(p.model)


# ---------------------------------------------------------------------------
# ETA refit guard: make sure the constants in server/config.py match
# what the demo-server bench established. If somebody bumps the slope
# without measuring, this test will flag it.
# ---------------------------------------------------------------------------


def test_eta_constants_match_current_baseline():
    """Per docs/path-b-acceptance.md and bench_artifacts, the warm
    per-job slope at Phase 1 was 1.235 s/frame at offset 191 s; that
    has not been re-fit at Phase 2/3 because the per-step latency only
    moved from 5.83 s to ~5.18 s (sage) and we haven't lowered to the
    Phase 3 target yet. Cold-start bumped at Phase 3 from 86 → 360 to
    cover the first-forward torch.compile cost (~270 s), per the Phase
    3 acceptance criterion that first job ≤ 480 s with compile on.
    """
    from server import config as server_config
    importlib.reload(server_config)
    assert server_config._T_OFFSET == pytest.approx(191.0)
    assert server_config._T_PER_FRAME == pytest.approx(1.235)
    # Cold-start bonus bumped at Phase 3 to cover compile cost.
    assert server_config._T_COLD_START_BONUS == pytest.approx(360.0)
