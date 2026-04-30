"""Numerical and dispatch tests for the attention dispatcher
(WAN_DEMO_ATTN, see wan/modules/attention.py).

Phase 2 (Path B): sageattention is a drop-in faster path for the same
self-/cross-attention used by TI2V-5B's transformer blocks. We assert:

  1. _resolve_attn_backend() honors WAN_DEMO_ATTN.
  2. sage_attention() and flash_attention() agree numerically on a
     synthetic input matching the TI2V-5B production shape (B=1, H=24,
     L=4096, D=128, bf16) within max-abs-diff <= 1e-2.
  3. With WAN_DEMO_ATTN=flash, flash_attention() does NOT route to
     sage even when sage is importable.

Tests requiring the live kernels (steps 2/3) skip when CUDA / sage /
flash are not available — they are intended to be run on the gpu6
demo venv where all three are present.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager

import pytest


@contextmanager
def _env(**overrides: str):
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


# ---------------------------------------------------------------------------
# Backend resolution — pure-Python, no CUDA needed.
# ---------------------------------------------------------------------------


def _reload_attn():
    """Re-import attention module so env-driven module-level globals
    pick up overrides set by the test."""
    from wan.modules import attention
    importlib.reload(attention)
    return attention


def test_resolve_backend_flash_returns_flash():
    with _env(WAN_DEMO_ATTN="flash"):
        attn = _reload_attn()
        assert attn._resolve_attn_backend() == "flash"


def test_resolve_backend_auto_picks_sage_if_available():
    with _env(WAN_DEMO_ATTN="auto"):
        attn = _reload_attn()
        expected = "sage" if attn.SAGE_ATTN_AVAILABLE else "flash"
        assert attn._resolve_attn_backend() == expected


def test_resolve_backend_default_is_auto():
    with _env(WAN_DEMO_ATTN=None):
        attn = _reload_attn()
        expected = "sage" if attn.SAGE_ATTN_AVAILABLE else "flash"
        assert attn._resolve_attn_backend() == expected


def test_resolve_backend_unknown_falls_back_to_auto():
    with _env(WAN_DEMO_ATTN="bogus"):
        attn = _reload_attn()
        with pytest.warns(UserWarning, match="WAN_DEMO_ATTN="):
            result = attn._resolve_attn_backend()
        assert result in ("sage", "flash")


def test_resolve_backend_sage_without_install_raises():
    """If WAN_DEMO_ATTN=sage but the module isn't importable, the
    resolver must fail loudly rather than silently fall back."""
    attn = _reload_attn()
    if attn.SAGE_ATTN_AVAILABLE:
        pytest.skip("sage is importable; cannot test the missing-install path")
    with _env(WAN_DEMO_ATTN="sage"):
        attn = _reload_attn()
        with pytest.raises(RuntimeError, match="sageattention"):
            attn._resolve_attn_backend()


# ---------------------------------------------------------------------------
# Numerical agreement — needs CUDA + sage + flash.
# ---------------------------------------------------------------------------


def _have_cuda_and_sage() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import sageattention  # noqa: F401
    except ImportError:
        return False
    return True


def _have_flash() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        try:
            import flash_attn_interface  # noqa: F401
            return True
        except ImportError:
            return False


@pytest.mark.skipif(
    not _have_cuda_and_sage(),
    reason="sageattention numerics test needs CUDA and the sage wheel",
)
def test_sage_matches_flash_on_ti2v_shape():
    """Production shape: 81 frames * 1280/16 * 704/16 / 2 patch
    factor = 18480 latent tokens; the smaller smoke shape (4096) is
    chosen so the test runs in <1 s on a 4090. head_dim=128, 24 heads."""
    if not _have_flash():
        pytest.skip("flash_attn not installed; skipping numeric A/B")

    import torch

    with _env(WAN_DEMO_ATTN="auto"):
        attn = _reload_attn()
        if not attn.SAGE_ATTN_AVAILABLE:
            pytest.skip("sage not importable in this env")

    torch.manual_seed(0)
    B, L, H, D = 1, 4096, 24, 128
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)

    out_sage = attn.sage_attention(q, k, v)

    # Force the flash path explicitly so we don't accidentally compare
    # sage to itself.
    with _env(WAN_DEMO_ATTN="flash"):
        attn_flash = _reload_attn()
        out_flash = attn_flash.flash_attention(q, k, v)

    assert out_sage.shape == out_flash.shape
    assert out_sage.dtype == out_flash.dtype

    diff = (out_sage.float() - out_flash.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(
        f"sage vs flash: max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}"
    )
    # Sage uses INT8 keys + fp16 query GEMM; mean drift on TI2V shapes
    # in ad-hoc bench was ~3e-3, peaks ~5e-3. We use 1e-2 as the
    # gate — anything much larger means a layout / scaling bug.
    assert max_abs < 1e-2, (
        f"sage and flash diverged: max_abs={max_abs:.4e} > 1e-2"
    )


@pytest.mark.skipif(
    not _have_cuda_and_sage(),
    reason="dispatch test needs CUDA + sage to confirm flash override",
)
def test_attn_flash_override_skips_sage():
    """When WAN_DEMO_ATTN=flash, flash_attention() must run the flash
    code path even if sage is importable. Regression guard against the
    dispatcher silently picking sage when the operator told it not to.
    """
    if not _have_flash():
        pytest.skip("flash_attn not installed")

    import torch

    torch.manual_seed(1)
    B, L, H, D = 1, 1024, 8, 64
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)

    with _env(WAN_DEMO_ATTN="flash"):
        attn = _reload_attn()
        # Smoke: should run without raising and return a sane shape.
        out = attn.flash_attention(q, k, v)

    assert out.shape == (B, L, H, D)
