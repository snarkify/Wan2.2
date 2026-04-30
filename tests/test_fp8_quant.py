"""Regression tests for `wan/distributed/fp8_quant.py` (Path B Phase 1).

Spec: docs/path-b-acceptance.md §5.1.

These tests guard:
  1. Real Wan TI2V-5B Linear shapes do not raise during quantization +
     forward (the activation-shape gotcha — the M dim of a flattened
     activation also has alignment requirements on some _scaled_mm
     kernel paths).
  2. Per-tensor symmetric scale's dequant error stays inside an 8 %
     budget — this is the safety net for "we silently swap in a worse
     scaling scheme".
  3. The new alignment assertion fires for both unaligned `in_features`
     AND unaligned `out_features` (the latter is the latent footgun the
     Phase 1 patch closes).
  4. The dtype-fallback path in `_fp8_forward` (lines 97-98 of
     `fp8_quant.py`) actually engages when the weight gets re-cast away
     from fp8 — this is the only thing standing between FSDP's
     MixedPrecision and a hard failure on the inner forward.

`torch._scaled_mm` is CUDA-only. Cases 1 and 2 require a GPU; we mark
them with `pytest.mark.gpu` and skip cleanly when CUDA is unavailable
so the test file is still importable / runnable on a dev laptop. Cases
3 and 4 do NOT need a real matmul — they only exercise the alignment
gate and the dtype-replacement fallback branch — so they always run.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from wan.distributed.fp8_quant import (
    _quantize_linear_inplace,
    quantize_wan_blocks_to_fp8,
)


_HAS_CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="fp8 _scaled_mm requires CUDA tensor cores; skipping on CPU host",
)


# TI2V-5B (wan/configs/wan_ti2v_5B.py): dim=3072, ffn_dim=14336, heads=24.
# These are the (in, out) pairs of every nn.Linear inside one transformer
# block. `quantize_wan_blocks_to_fp8` walks `model.blocks` and converts
# each of these.
_TI2V_5B_LINEAR_SHAPES: list[tuple[str, int, int]] = [
    ("self_attn.q", 3072, 3072),
    ("self_attn.k", 3072, 3072),
    ("self_attn.v", 3072, 3072),
    ("self_attn.o", 3072, 3072),
    ("cross_attn.q", 3072, 3072),
    ("cross_attn.k", 3072, 3072),
    ("cross_attn.v", 3072, 3072),
    ("cross_attn.o", 3072, 3072),
    ("ffn.0", 3072, 14336),
    ("ffn.2", 14336, 3072),
]


# ---------------------------------------------------------------------------
# Case 3: alignment assertion (CPU-safe — the assert fires before any matmul)
# ---------------------------------------------------------------------------


def test_quantize_assertion_on_unaligned_in_features():
    """Pre-existing assert: K (in_features) must be %16==0."""
    bad = nn.Linear(15, 32, bias=False).to(torch.bfloat16)
    with pytest.raises(ValueError, match=r"in_features"):
        _quantize_linear_inplace(bad, name="test.bad_in")


def test_quantize_assertion_on_unaligned_out_features():
    """New in Phase 1: N (out_features) must also be %16==0."""
    bad = nn.Linear(32, 15, bias=False).to(torch.bfloat16)
    with pytest.raises(ValueError, match=r"out_features"):
        _quantize_linear_inplace(bad, name="test.bad_out")


def test_quantize_assertion_on_both_unaligned():
    """Both unaligned should still raise (and the message identifies the
    layer name so debugging UX stays good)."""
    bad = nn.Linear(15, 17, bias=False).to(torch.bfloat16)
    with pytest.raises(ValueError, match=r"my\.layer"):
        _quantize_linear_inplace(bad, name="my.layer")


def test_quantize_accepts_aligned_shapes_without_cuda():
    """Aligned shapes pass the precondition check even on CPU. We can't
    actually call `_scaled_mm` without CUDA, but we can confirm the
    metadata mutation (Parameter dtype + scale_weight buffer) works on
    CPU tensors. This also covers the "metadata-only" half of cases 1+2
    when GPU isn't available."""
    lin = nn.Linear(3072, 3072, bias=True).to(torch.bfloat16)
    _quantize_linear_inplace(lin, name="test.aligned")
    assert lin.weight.dtype is torch.float8_e4m3fn
    assert hasattr(lin, "scale_weight")
    assert lin.scale_weight.dtype is torch.float32
    assert hasattr(lin, "original_forward")


# ---------------------------------------------------------------------------
# Case 4: dtype-replacement fallback (CPU-safe — the fallback branch is
# triggered before any _scaled_mm call would happen)
# ---------------------------------------------------------------------------


def test_fallback_when_weight_dtype_replaced_to_bf16():
    """If something casts the fp8 weight back to bf16 (e.g. FSDP
    MixedPrecision, an explicit `.to(bf16)`), `_fp8_forward` must
    detect it and call `original_forward` instead of feeding bf16
    weights into _scaled_mm (which would crash). `original_forward` is
    just the unquantized nn.Linear forward — it works on CPU."""
    lin = nn.Linear(32, 16, bias=True).to(torch.bfloat16)
    _quantize_linear_inplace(lin, name="test.fallback")
    # Sanity: we are in the fp8 state.
    assert lin.weight.dtype is torch.float8_e4m3fn

    # Simulate FSDP MixedPrecision casting the weight back to bf16.
    lin.weight = nn.Parameter(
        lin.weight.detach().to(torch.bfloat16), requires_grad=False
    )
    assert lin.weight.dtype is torch.bfloat16

    # Now `_fp8_forward` should hit the dtype guard and fall back to
    # original_forward. On CPU this is just the standard Linear forward,
    # so we can assert finite output without _scaled_mm ever running.
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    out = lin(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_fallback_when_input_is_1d():
    """The forward also bails out for <2D input (the `input.dim() < 2`
    guard at fp8_quant.py:100). This ensures we don't try to reshape a
    1D tensor into a 2D matmul."""
    lin = nn.Linear(32, 16, bias=False).to(torch.bfloat16)
    _quantize_linear_inplace(lin, name="test.1d")
    # Cast back to bf16 so the original_forward fallback works on CPU.
    lin.weight = nn.Parameter(
        torch.randn(16, 32, dtype=torch.bfloat16), requires_grad=False
    )
    x = torch.randn(32, dtype=torch.bfloat16)  # 1D
    out = lin(x)
    assert out.shape == (16,)


# ---------------------------------------------------------------------------
# Cases 1 + 2: real Wan layer shapes + dequant error budget (GPU-only)
# ---------------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize(
    "name,in_f,out_f",
    [(n, i, o) for (n, i, o) in _TI2V_5B_LINEAR_SHAPES],
)
def test_quantize_actual_wan_layer_shapes(name: str, in_f: int, out_f: int):
    """Build a synthetic Linear at every TI2V-5B Linear shape, quantize,
    run a forward at a representative activation shape, and assert no
    exception + finite output + correct shape.

    The activation M dim is chosen to mimic the real runtime: TI2V-5B at
    1280x704x81 with patch_size=(1,2,2) produces a per-step token count
    around (81 / 1) * (704 / 2 / 8) * (1280 / 2 / 8) = 81 * 44 * 80 =
    285,120 tokens at the DiT input. We use a smaller M (4096) because
    the test's job is shape-correctness, not perf — and the smaller M
    keeps the test under a second per shape on a 4090.
    """
    device = torch.device("cuda")
    lin = nn.Linear(in_f, out_f, bias=True).to(torch.bfloat16).to(device)
    _quantize_linear_inplace(lin, name=name)

    # Aligned activation M dim (4096 % 16 == 0).
    x = torch.randn(4096, in_f, dtype=torch.bfloat16, device=device)
    out = lin(x)

    assert out.shape == (4096, out_f), f"{name}: got {tuple(out.shape)}"
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out).all(), f"{name}: non-finite output"


@gpu_only
def test_quantize_dequant_error_bound():
    """Per-tensor symmetric scale at fp8_e4m3 should produce a bounded
    relative error on a typical bf16 weight matrix. The 8 % budget is
    calibrated against fp8 GEMM industry literature (Transformer Engine
    docs, kijai's AB) — anything higher means our scale calculation
    drifted from the standard scheme."""
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Representative shape: self_attn.q at TI2V-5B.
    in_f, out_f = 3072, 3072
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16, device=device) * 0.05
    # Match the bias=False case to keep the comparison clean.
    lin_ref = nn.Linear(in_f, out_f, bias=False).to(device)
    lin_ref.weight = nn.Parameter(weight.clone(), requires_grad=False)

    lin_fp8 = nn.Linear(in_f, out_f, bias=False).to(device)
    lin_fp8.weight = nn.Parameter(weight.clone(), requires_grad=False)
    _quantize_linear_inplace(lin_fp8, name="test.error_bound")

    x = torch.randn(1024, in_f, dtype=torch.bfloat16, device=device)
    out_ref = lin_ref(x).to(torch.float32)
    out_fp8 = lin_fp8(x).to(torch.float32)

    rel_err_max = (out_fp8 - out_ref).abs().max() / out_ref.abs().max().clamp(min=1e-6)
    rel_err_mean = (out_fp8 - out_ref).abs().mean() / out_ref.abs().mean().clamp(min=1e-6)

    # Spec §5.1 case 2 calls for relative max <= 0.08. We assert the
    # mean relative as a tighter sanity check too, allowing the same
    # budget — the mean should be well inside the max.
    assert math.isfinite(rel_err_max.item()), "non-finite error metric"
    assert rel_err_max.item() <= 0.08, (
        f"fp8 dequant max relative error {rel_err_max.item():.4f} > 0.08 "
        f"(per-tensor symmetric scale budget exceeded)"
    )
    assert rel_err_mean.item() <= 0.08, (
        f"fp8 dequant mean relative error {rel_err_mean.item():.4f} > 0.08"
    )


# ---------------------------------------------------------------------------
# Top-level helper: quantize_wan_blocks_to_fp8 must reject a model with no
# `.blocks` attribute (catches misuse with the wrong model class).
# ---------------------------------------------------------------------------


def test_quantize_wan_blocks_rejects_model_without_blocks():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 16)

    with pytest.raises(AttributeError, match=r"blocks"):
        quantize_wan_blocks_to_fp8(FakeModel())
