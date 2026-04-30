# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn as _sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'sage_attention',
    'SAGE_ATTN_AVAILABLE',
]


# Tracks whether we've already emitted the B>1 fallback warning, so the
# log doesn't get spammed once per attention call.
_SAGE_BATCH_WARNED = False


def _resolve_attn_backend() -> str:
    """Pick the attention backend per WAN_DEMO_ATTN.

    Values:
        "auto" (default) — sage if importable, else flash.
        "sage"           — sage; raise if not importable.
        "flash"          — flash (FA3 if available, else FA2/SDPA).
    """
    mode = os.environ.get("WAN_DEMO_ATTN", "auto").lower()
    if mode not in ("auto", "sage", "flash"):
        warnings.warn(
            f"WAN_DEMO_ATTN={mode!r} is not one of "
            "'auto'|'sage'|'flash'; falling back to 'auto'."
        )
        mode = "auto"

    if mode == "sage":
        if not SAGE_ATTN_AVAILABLE:
            raise RuntimeError(
                "WAN_DEMO_ATTN=sage but sageattention is not importable. "
                "pip install sageattention or set WAN_DEMO_ATTN=flash."
            )
        return "sage"
    if mode == "flash":
        return "flash"
    # auto
    return "sage" if SAGE_ATTN_AVAILABLE else "flash"


def sage_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    softmax_scale=None,
    q_scale=None,
    dtype=torch.bfloat16,
):
    """SageAttention path, callable in place of flash_attention.

    Inputs use the same `[B, L, n_heads, head_dim]` layout as
    flash_attention. We B=1-fast-path: convert to sage's expected
    `[B, H, L, D]` with a single transpose, no varlen packing. For
    B>1 we fall back to flash_attention with a one-time warning,
    since the demo server only ever runs B=1.

    softmax_scale / q_scale follow flash_attention semantics.
    """
    global _SAGE_BATCH_WARNED
    assert SAGE_ATTN_AVAILABLE, "sageattention not importable"
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)

    # Sage handles head_dim up to 128 on Ampere/Hopper; head_dim 128
    # is exactly what TI2V-5B uses (24 heads * 128 = 3072), so we're
    # in the supported range. Any caller sending head_dim>128 routes
    # back to flash.
    if q.size(-1) > 128 or k.size(-1) > 128:
        return flash_attention(
            q, k, v, q_lens=q_lens, k_lens=k_lens,
            softmax_scale=softmax_scale, q_scale=q_scale,
            dtype=dtype,
        )

    # B>1 in this codebase only happens if someone wires it up; the
    # demo server is hard B=1. Fall back rather than implement varlen
    # packing for sage, which doesn't have a varlen API in 1.0.6.
    if b > 1 or (q_lens is not None and (q_lens != lq).any()) \
       or (k_lens is not None and (k_lens != lk).any()):
        if not _SAGE_BATCH_WARNED:
            warnings.warn(
                "sage_attention: falling back to flash for B>1 or "
                "ragged q_lens/k_lens (one-time warning)."
            )
            _SAGE_BATCH_WARNED = True
        return flash_attention(
            q, k, v, q_lens=q_lens, k_lens=k_lens,
            softmax_scale=softmax_scale, q_scale=q_scale,
            dtype=dtype,
        )

    # cast to half precision — sage only operates on fp16/bf16
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)
    # match dtypes within the call (sage requires q/k/v to share dtype)
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # Layout swap: [B, L, H, D] -> [B, H, L, D]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    out = _sageattn(q, k, v, sm_scale=softmax_scale)
    # Back to [B, L, H, D]
    out = out.transpose(1, 2).contiguous()
    return out.type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.

    Dispatch: when WAN_DEMO_ATTN selects 'sage' (or 'auto' and sage is
    importable), and the call is shape-compatible (no causal mask, no
    window mask, no dropout), routes through sage_attention. Otherwise
    runs FA3/FA2 below. The window/causal/dropout features are only
    used by hypothetical callers — Wan2.2's TI2V-5B model.py uses
    plain self-/cross-attention with default flags.
    """
    backend = _resolve_attn_backend()
    sage_compatible = (
        backend == "sage"
        and not causal
        and window_size == (-1, -1)
        and dropout_p == 0.0
        and not deterministic
    )
    if sage_compatible:
        return sage_attention(
            q, k, v,
            q_lens=q_lens, k_lens=k_lens,
            softmax_scale=softmax_scale, q_scale=q_scale,
            dtype=dtype,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
