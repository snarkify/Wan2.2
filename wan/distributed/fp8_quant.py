"""In-place fp8 (e4m3fn) weight quantization for WanModel transformer blocks.

This is the demo-server bench branch's fp8 path. Goal: get the same kind
of speedup ComfyUI's WanVideoWrapper sees from `fp8_e4m3fn_fast` — real
fp8 matmul via `torch._scaled_mm` on Ada (RTX 4090) tensor cores —
inside our FSDP+Ulysses parallelism.

Approach (mirrors kijai/ComfyUI-WanVideoWrapper/fp8_optimization.py):

1. After `model.to(bf16)` and BEFORE `shard_fn(model)`, walk the
   transformer blocks and:
     - cast each `nn.Linear` weight (in self_attn/cross_attn/ffn) to
       `torch.float8_e4m3fn` storage — per-tensor max-abs scale baked
       into a `scale_weight` buffer so we can recover bf16 matmul
       output.
     - replace `Linear.forward` with a closure that does
       `torch._scaled_mm` of (fp8 input, fp8 weight.t()) returning bf16.

2. We deliberately leave `patch_embedding` (Conv3d), `text_embedding`,
   `time_embedding`, `time_projection`, and the final `head` in bf16:
   they're either tiny or numerically sensitive and live outside
   `model.blocks`, so they are not auto-wrapped by FSDP either.

FSDP gotcha: `MixedPrecision(param_dtype=bf16)` would cast our fp8
weights back to bf16 in forward, killing the speedup. The caller must
disable MixedPrecision (or use param_dtype=None) when fp8 is enabled.
We pre-cast non-fp8 params to bf16 ourselves before sharding, so MP is
not needed for correctness either.

`_scaled_mm` requires K to be a multiple of 16. For TI2V-5B (dim=3072,
ffn_dim=14336) every Linear we touch has K in {3072, 14336}: both
divisible by 16. We assert this at conversion time.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

log = logging.getLogger("wan.fp8_quant")

# fp8_e4m3fn dynamic range is +/- 448. We scale weights so that the
# largest per-tensor abs value lands at 448 -- standard "tensorwise"
# scheme used by torchao / kijai / ComfyUI native fp8 path.
_FP8_MAX = 448.0


def _quantize_linear_inplace(linear: nn.Linear, name: str = "<unnamed>") -> None:
    """Cast `linear.weight` to fp8_e4m3fn in place + register `scale_weight`.

    After this call:
      - linear.weight is a Parameter with dtype torch.float8_e4m3fn
      - linear.scale_weight is a float32 scalar buffer s.t.
          dequant(linear.weight) ~= linear.weight.to(bf16) * scale_weight
      - linear.original_forward holds the original Linear.forward
      - linear.forward is replaced with the fp8 scaled_mm path

    `name` is used purely for diagnostics on alignment failure.
    """
    w = linear.weight.data  # bf16 at this point
    out_features, in_features = w.shape[0], w.shape[1]
    # `torch._scaled_mm` requires both K (in_features) and N (out_features)
    # to be multiples of 16. The K check has been here from day one; the N
    # check was a latent footgun — TI2V-5B happens to align on N today, but
    # any future Wan variant with an odd projection width would silently
    # fail at runtime. Log layer name + shape before raising so the offender
    # is immediately identifiable.
    if in_features % 16 != 0 or out_features % 16 != 0:
        log.error(
            "fp8_quant: alignment failure on Linear %s: in=%d out=%d (need both %% 16 == 0)",
            name, in_features, out_features,
        )
        raise ValueError(
            f"fp8 _scaled_mm requires both in_features%16==0 and "
            f"out_features%16==0, got in={in_features} out={out_features} "
            f"for Linear {name!r} ({linear.in_features}->{linear.out_features})"
        )

    # Per-tensor symmetric scale.
    amax = w.abs().max().to(torch.float32).clamp(min=1e-12)
    scale = (amax / _FP8_MAX).to(torch.float32)  # scalar

    # Cast: w_fp8 = round(w / scale) clamped to fp8 range.
    w_scaled = (w.to(torch.float32) / scale).clamp(-_FP8_MAX, _FP8_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)

    # Replace the Parameter so requires_grad metadata is preserved.
    new_param = nn.Parameter(w_fp8, requires_grad=False)
    linear.weight = new_param

    # Bias stays in bf16 (small, numerically sensitive).
    # scale_weight as a buffer so it moves with .to(device) / FSDP.
    linear.register_buffer(
        "scale_weight", scale.detach().clone(), persistent=False
    )

    # Stash original forward so we can fall back if needed (e.g. weight
    # got cast away by some later op).
    linear.original_forward = linear.forward

    base_dtype = torch.bfloat16

    def _fp8_forward(input, m=linear, base_dtype=base_dtype):
        # If something upstream cast the weight away from fp8 (FSDP MP,
        # .to(bf16), etc.), bail out to the original path so we at least
        # produce correct output.
        if m.weight.dtype != torch.float8_e4m3fn:
            return m.original_forward(input)

        if input.dim() < 2:
            return m.original_forward(input)

        # Flatten leading dims to a single M dim for _scaled_mm (which
        # is strictly 2D in -> 2D out).
        in_shape = input.shape
        x = input.reshape(-1, in_shape[-1])

        # fp8 e4m3 dynamic range is +/- 448; pre-clamp activations.
        x = torch.clamp(x.to(base_dtype), min=-_FP8_MAX, max=_FP8_MAX)
        x_fp8 = x.to(torch.float8_e4m3fn).contiguous()

        scale_input = torch.ones((), device=x.device, dtype=torch.float32)
        scale_w = m.scale_weight.to(x.device).to(torch.float32).reshape(())

        bias = m.bias.to(base_dtype) if m.bias is not None else None

        out = torch._scaled_mm(
            x_fp8,
            m.weight.t(),
            out_dtype=base_dtype,
            bias=bias,
            scale_a=scale_input,
            scale_b=scale_w,
        )
        return out.reshape(*in_shape[:-1], m.out_features)

    linear.forward = _fp8_forward


def _iter_block_linears(block: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    """Yield (qualified_name, linear) for every nn.Linear inside one
    transformer block. Skips nothing — the whole point is the block's
    Linears are the speedup target."""
    for name, sub in block.named_modules():
        if isinstance(sub, nn.Linear):
            yield name, sub


def quantize_wan_blocks_to_fp8(model: nn.Module) -> dict:
    """Quantize every Linear in `model.blocks` to fp8_e4m3fn in place.

    Leaves patch_embedding / text_embedding / time_embedding / head in bf16.

    Returns a small stats dict for logging.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            "quantize_wan_blocks_to_fp8: model has no .blocks attribute "
            "(expected WanModel)"
        )

    n_linears = 0
    n_params_quantized = 0
    for block_idx, block in enumerate(model.blocks):
        for name, linear in _iter_block_linears(block):
            n_params_quantized += linear.weight.numel()
            _quantize_linear_inplace(
                linear, name=f"blocks[{block_idx}].{name}"
            )
            n_linears += 1
        if block_idx == 0:
            # Sanity log on the first block.
            names = [n for n, _ in _iter_block_linears(block)]
            log.info(
                "fp8_quant: block[0] Linears converted: %s",
                ", ".join(names),
            )

    log.info(
        "fp8_quant: converted %d Linears across %d blocks (%.2f M params -> fp8)",
        n_linears,
        len(model.blocks),
        n_params_quantized / 1e6,
    )
    return {
        "n_linears": n_linears,
        "n_blocks": len(model.blocks),
        "params_quantized_M": n_params_quantized / 1e6,
    }
