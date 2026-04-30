"""TeaCache: training-free step caching for Wan2.2 DiT.

Caches the residual produced by the block stack across diffusion steps
when the modulated time embedding has only changed slightly. The
published recipe gives ~2x wall-clock speedup at threshold 0.20-0.25
with negligible quality drift; we re-use it on Wan2.2 5B.

Reference: ali-vilab/TeaCache, TeaCache4Wan2.1/teacache_generate.py.

Wan2.2 vs Wan2.1 differences this port handles:
  - `e0` is `unflatten(2, (6, dim))` → shape `[B, T, 6, D]` (per-token
    timesteps), vs Wan2.1's `[B, 6, D]`. The rel-L1 distance reduction
    `(.abs().mean())` is shape-agnostic so the math still applies; the
    polynomial calibration is technically sized for the older shape but
    the magnitudes are close enough to be useful at thresholds 0.15-0.30.
  - Block loop is wrapped in `trace_span(f"block_{i}")` for profiling.
    The compute path preserves these spans; the cache-hit path bypasses
    them entirely (no spans for skipped block-loop calls).

Compile interaction:
  - The forward has a Python-level data-dependent branch (`if not
    should_calc:`) fed by `.cpu().item()` on a tensor reduction. Under
    `torch.compile()` this forces a graph break and recompiles per
    branch, so callers should disable model-level compile when TeaCache
    is enabled. The algorithmic 2x dominates the lost ~22% compile
    speedup. Block-level compile (compiling each `WanAttentionBlock`
    individually instead of the whole model) is the cleaner long-term
    fix and is left as a follow-up.

Even/odd split:
  - One diffusion step calls `model.forward()` twice (cond + uncond for
    CFG). The cond and uncond residuals diverge across timesteps, so we
    track them in separate buffers. `cnt` advances by 1 per call; total
    `num_steps` passed to `enable_teacache()` is `2 * sampling_steps`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional, Sequence

import numpy as np
import torch

from ..modules.model import sinusoidal_embedding_1d
from ..profiling import trace_span
from ..profiling._config import get_config


# Polynomials map raw rel-L1 distance → calibrated rel-L1 estimate of
# true model-output drift. Values published in TeaCache4Wan2.1; we use
# the 14B set for 5B since (a) no published 5B coefficients exist and
# (b) Voltage Park's Wan2.2 result also took this path. If quality A/B
# fails we'll fit a 5B-specific polynomial in a follow-up calibration
# pass (~30 min of GPU time on gpu6).
COEFFICIENTS_WAN_14B = (
    -5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404,
)
COEFFICIENTS_WAN_14B_RET = (
    -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01,
    -3.15583525e-01,
)
COEFFICIENTS_WAN_1_3B = (
    2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00,
    1.37887774e-01,
)
COEFFICIENTS_WAN_1_3B_RET = (
    -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01,
    -4.99875664e-02,
)


def _teacache_forward(self, x, t, context, seq_len, y=None):
    """Wan2.2 forward with TeaCache step-caching. Replaces
    `WanModel.forward` when caching is enabled.

    Mirrors the original forward at `wan/modules/model.py:413` block by
    block, with three modifications around the block loop:
      1. compute `modulated_inp` (= e0 or e per `_tc_use_ret_steps`)
      2. accumulate rescaled rel-L1 distance vs the previous step's
         modulated_inp; decide `should_calc` against the threshold
      3. on cache-hit, skip the block loop and add the prior residual
    """
    if self.model_type == 'i2v':
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    _detail = get_config().model_detail

    with trace_span("patch_embedding") if _detail else nullcontext():
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat(
                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                dim=1) for u in x
        ])

    with trace_span("time_embedding") if _detail else nullcontext():
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            tf = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.freq_dim, tf).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

    with trace_span("text_embedding") if _detail else nullcontext():
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    # --- TeaCache decision -------------------------------------------
    modulated_inp = e0 if self._tc_use_ret_steps else e
    is_even = (self._tc_cnt % 2 == 0)
    parity = "even" if is_even else "odd"
    prev_e0_attr = f"_tc_previous_e0_{parity}"
    accum_attr = f"_tc_accumulated_rel_l1_{parity}"
    residual_attr = f"_tc_previous_residual_{parity}"

    in_warmup = self._tc_cnt < self._tc_ret_steps
    in_cooldown = self._tc_cnt >= self._tc_cutoff_steps
    prev_e0 = getattr(self, prev_e0_attr)

    if in_warmup or in_cooldown or prev_e0 is None:
        should_calc = True
        setattr(self, accum_attr, 0.0)
        if self._tc_debug:
            logging.info(
                "[TEACACHE] cnt=%d %s warmup/cooldown -> compute",
                self._tc_cnt, parity,
            )
    else:
        # `.cpu().item()` forces a sync; this is intentional (without
        # it the rescaled accumulator can't drive a Python branch).
        rel_l1 = ((modulated_inp - prev_e0).abs().mean()
                  / prev_e0.abs().mean()).cpu().item()
        if self._tc_use_polynomial:
            rescale = float(np.poly1d(self._tc_coefficients)(rel_l1))
        else:
            rescale = rel_l1  # raw mode: skip the polynomial entirely
        new_accum = getattr(self, accum_attr) + rescale
        if new_accum < self._tc_thresh:
            should_calc = False
            setattr(self, accum_attr, new_accum)
        else:
            should_calc = True
            setattr(self, accum_attr, 0.0)
        if self._tc_debug:
            logging.info(
                "[TEACACHE] cnt=%d %s rel_l1=%.6f rescale=%.6f accum=%.6f -> %s",
                self._tc_cnt, parity, rel_l1, rescale, new_accum,
                "compute" if should_calc else "skip",
            )

    setattr(self, prev_e0_attr, modulated_inp.clone())

    # --- Block stack: compute path or cache-hit path -----------------
    prev_residual = getattr(self, residual_attr)
    if should_calc or prev_residual is None:
        ori_x = x.clone()
        for i, block in enumerate(self.blocks):
            with trace_span(f"block_{i}") if _detail else nullcontext():
                x = block(x, **kwargs)
        setattr(self, residual_attr, x - ori_x)
        self._tc_stats_compute += 1
    else:
        x = x + prev_residual
        self._tc_stats_skip += 1

    with trace_span("head") if _detail else nullcontext():
        x = self.head(x, e)

    with trace_span("unpatchify") if _detail else nullcontext():
        x = self.unpatchify(x, grid_sizes)

    self._tc_cnt += 1
    if self._tc_cnt >= self._tc_num_steps:
        # Wrap-around safety net for callers that don't reset between
        # generations. Production path calls `reset_teacache()` at the
        # top of each generate, making this unreachable in normal use.
        self._tc_cnt = 0
    return [u.float() for u in x]


def enable_teacache(
    model,
    *,
    thresh: float,
    num_steps: int,
    use_ret_steps: bool = False,
    ret_steps: int = 1,
    coefficients: Optional[Sequence[float]] = None,
    use_polynomial: bool = True,
    debug: bool = False,
) -> None:
    """Install TeaCache on a Wan2.2 DiT module.

    Replaces `model.forward` with the cached variant and stashes
    per-instance state under `_tc_*` attributes. Idempotent.

    Args:
        model: WanModel instance, or a `torch.compile()`'d wrapper —
            we patch the underlying class either way (and recommend
            disabling compile when TeaCache is on; see module docstring).
        thresh: rescaled-rel-L1 accumulator threshold. Higher = more
            aggressive caching, more drift. Recommended 0.15-0.25 on 5B.
        num_steps: total forward calls per generate (= `2 *
            sampling_steps` for CFG). Used for the cnt wrap-around.
        use_ret_steps: True selects the `_RET` polynomial calibrated on
            e0 (vs e). Slightly higher quality at the same threshold;
            slightly less speedup. Default False matches the standard
            recipe.
        ret_steps: diffusion steps at the start of generation that
            always recompute (counted in steps, not forward calls).
            Default 1 (= first step uncached). With use_ret_steps=True
            increase to 5 per the reference recipe.
        coefficients: 5-element polynomial. Defaults to the 14B set
            (closest published to our 5B model).
        use_polynomial: True applies the rescaling polynomial; False
            uses raw rel-L1 directly (bypass mode). The 14B polynomial
            produces negative rescaled values for typical 5B/Wan2.2
            rel-L1 inputs (which are O(0.01-0.05), well below the
            polynomial's calibration domain), so the accumulator can
            never grow past the threshold. Bypass mode reads raw
            rel-L1 as a positive monotonic distance signal — at the
            cost of losing the calibration that maps embedding-space
            distance to true output drift. Recommended threshold range
            shifts to 0.05-0.20 in raw mode.
        debug: if True, log per-call rel_l1, rescale, accum, decision.
            Set this only for short calibration runs — the format is
            verbose and the .cpu().item() is already on the hot path.
    """
    if coefficients is None:
        coefficients = (
            COEFFICIENTS_WAN_14B_RET if use_ret_steps
            else COEFFICIENTS_WAN_14B
        )

    # `torch.compile()` returns an OptimizedModule wrapper — peel it
    # off to access the real module class.
    target = getattr(model, "_orig_mod", model)
    cls = target.__class__

    cls.forward = _teacache_forward

    # Register cache tensors as non-persistent buffers so they ride with
    # the module on `.cpu()` / `.to(device)` calls. Without this they
    # stay pinned on GPU when `pipeline.generate()` does `model.cpu()`
    # before VAE decode, and the extra ~300 MB pushes us past the 24 GB
    # 4090 ceiling. Only register if the buffer doesn't already exist —
    # this function is idempotent.
    for buf_name in (
        "_tc_previous_e0_even",
        "_tc_previous_e0_odd",
        "_tc_previous_residual_even",
        "_tc_previous_residual_odd",
    ):
        if buf_name not in target._buffers:
            target.register_buffer(buf_name, None, persistent=False)

    target._tc_thresh = float(thresh)
    target._tc_num_steps = int(num_steps)
    target._tc_use_ret_steps = bool(use_ret_steps)
    target._tc_ret_steps = int(ret_steps) * 2  # in forward-call units
    # Always-recompute for the very last step (matches reference): pull
    # cutoff back by 2 forward-calls when use_ret_steps is False, since
    # the standard recipe leaves the last step uncached. With
    # use_ret_steps=True the polynomial is robust enough to skip cutoff.
    target._tc_cutoff_steps = int(
        num_steps if use_ret_steps else num_steps - 2
    )
    target._tc_coefficients = tuple(coefficients)
    target._tc_use_polynomial = bool(use_polynomial)
    target._tc_debug = bool(debug)
    reset_teacache(target)

    logging.info(
        "[WAN_DEMO_TEACACHE] enabled thresh=%.3f num_steps=%d "
        "ret_steps=%d cutoff_steps=%d use_ret_steps=%s "
        "use_polynomial=%s debug=%s",
        thresh, num_steps, ret_steps, target._tc_cutoff_steps,
        use_ret_steps, use_polynomial, debug,
    )


def reset_teacache(model) -> None:
    """Zero the per-generation TeaCache state. Call before each
    `pipeline.generate()` to keep cnt/accumulators/residuals scoped to
    one generation; otherwise state from gen N leaks into gen N+1 and
    misaligns the even/odd parity for the first ~2 forwards."""
    target = getattr(model, "_orig_mod", model)
    target._tc_cnt = 0
    target._tc_accumulated_rel_l1_even = 0.0
    target._tc_accumulated_rel_l1_odd = 0.0
    target._tc_previous_e0_even = None
    target._tc_previous_e0_odd = None
    target._tc_previous_residual_even = None
    target._tc_previous_residual_odd = None
    target._tc_stats_compute = 0
    target._tc_stats_skip = 0


def teacache_stats(model) -> dict:
    """Return per-generation cache hit/miss counters since the last
    `reset_teacache()`. Used by the bench harness to verify caching
    actually engaged (compute_calls + skip_calls == num_steps)."""
    target = getattr(model, "_orig_mod", model)
    return {
        "compute_calls": getattr(target, "_tc_stats_compute", 0),
        "skip_calls": getattr(target, "_tc_stats_skip", 0),
    }


def is_teacache_enabled(model) -> bool:
    """True if TeaCache has been installed on this model."""
    target = getattr(model, "_orig_mod", model)
    return hasattr(target, "_tc_thresh")
