"""Auto-discovery hook registration for model components."""

import functools
import threading

import torch

from wan.profiling._config import get_config

# Known attribute names across all 5 pipelines (verified)
_MODEL_ATTRS = ["low_noise_model", "high_noise_model", "noise_model", "model"]
_TEXT_ENCODER_ATTRS = ["text_encoder"]
_VAE_ATTRS = ["vae"]

# Thread-local storage for hook span state (pre-hook opens, post-hook closes)
_hook_state = threading.local()


def _get_hook_spans():
    if not hasattr(_hook_state, "spans"):
        _hook_state.spans = {}
    return _hook_state.spans


def _open_hook_span(span_name):
    """Called from pre-hook: open a timing span.

    Uses deferred mode when inside a profiled_loop() to avoid
    per-call torch.cuda.synchronize() in the hot path.
    """
    from wan.profiling import _in_loop, _make_deferred_span, trace_span

    spans = _get_hook_spans()
    if _in_loop:
        ctx = _make_deferred_span(span_name, -1)
    else:
        ctx = trace_span(span_name)
    ctx.__enter__()
    spans[span_name] = ctx


def _close_hook_span(span_name):
    """Called from post-hook: close the timing span."""
    spans = _get_hook_spans()
    ctx = spans.pop(span_name, None)
    if ctx is not None:
        ctx.__exit__(None, None, None)


def _make_forward_pre_hook(span_name):
    """Pre-hook opens the timing span before forward() runs."""

    def hook(module, input):
        _open_hook_span(span_name)

    return hook


def _make_forward_hook(span_name):
    """Post-hook closes the span. Never inspects output (N-1).

    Works with all return types: Tensor, List[Tensor], TensorList.
    """

    def hook(module, input, output):
        _close_hook_span(span_name)

    return hook


def _wrap_method(obj, method_name, span_name):
    """Wrap a plain method with profiling. For non-nn.Module classes (VAE).

    Uses deferred mode when inside a profiled_loop().
    """
    original = getattr(obj, method_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        from wan.profiling import _in_loop, _make_deferred_span, trace_span

        ctx = _make_deferred_span(span_name, -1) if _in_loop else trace_span(span_name)
        with ctx:
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def setup_profiling(pipeline):
    """Auto-discover and hook model components on a pipeline instance.

    Works for all 5 pipeline classes without per-pipeline branching:
      - WanT2V:  low_noise_model, high_noise_model (WanModel)
      - WanI2V:  low_noise_model, high_noise_model (WanModel)
      - WanS2V:  noise_model (WanModel_S2V)
      - WanTI2V: model (WanModel)
      - WanAnimate: noise_model (WanAnimateModel)
    """
    config = get_config()
    if not config.enabled:
        return

    # Hook all model variants found
    for attr in _MODEL_ATTRS:
        model = getattr(pipeline, attr, None)
        if model is None:
            continue
        # Defensive: handle torch.compile (I-5)
        target = getattr(model, "_orig_mod", model)
        if not isinstance(target, torch.nn.Module):
            continue
        span_name = f"model_forward/{attr}"
        target.register_forward_pre_hook(_make_forward_pre_hook(span_name))
        target.register_forward_hook(_make_forward_hook(span_name))

    # Hook text encoder (N-2: hooks inner model, minor tokenization gap)
    for attr in _TEXT_ENCODER_ATTRS:
        encoder = getattr(pipeline, attr, None)
        if encoder is None:
            continue
        target = getattr(encoder, "model", encoder)
        if isinstance(target, torch.nn.Module):
            target.register_forward_pre_hook(
                _make_forward_pre_hook("text_encoding")
            )
            target.register_forward_hook(
                _make_forward_hook("text_encoding")
            )

    # Wrap VAE methods (NOT nn.Module — functools.wraps)
    # N-3: Multiple vae.encode() calls get same span name, but manual spans
    # (reference_encoding, pose_encoding) provide nesting context.
    for attr in _VAE_ATTRS:
        vae = getattr(pipeline, attr, None)
        if vae is None:
            continue
        if hasattr(vae, "encode"):
            _wrap_method(vae, "encode", "vae_encode")
        if hasattr(vae, "decode"):
            _wrap_method(vae, "decode", "vae_decode")
