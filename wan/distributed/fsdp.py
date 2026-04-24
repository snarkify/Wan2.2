# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False
):
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False)
    return model


def free_model(model):
    """Release all CUDA memory held by an FSDP-wrapped model.

    Freeing just the sharded `flat_param.data` is insufficient because FSDP1
    also keeps around a larger all-gather destination buffer
    (`_full_param_padded`) that holds the unsharded parameters during
    forward. Under eager mode FSDP frees it via `_reshard` after each layer,
    but the Python tensor object survives on the handle, and in certain
    control-flow paths (eval mode, mixed precision, reuse=False teardown)
    the underlying storage is not released. This caused a ~180 MB-per-job
    leak on 24 GB 4090s as observed via torch.cuda.memory._snapshot()
    stack-tracing to torch/distributed/fsdp/_flat_param.py:1381
    `_alloc_padded_unsharded_flat_param`.

    So free every flat_param-adjacent storage we can find before dropping
    the module references.
    """
    # Attribute names FSDP1 may hang unsharded buffers on (per handle).
    _UNSHARDED_ATTRS = (
        "_full_param_padded",
        "_full_prec_full_param_padded",
        "_padded_unsharded_flat_param",
        "_full_unsharded_flat_param",
    )
    for m in model.modules():
        if not isinstance(m, FSDP):
            continue
        handles = []
        if getattr(m, "_handle", None) is not None:
            handles.append(m._handle)
        # Newer FSDP1 revs expose `._handles` (plural) instead.
        for h in getattr(m, "_handles", None) or ():
            if h is not None:
                handles.append(h)
        for h in handles:
            fp = getattr(h, "flat_param", None)
            if fp is None:
                continue
            # Sharded shard storage.
            try:
                _free_storage(fp.data)
            except Exception:
                pass
            # Unsharded all-gather destination buffers kept on the
            # FlatParameter / handle itself.
            for obj in (fp, h):
                for attr in _UNSHARDED_ATTRS:
                    t = getattr(obj, attr, None)
                    if t is None:
                        continue
                    try:
                        _free_storage(t.data if hasattr(t, "data") else t)
                    except Exception:
                        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
