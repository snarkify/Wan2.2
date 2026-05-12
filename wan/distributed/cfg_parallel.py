# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""CFG-parallel: split classifier-free guidance forwards across rank groups.

For the cond/uncond forward pair in each diffusion step, dispatch to different
rank groups so they run concurrently rather than sequentially. After both
forwards complete, an all_gather across the CFG group makes both tensors
available on every rank for the guidance merge.

Layout for `cfg_parallel_size=C`, `ulysses_size=S`, `world_size=C*S`:
- world is partitioned into C contiguous "inner" groups of size S each.
  Inner group `c` owns ranks [c*S, (c+1)*S). Ulysses operates inside an inner
  group (TODO: Ulysses today uses dist.group.WORLD; combining with cfg requires
  scoping its all_to_all/all_gather to the inner group).
- inner-rank `r` (0..S-1) is paired across all C inner groups, forming a
  "cfg group" of ranks [r, r+S, r+2S, ...). all_gather across this group
  exchanges the cond and uncond results.

For C=2, S=1 (the 2-GPU Phase 2 case): rank 0 owns cond, rank 1 owns uncond,
no Ulysses inside, single all_gather of shape `(C, *result_shape)` per step.
"""

import torch
import torch.distributed as dist


_CFG_SIZE = 1
_CFG_RANK = 0  # 0 = cond branch, 1 = uncond branch (when CFG_SIZE >= 2)
_CFG_GROUP = None
_INNER_GROUP = None
_WORLD_INITIALIZED = False


def init_cfg_parallel(cfg_size: int) -> None:
    """Initialize CFG-parallel groups. Must be called after `dist.init_process_group`.
    Idempotent for `cfg_size == 1`."""
    global _CFG_SIZE, _CFG_RANK, _CFG_GROUP, _INNER_GROUP, _WORLD_INITIALIZED
    _WORLD_INITIALIZED = True
    if cfg_size <= 1:
        _CFG_SIZE = 1
        _CFG_RANK = 0
        _CFG_GROUP = None
        _INNER_GROUP = None
        return

    if cfg_size != 2:
        raise ValueError(
            f"cfg_parallel_size must be 1 or 2 (got {cfg_size}); "
            "Wan2.2 has exactly two CFG forwards per step.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % cfg_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"cfg_parallel_size ({cfg_size}).")

    inner_size = world_size // cfg_size
    _CFG_SIZE = cfg_size
    _CFG_RANK = rank // inner_size
    inner_rank = rank % inner_size

    # Inner groups (each holds one CFG branch's ranks).
    for cfg in range(cfg_size):
        members = list(range(cfg * inner_size, (cfg + 1) * inner_size))
        group = dist.new_group(members)
        if cfg == _CFG_RANK:
            _INNER_GROUP = group

    # CFG groups (each holds the same inner_rank across all CFG branches).
    for inner in range(inner_size):
        members = list(range(inner, world_size, inner_size))
        group = dist.new_group(members)
        if inner == inner_rank:
            _CFG_GROUP = group


def get_cfg_size() -> int:
    return _CFG_SIZE


def get_cfg_rank() -> int:
    return _CFG_RANK


def is_cond_rank() -> bool:
    """True if this rank should run the conditional forward (cfg_rank == 0)."""
    return _CFG_RANK == 0


def cfg_all_gather(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather a CFG-local tensor across the CFG group.

    Returns (cond, uncond) — each rank receives both, regardless of which
    branch it computed locally.

    For `cfg_size == 1` this is a no-op: returns (tensor, tensor) so the caller
    can use the same code path with `cond == uncond` (which collapses the
    guidance merge to no-op too).
    """
    if _CFG_SIZE == 1 or _CFG_GROUP is None:
        return tensor, tensor

    contig = tensor.contiguous()
    gathered = [torch.empty_like(contig) for _ in range(_CFG_SIZE)]
    dist.all_gather(gathered, contig, group=_CFG_GROUP)
    # Explicitly synchronize so downstream ops (scheduler.step's torch.tensor
    # allocations) don't trigger sync-on-alloc against the NCCL stream — that
    # implicit sync was costing ~2.9 s / step in Y4 (xDiT path doesn't have it).
    torch.cuda.current_stream().synchronize()
    return gathered[0], gathered[1]
