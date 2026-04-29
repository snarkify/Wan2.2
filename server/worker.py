"""Cross-rank job dispatch.

This module owns the WanTI2V pipeline lifecycle. The pipeline is a
**module-level singleton** (`_PIPELINE`), built lazily on the first job
and reused across requests for the lifetime of the process. JobStore
holds job state and Config holds env-var settings; the pipeline is
genuinely worker-owned, so it lives here rather than being injected.

Why a singleton (replaces the old per-request reload):

The original demo-server path constructed a fresh WanTI2V on every job
because FSDP's `to('cpu')` between requests didn't release the
FlatParameter storage on a 24 GB 4090 — the second job VAE-decode-OOMed.
Reload-per-request added ~95 s of model-load to every job. With the
1-GPU FSDP-bypass guard in `wan/textimage2video.py` (world_size==1
demotes `dit_fsdp` to False), there's no FlatParameter to leak. We can
keep the pipeline live and pay the 86 s constructor cost exactly once.

Validated warm steady-state (bench_artifacts/stats_warm_fp8_v2.json,
4090, 81 frames, 50 steps, fp8, offload_model=True):
  gen 1: 287.7 s   gen 2: 288.7 s   gen 3: 288.1 s
  peak alloc 23.1 GB stable, no across-request drift.

torchrun protocol (preserved for runbook/script muscle memory): the
launcher still wraps us with `torchrun --nproc_per_node=N`. At N=1 we
init a degenerate 1-rank process group; non-driver ranks (when N>1) sit
in `worker_client_loop` waiting for broadcasts, but production runs
N=1 and that loop is dormant.

Cross-rank protocol (only used when WAN_DEMO_GPUS>1, retained for the
multi-GPU path):
    {"op": "generate", "kwargs": {...}}     # run a job
    {"op": "shutdown"}                       # clean exit
    {"op": "force_gc"}                       # debug: gc + empty_cache
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from typing import Any, Optional

import torch
import torch.distributed as dist

from server.callbacks import deliver_callback
from server.config import Config
from server.jobstore import Job, JobStore


log = logging.getLogger("server.worker")

_DRIVER_RANK = 0

# Module-level pipeline singleton. None until the first job triggers
# `_get_or_build_pipeline`. Set once and reused for the lifetime of the
# worker process. See module docstring for the rationale.
_PIPELINE = None  # type: Optional["wan.textimage2video.WanTI2V"]


def is_pipeline_warm() -> bool:
    """True once the singleton has been built. Used by the ETA endpoint
    to decide whether to charge callers for the one-time constructor."""
    return _PIPELINE is not None


def _broadcast(msg: dict[str, Any]) -> None:
    """Send a message from rank 0 to all ranks (including self)."""
    payload: list[Any] = [msg]
    dist.broadcast_object_list(payload, src=_DRIVER_RANK)


def _receive() -> dict[str, Any]:
    """Called on non-driver ranks to receive the next message."""
    payload: list[Any] = [None]
    dist.broadcast_object_list(payload, src=_DRIVER_RANK)
    return payload[0]


async def worker_driver_loop(
    store: JobStore,
    config: Config,
    video_base_url: str,
) -> None:
    """Rank 0 worker loop: pull jobs from the store, dispatch, post-process."""
    loop = asyncio.get_running_loop()
    while True:
        job = await store.next_queued()
        if job is None:
            store.wakeup.clear()
            await store.wakeup.wait()
            continue

        log.info("worker: picking up job=%s prompt=%r", job.id, job.prompt[:80])
        await store.mark_running(job.id)

        # Pre-compute the save path so the tensor never has to be
        # returned from the executor — save happens inside _run_job.
        save_path = os.path.join(config.output_dir, job.id, "video.mp4")
        msg = {
            "op": "generate",
            "kwargs": _build_kwargs(job),
            "ckpt_dir": config.ckpt_dir,
            "save_path": save_path,
        }
        try:
            # Only broadcast if there are other ranks listening. At
            # world=1 the broadcast is harmless (collective is a no-op)
            # but skipping it avoids the dist round-trip on the hot path.
            if dist.is_initialized() and dist.get_world_size() > 1:
                _broadcast(msg)
            # The executor returns only the path (a short string), so
            # nothing large crosses the asyncio / ThreadPoolExecutor
            # boundary and gets pinned by the Future's result slot.
            returned_path = await loop.run_in_executor(
                None, _run_job,
                msg["ckpt_dir"], msg["kwargs"], save_path,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.exception("worker: job=%s generation failed", job.id)
            try:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                _mem_snapshot("driver_after_exception_cleanup")
            except Exception:
                pass
            await store.mark_failed(job.id, err)
            job.status = "failed"
            job.error = err
            job.finished_at = time.time()
            asyncio.create_task(
                deliver_callback(store, job, video_base_url)
            )
            continue

        if returned_path is None:
            err = "generate returned no path (rank 0 did not save)"
            log.error("worker: job=%s %s", job.id, err)
            await store.mark_failed(job.id, err)
            job.status = "failed"
            job.error = err
            job.finished_at = time.time()
        else:
            await store.mark_done(job.id, returned_path)
            job.status = "done"
            job.video_path = returned_path
            job.finished_at = time.time()
            log.info("worker: job=%s done path=%s", job.id, returned_path)

        # Best-effort post-job cleanup. With the warm singleton we keep
        # the pipeline alive across jobs; only stray temporaries get
        # released here. The warm bench shows stable 23.1 GB peak alloc
        # across consecutive gens, so this is mostly a defensive sweep.
        try:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            _mem_snapshot("driver_after_save_cleanup")
        except Exception:
            pass

        if job.callback_url:
            asyncio.create_task(
                deliver_callback(store, job, video_base_url)
            )


def worker_client_loop() -> None:
    """Ranks 1..N-1: receive broadcasts and participate in generation.

    Dormant at production world=1 (only rank 0 exists). Retained for the
    multi-GPU path — the singleton works the same across ranks because
    every rank constructs its own pipeline lazily on first job.
    """
    log.info("worker client loop started on rank=%s", os.environ.get("RANK"))
    while True:
        try:
            msg = _receive()
        except Exception:
            log.exception("worker_client: receive failed; exiting")
            return
        op = msg.get("op")
        if op == "shutdown":
            log.info("worker_client: shutdown received")
            return
        if op == "generate":
            try:
                # save_path is rank-0-only (ignored on clients since
                # only rank 0 receives the VAE output tensor), but we
                # pass it through for symmetry with the driver call.
                _run_job(
                    msg["ckpt_dir"], msg["kwargs"], msg.get("save_path")
                )
            except Exception:
                # On non-driver ranks, errors can't be reported back over
                # the distributed group without breaking the next
                # broadcast; log and continue (rank 0 will observe).
                log.exception("worker_client: generate raised")
            continue
        if op == "force_gc":
            _force_gc("client")
            continue
        log.warning("worker_client: unknown op=%r", op)


def _force_gc(label: str) -> dict[str, int]:
    """Aggressive memory reclaim; logs before/after, returns rank-0 stats."""
    import gc as _gc
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    before_alloc = torch.cuda.memory_allocated(local_rank) // (1024 * 1024)
    before_reserved = torch.cuda.memory_reserved(local_rank) // (1024 * 1024)
    before_free, before_total = torch.cuda.mem_get_info(local_rank)
    before_nvml = (before_total - before_free) // (1024 * 1024)

    _gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        # Internal: clears cached cuBLAS workspaces (typically 50-300 MB
        # per device). Bounded to the "matmul workspace config" size.
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    _gc.collect()
    torch.cuda.empty_cache()

    after_alloc = torch.cuda.memory_allocated(local_rank) // (1024 * 1024)
    after_reserved = torch.cuda.memory_reserved(local_rank) // (1024 * 1024)
    after_free, _ = torch.cuda.mem_get_info(local_rank)
    after_nvml = (before_total - after_free) // (1024 * 1024)

    log.info(
        "force_gc[%s] rank=%d alloc %d->%d MB | reserved %d->%d MB "
        "| nvml.used %d->%d MB (delta=%d MB)",
        label, rank, before_alloc, after_alloc, before_reserved, after_reserved,
        before_nvml, after_nvml, before_nvml - after_nvml,
    )
    return {
        "rank": rank,
        "before_alloc_mb": before_alloc,
        "after_alloc_mb": after_alloc,
        "before_reserved_mb": before_reserved,
        "after_reserved_mb": after_reserved,
        "before_nvml_used_mb": before_nvml,
        "after_nvml_used_mb": after_nvml,
        "delta_nvml_mb": before_nvml - after_nvml,
    }


def broadcast_force_gc() -> dict[str, int]:
    """Rank-0-only entrypoint: broadcasts a force_gc op to clients and runs
    the same locally, returning rank 0's stats."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        _broadcast({"op": "force_gc"})
    return _force_gc("driver")


def broadcast_shutdown() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        _broadcast({"op": "shutdown"})


# ----- helpers -----

def _build_kwargs(job: Job) -> dict[str, Any]:
    p = job.params
    size_w, size_h = _parse_size(p["size"])
    # Resolve seed to a concrete value on rank 0 BEFORE the broadcast.
    # If we left -1 in the message, each rank would independently pick a
    # different random seed inside t2v(), producing different noise
    # tensors across ranks. Ulysses then gathers slices from those
    # inconsistent tensors, which shows up as noise in the middle of the
    # video (rank 0 and rank 3 happen to match at the sequence edges).
    # Still load-bearing at world=1 too: Ulysses is off, but pinning a
    # concrete seed gives us a reproducible value for the job record.
    seed = int(p.get("seed", -1))
    if seed < 0:
        import random as _random
        seed = _random.randint(0, 2**63 - 1)
    return {
        "input_prompt": job.prompt,
        "size": (size_w, size_h),
        "frame_num": int(p["frame_num"]),
        "seed": seed,
        "sampling_steps": int(p.get("sampling_steps", 50)),
        "offload_model": True,
    }


def _parse_size(s: str) -> tuple[int, int]:
    """'1280*704' -> (1280, 704)."""
    w, h = s.split("*")
    return int(w), int(h)


def _mem_snapshot(label: str) -> None:
    """Log torch + nvml memory so we can spot across-request accumulation."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        alloc = torch.cuda.memory_allocated(local_rank) // (1024 * 1024)
        reserved = torch.cuda.memory_reserved(local_rank) // (1024 * 1024)
        max_alloc = torch.cuda.max_memory_allocated(local_rank) // (1024 * 1024)
        max_reserved = torch.cuda.max_memory_reserved(local_rank) // (1024 * 1024)
        # nvml per-device free/total as a ground truth, independent of the
        # torch allocator (catches NCCL buffers / cuBLAS workspaces that
        # torch doesn't account for).
        free_b, total_b = torch.cuda.mem_get_info(local_rank)
        used_mb = (total_b - free_b) // (1024 * 1024)
        log.info(
            "mem[%s] rank=%d torch.alloc=%dMB reserved=%dMB peak_alloc=%dMB peak_reserved=%dMB nvml.used=%dMB",
            label, rank, alloc, reserved, max_alloc, max_reserved, used_mb,
        )
    except Exception:
        log.exception("mem[%s] snapshot failed", label)


def _get_or_build_pipeline(ckpt_dir: str):
    """Return the module-level WanTI2V singleton, building it on first call.

    Construction is deferred to the first job rather than process start
    so that `WAN_DEMO_QUANT` and `WAN_DEMO_CKPT_DIR` (read inside the
    constructor) reflect any environment surfaced by the launch script,
    and so that a server with zero traffic doesn't pay the 86 s
    constructor cost it would never use.

    At world=1 the pipeline auto-demotes dit_fsdp to False inside
    `WanTI2V.__init__` (see wan/textimage2video.py world-size guard), so
    there are no FSDP handles to leak between jobs.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    from wan import configs as wan_configs
    from wan.textimage2video import WanTI2V

    cfg = wan_configs.WAN_CONFIGS["ti2v-5B"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    log.info(
        "building pipeline singleton: rank=%d world=%d quant=%s ckpt=%s",
        rank, world_size,
        os.environ.get("WAN_DEMO_QUANT", "bf16"),
        ckpt_dir,
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    t0 = time.perf_counter()

    use_fsdp = world_size > 1
    use_sp = world_size > 1
    _PIPELINE = WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
        # At world=1 these are both False; at world>1 we keep the
        # original FSDP+Ulysses parallelism. The world-size guard in
        # WanTI2V.__init__ would also demote dit_fsdp itself, but
        # passing the right value here keeps logging honest.
        t5_fsdp=False,
        dit_fsdp=use_fsdp,
        use_sp=use_sp,
        t5_cpu=False,           # T5 must run on GPU
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    elapsed = time.perf_counter() - t0
    log.info("pipeline singleton built in %.2fs", elapsed)
    _mem_snapshot("after_pipeline_build")
    return _PIPELINE


def _run_job(
    ckpt_dir: str,
    kwargs: dict[str, Any],
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Run one generation against the warm singleton, save to disk, return path.

    The video tensor is saved to disk INSIDE this function, before the
    function returns. If the tensor crosses the ``loop.run_in_executor``
    boundary (e.g. returned to the asyncio task), the
    ``concurrent.futures.Future`` internally pins it — measured
    empirically as a 180 MB leak per job on rank 0 from
    ``torch.cuda.memory._snapshot()``. Saving inline and returning only
    the path string closes that leak.

    Called from all ranks. Rank 0 saves + returns the path; other ranks
    (when world>1) participate in the distributed collectives and
    return None.
    """
    import gc

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    _mem_snapshot("enter_run_job")
    # Reset peak so we can read a fresh high-water-mark per request.
    torch.cuda.reset_peak_memory_stats(local_rank)

    pipeline = _get_or_build_pipeline(ckpt_dir)

    result_path: Optional[str] = None
    try:
        # reuse=True keeps T5 + DiT loaded across calls — that is the
        # whole point of the warm singleton. T5 still offloads to CPU
        # after encode (built-in), and offload_model=True moves DiT to
        # CPU after the diffusion loop so VAE decode has the 24 GB it
        # needs for 81-frame jobs at 1280x704. Validated in
        # bench_artifacts/stats_warm_fp8_v2.json: peak alloc stays at
        # 23.1 GB across consecutive gens with this exact config.
        result = pipeline.generate(**kwargs, reuse=True)
        _mem_snapshot("after_generate")

        # Save the video INSIDE this function (rank 0 only has the tensor).
        if rank == 0 and save_path is not None and result is not None:
            try:
                from wan.utils.utils import save_video
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_video(
                    tensor=result[None],
                    save_file=save_path,
                    fps=24,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                result_path = save_path
            finally:
                # Drop the tensor BEFORE returning so the executor's
                # result slot doesn't pin it.
                result = None  # noqa: F841
        _mem_snapshot("after_save_inline")
        return result_path
    finally:
        # Defensive sweep. The pipeline is intentionally retained
        # (singleton); we only collect any locals we may have created.
        gc.collect()
        torch.cuda.empty_cache()
        _mem_snapshot("after_cleanup")
