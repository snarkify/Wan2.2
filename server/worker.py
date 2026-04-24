"""Cross-rank job dispatch.

rank 0 drives the HTTP server + queue, and broadcasts each job to all
ranks via torch.distributed. Ranks 1..3 live in worker_client_loop()
waiting for broadcasts and calling the same pipeline.generate().

A fresh WanTI2V pipeline is **constructed for every request**. We tried
keeping a persistent pipeline with reuse=True (moving T5+DiT between
CPU and GPU between requests), but .to('cpu') on an FSDP-wrapped model
doesn't release the FlatParameter storage, so VAE decode OOMed on
24 GB 4090s. Reloading per request adds ~95s of model-load overhead
per job but is known-working (same path as the measured standalone
runs in docs/profiling-results.md).

Protocol (wrapped in a one-element list for broadcast_object_list):
    {"op": "generate", "kwargs": {...}}     # run a job
    {"op": "shutdown"}                       # clean exit
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
        # returned from the executor — save happens inside _build_and_run.
        save_path = os.path.join(config.output_dir, job.id, "video.mp4")
        msg = {
            "op": "generate",
            "kwargs": _build_kwargs(job),
            "ckpt_dir": config.ckpt_dir,
            "save_path": save_path,
        }
        try:
            _broadcast(msg)
            # The executor returns only the path (a short string), so
            # nothing large crosses the asyncio / ThreadPoolExecutor
            # boundary and gets pinned by the Future's result slot.
            returned_path = await loop.run_in_executor(
                None, _build_and_run,
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

        # Best-effort post-job cleanup. Nothing tensor-shaped lives in
        # the driver loop anymore, but an extra gc+empty_cache catches
        # any stray cross-task references.
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
    """Ranks 1..3: loop receiving broadcasts and participating in generation.

    Constructs a fresh pipeline per request (matches rank 0 behavior)
    so FSDP/NCCL state is reset between jobs.
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
                _build_and_run(
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
    _broadcast({"op": "force_gc"})
    return _force_gc("driver")


def broadcast_shutdown() -> None:
    _broadcast({"op": "shutdown"})


# ----- helpers -----

def _build_kwargs(job: Job) -> dict[str, Any]:
    p = job.params
    size_w, size_h = _parse_size(p["size"])
    return {
        "input_prompt": job.prompt,
        "size": (size_w, size_h),
        "frame_num": int(p["frame_num"]),
        "seed": int(p.get("seed", -1)),
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


def _teardown_pipeline(pipeline) -> None:
    """Drop every attribute that might hold a GPU allocation.

    t2v(reuse=False) already `del`s text_encoder and calls free_model on
    the DiT before returning, so in the happy path this is a no-op. But
    when generate() raises mid-way (e.g., OOM in VAE decode), the
    pipeline still owns self.vae / self.model / self.text_encoder with
    live CUDA storage. The FSDP wrapper also creates a reference cycle
    via the sequence-parallel `model.forward = MethodType(..., model)`
    monkey-patch, so a single gc.collect() is not enough — we null the
    attributes explicitly to break the cycle before collecting.
    """
    for attr in ("vae", "text_encoder", "model"):
        if hasattr(pipeline, attr):
            obj = getattr(pipeline, attr)
            if obj is not None:
                # Break FSDP's `model.forward -> MethodType(fn, model)` cycle
                # by dropping the bound method before dropping the module.
                if hasattr(obj, "forward") and hasattr(obj.forward, "__self__"):
                    try:
                        del obj.forward
                    except (AttributeError, TypeError):
                        pass
            setattr(pipeline, attr, None)


def _build_and_run(
    ckpt_dir: str,
    kwargs: dict[str, Any],
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Construct a fresh WanTI2V, run one generation, save the video, let GC destroy it.

    IMPORTANT: the video tensor is saved to disk INSIDE this function, before
    the function returns. If the tensor crosses the ``loop.run_in_executor``
    boundary (e.g. returned to the asyncio task), the ``concurrent.futures.Future``
    internally pins it — measured empirically as a 180 MB leak per job on
    rank 0 from `torch.cuda.memory._snapshot()`. Saving inline and returning
    only the path string closes that leak.

    Called from all ranks. Rank 0 saves + returns the path; other ranks
    participate in the distributed collectives and return None.
    """
    import gc
    from wan import configs as wan_configs
    from wan.textimage2video import WanTI2V

    cfg = wan_configs.WAN_CONFIGS["ti2v-5B"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    _mem_snapshot("enter_build_and_run")
    # Reset peak so we can read a fresh high-water-mark per request.
    torch.cuda.reset_peak_memory_stats(local_rank)

    pipeline = WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=True,
        use_sp=True,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    _mem_snapshot("after_pipeline_ctor")
    result_path: Optional[str] = None
    try:
        # reuse=False means the pipeline frees T5 and DiT cleanly at the
        # end; the WanTI2V object becomes unusable afterwards but we
        # throw it away here.
        result = pipeline.generate(**kwargs, reuse=False)
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
        _teardown_pipeline(pipeline)
        pipeline = None  # noqa: F841
        # Two passes: the first collects the pipeline + wrapper modules,
        # the second collects anything they were keeping alive via cycles.
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        _mem_snapshot("after_cleanup")
