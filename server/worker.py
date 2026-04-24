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

        msg = {"op": "generate", "kwargs": _build_kwargs(job), "ckpt_dir": config.ckpt_dir}
        video_tensor = None
        try:
            _broadcast(msg)
            # Run the actual generation on rank 0's thread pool so the
            # asyncio loop stays responsive to HTTP and health checks.
            # We construct and destroy the pipeline inside the executor
            # so nothing stays resident between requests.
            video_tensor = await loop.run_in_executor(
                None, _build_and_run, msg["ckpt_dir"], msg["kwargs"]
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.exception("worker: job=%s generation failed", job.id)
            # Best-effort cleanup so the next request isn't starved by
            # leftover tensors from a mid-decode OOM. Drop the partial
            # video tensor reference explicitly; the outer local would
            # otherwise pin it until the next successful reassignment.
            try:
                import gc as _gc
                video_tensor = None  # noqa: F841
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

        # Write the video to disk. rank 0 has the tensor; other ranks returned None.
        try:
            video_path = await loop.run_in_executor(
                None, _save_video_for_job, video_tensor, job, config
            )
            await store.mark_done(job.id, video_path)
            job.status = "done"
            job.video_path = video_path
            job.finished_at = time.time()
            log.info("worker: job=%s done path=%s", job.id, video_path)
        except Exception as e:
            err = f"save_video: {type(e).__name__}: {e}"
            log.exception("worker: job=%s save failed", job.id)
            await store.mark_failed(job.id, err)
            job.status = "failed"
            job.error = err
            job.finished_at = time.time()
        finally:
            # Release the returned video tensor before looping to the next
            # job. Without this the ~876 MB (81f, fp32, 1280x704) tensor
            # stays bound to the outer local until the next successful
            # reassignment — pushing 81f VAE decode over the 24 GB budget
            # on the *next* request and causing mid-decode OOMs.
            try:
                import gc as _gc
                video_tensor = None  # noqa: F841
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
                _build_and_run(msg["ckpt_dir"], msg["kwargs"])
            except Exception:
                # On non-driver ranks, errors can't be reported back over
                # the distributed group without breaking the next
                # broadcast; log and continue (rank 0 will observe).
                log.exception("worker_client: generate raised")
            continue
        log.warning("worker_client: unknown op=%r", op)


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


def _build_and_run(ckpt_dir: str, kwargs: dict[str, Any]):
    """Construct a fresh WanTI2V, run one generation, let GC destroy it.

    Called from all ranks. Returns the tensor on rank 0, None elsewhere.
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
    try:
        # reuse=False means the pipeline frees T5 and DiT cleanly at the
        # end; the WanTI2V object becomes unusable afterwards but we
        # throw it away here.
        result = pipeline.generate(**kwargs, reuse=False)
        _mem_snapshot("after_generate")
        return result
    finally:
        _teardown_pipeline(pipeline)
        pipeline = None  # noqa: F841
        # Two passes: the first collects the pipeline + wrapper modules,
        # the second collects anything they were keeping alive via cycles.
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        _mem_snapshot("after_cleanup")


def _save_video_for_job(video_tensor, job: Job, config: Config) -> str:
    """Write the mp4 under OUTPUT_DIR/<job_id>/video.mp4; return that path."""
    from wan.utils.utils import save_video

    job_dir = os.path.join(config.output_dir, job.id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "video.mp4")
    # save_video expects (B, C, T, H, W); WanTI2V returns (C, T, H, W).
    save_video(
        tensor=video_tensor[None],
        save_file=path,
        fps=24,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return path
