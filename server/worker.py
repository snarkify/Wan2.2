"""Cross-rank job dispatch.

rank 0 drives the HTTP server + queue, and broadcasts each job to all
ranks via torch.distributed. Ranks 1..3 live in worker_client_loop()
waiting for broadcasts and calling the same pipeline.generate().

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
    pipeline,            # type: WanTI2V  (avoid top-level import cost)
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

        msg = {"op": "generate", "kwargs": _build_kwargs(job)}
        try:
            _broadcast(msg)
            # Run the actual generation on rank 0's thread pool so the
            # asyncio loop stays responsive to HTTP and health checks.
            video_tensor = await loop.run_in_executor(
                None, _run_generate, pipeline, msg["kwargs"]
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log.exception("worker: job=%s generation failed", job.id)
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

        if job.callback_url:
            asyncio.create_task(
                deliver_callback(store, job, video_base_url)
            )


def worker_client_loop(pipeline) -> None:
    """Ranks 1..3: loop receiving broadcasts and participating in generation."""
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
                _run_generate(pipeline, msg["kwargs"])
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
        "reuse": True,
    }


def _parse_size(s: str) -> tuple[int, int]:
    """'1280*704' -> (1280, 704)."""
    w, h = s.split("*")
    return int(w), int(h)


def _run_generate(pipeline, kwargs: dict[str, Any]):
    """Call the pipeline; returns tensor on rank 0, None on other ranks."""
    return pipeline.generate(**kwargs)


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
