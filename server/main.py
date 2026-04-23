"""Entrypoint for the demo API service.

Launched as:
    torchrun --nproc_per_node=4 --master_port=29600 -m server.main

Rank 0 starts uvicorn + the worker driver loop. Ranks 1..3 sit in
server.worker.worker_client_loop() receiving job broadcasts and
participating in Ulysses all-to-alls during generation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
import uvicorn

from server import config as config_module
from server.app import create_app
from server.jobstore import JobStore
from server.worker import (
    broadcast_shutdown,
    worker_client_loop,
    worker_driver_loop,
)
from server.callbacks import deliver_callback


log = logging.getLogger("server.main")


def _init_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _prewarm_cuda() -> None:
    """Touch CUDA so the first request doesn't pay for context init."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Allocate + free a tiny tensor to trigger CUDA context creation.
    _ = torch.zeros(1, device=f"cuda:{local_rank}")
    del _
    torch.cuda.empty_cache()


def _run_rank0(
    config: config_module.Config,
    store: JobStore,
) -> None:
    video_base_url = os.environ.get(
        "WAN_DEMO_PUBLIC_URL", f"http://{config.host}:{config.port}"
    )
    started_at = time.time()
    app = create_app(config, store, video_base_url, started_at)

    async def main() -> None:
        stale = await store.reconcile_on_startup()
        log.info("reconciled %d stale running jobs", len(stale))
        # Fire callbacks for previously-running jobs (best-effort).
        for j in stale:
            if j.callback_url:
                asyncio.create_task(
                    deliver_callback(store, j, video_base_url)
                )

        # Start the worker loop as a background task.
        worker_task = asyncio.create_task(
            worker_driver_loop(store, config, video_base_url)
        )

        # Start uvicorn within the same event loop so the worker and
        # HTTP server share it. uvicorn.Server handles SIGINT/SIGTERM.
        uvi_config = uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            loop="asyncio",
        )
        server = uvicorn.Server(uvi_config)
        try:
            await server.serve()
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            broadcast_shutdown()
            store.close()

    asyncio.run(main())


def _run_other_rank() -> None:
    """Ranks 1..3: just participate in broadcasts + generation."""
    worker_client_loop()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    rank, world_size, local_rank = _init_distributed()
    log.info(
        "process started rank=%d world=%d local_rank=%d",
        rank, world_size, local_rank,
    )

    _prewarm_cuda()

    if rank == 0:
        config = config_module.load()
        os.makedirs(config.output_dir, exist_ok=True)
        db_path = os.path.join(config.output_dir, "jobs.db")
        store = JobStore(db_path)
        log.info("rank 0 JobStore opened at %s", db_path)
        _run_rank0(config, store)
    else:
        _run_other_rank()

    log.info("process exiting rank=%d", rank)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
