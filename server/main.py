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


def _load_pipeline(config: config_module.Config):
    """Construct WanTI2V on every rank (FSDP+Ulysses needs all ranks)."""
    from wan import configs as wan_configs
    from wan.textimage2video import WanTI2V

    cfg = wan_configs.WAN_CONFIGS["ti2v-5B"]
    # We DO enable FSDP sharding + Ulysses-4 (matches measured best config).
    pipeline = WanTI2V(
        config=cfg,
        checkpoint_dir=config.ckpt_dir,
        device_id=int(os.environ.get("LOCAL_RANK", 0)),
        rank=dist.get_rank(),
        t5_fsdp=False,
        dit_fsdp=True,
        use_sp=True,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    return pipeline


def _run_rank0(
    config: config_module.Config,
    store: JobStore,
    pipeline,
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
            worker_driver_loop(store, pipeline, config, video_base_url)
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


def _run_other_rank(pipeline) -> None:
    """Ranks 1..3: just participate in broadcasts + generation."""
    worker_client_loop(pipeline)


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

    if rank == 0:
        config = config_module.load()
        os.makedirs(config.output_dir, exist_ok=True)
    else:
        config = None

    log.info("loading WanTI2V pipeline on rank=%d", rank)
    # All ranks must construct the pipeline; on non-rank-0, we need a
    # minimal config — pull ckpt_dir from env directly since rank-0's
    # Config isn't broadcast.
    if rank != 0:
        ckpt_dir = os.environ.get("WAN_DEMO_CKPT_DIR")
        if not ckpt_dir:
            raise RuntimeError(
                "WAN_DEMO_CKPT_DIR must be set for all ranks"
            )
        # Build a minimal Config; only fields the pipeline needs matter.
        config = config_module.Config(
            host="", port=0, token="",
            ckpt_dir=ckpt_dir,
            output_dir=os.environ.get("WAN_DEMO_OUTPUT_DIR", "/tmp"),
            max_queue=0, allow_private_callback=False,
        )
    pipeline = _load_pipeline(config)
    log.info("pipeline loaded on rank=%d", rank)

    if rank == 0:
        db_path = os.path.join(config.output_dir, "jobs.db")
        store = JobStore(db_path)
        log.info("rank 0 JobStore opened at %s", db_path)
        _run_rank0(config, store, pipeline)
    else:
        _run_other_rank(pipeline)

    log.info("process exiting rank=%d", rank)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
