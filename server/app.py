"""FastAPI routes for the demo API.

All routes require `Authorization: Bearer <WAN_DEMO_TOKEN>` except for
GET /v1/generations/{id}/video which also accepts a per-job one-time
token via `?token=...` so that external services (e.g. Slack) can
download the mp4 without our shared secret.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from server.callbacks import (
    CallbackValidationError,
    validate_callback_headers,
    validate_callback_url,
)
from server.config import MAX_FRAMES, MIN_FRAMES, Config
from server.jobstore import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobStore,
)


log = logging.getLogger("server.app")


# ----- request / response models -----


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    frame_num: int = 81
    seed: int = -1
    size: str = "1280*704"
    sampling_steps: int = 50
    callback_url: Optional[HttpUrl] = None
    callback_headers: Optional[dict[str, str]] = None
    callback_metadata: Optional[dict[str, Any]] = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int
    eta_seconds: float


class JobResponse(BaseModel):
    job_id: str
    status: str
    prompt: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_position: Optional[int] = None
    error: Optional[str] = None
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    callback_status: Optional[str] = None
    callback_attempts: Optional[int] = None


class HealthResponse(BaseModel):
    model_loaded: bool
    queue_size: int
    active_jobs: int
    uptime_s: float
    gpu_free_mb: list[int]


# ----- app factory -----


def create_app(
    config: Config,
    store: JobStore,
    video_base_url: str,
    started_at: float,
) -> FastAPI:
    app = FastAPI(title="Wan2.2 TI2V-5B Demo API", version="1.0.0")

    def require_bearer(authorization: Optional[str] = Header(None)) -> None:
        expected = f"Bearer {config.token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    # ---- POST /v1/generations ----

    @app.post(
        "/v1/generations",
        response_model=GenerateResponse,
        dependencies=[Depends(require_bearer)],
    )
    async def create_job(req: GenerateRequest):
        _validate_params(req)
        try:
            validate_callback_url(
                str(req.callback_url) if req.callback_url else "",
                allow_private=config.allow_private_callback,
            )
            validate_callback_headers(req.callback_headers)
        except CallbackValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        depth = await store.queue_depth()
        if depth >= config.max_queue:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"queue full ({depth}/{config.max_queue})",
                },
                headers={"Retry-After": "600"},
            )

        params = {
            "frame_num": req.frame_num,
            "seed": req.seed,
            "size": req.size,
            "sampling_steps": req.sampling_steps,
        }
        job = await store.enqueue(
            prompt=req.prompt,
            params=params,
            callback_url=str(req.callback_url) if req.callback_url else None,
            callback_headers=req.callback_headers,
            callback_metadata=req.callback_metadata,
        )
        position = await store.queue_position(job.id)
        if position is None:
            position = depth  # fallback
        return GenerateResponse(
            job_id=job.id,
            status=job.status,
            queue_position=position,
            eta_seconds=config.eta_seconds(position, req.frame_num),
        )

    # ---- GET /v1/generations/{id} ----

    @app.get(
        "/v1/generations/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_bearer)],
    )
    async def get_job(job_id: str):
        job = await store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_to_response(job, video_base_url, store_position=None)

    # ---- DELETE /v1/generations/{id} ----

    @app.delete(
        "/v1/generations/{job_id}",
        dependencies=[Depends(require_bearer)],
    )
    async def cancel_job(job_id: str):
        job = await store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status == STATUS_RUNNING:
            raise HTTPException(
                status_code=409,
                detail="job already running; cannot cancel",
            )
        if job.status != STATUS_QUEUED:
            raise HTTPException(
                status_code=409,
                detail=f"job is {job.status}; nothing to cancel",
            )
        cancelled = await store.cancel_if_queued(job_id)
        if not cancelled:
            raise HTTPException(status_code=409, detail="state changed; retry")
        return {"job_id": job_id, "status": STATUS_CANCELLED}

    # ---- GET /v1/generations/{id}/video ----

    @app.get("/v1/generations/{job_id}/video")
    async def download_video(
        job_id: str,
        token: Optional[str] = Query(None),
        authorization: Optional[str] = Header(None),
    ):
        job = await store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        if token is not None:
            if token != job.video_token:
                raise HTTPException(status_code=401, detail="bad video token")
        else:
            if authorization != f"Bearer {config.token}":
                raise HTTPException(status_code=401, detail="unauthorized")

        if job.status != STATUS_DONE or not job.video_path:
            raise HTTPException(
                status_code=404,
                detail=f"video not available (status={job.status})",
            )
        if not os.path.exists(job.video_path):
            raise HTTPException(
                status_code=410,
                detail="video file missing on disk",
            )
        return FileResponse(
            job.video_path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    # ---- GET /v1/health ----

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            model_loaded=True,
            queue_size=await store.queue_depth(),
            active_jobs=len(await store.list_active()),
            uptime_s=time.time() - started_at,
            gpu_free_mb=_gpu_free_mb(),
        )

    # ---- GET /v1/debug/memory ----
    # Rank-0 torch-allocator snapshot. Authenticated because it's a
    # diagnostic; cheap enough to call between jobs.
    @app.get(
        "/v1/debug/memory",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_memory():
        return _debug_memory()

    # ---- POST /v1/debug/force_gc ----
    # Triggers gc.collect + empty_cache + ipc_collect + clearCublasWorkspaces
    # on every rank via broadcast. Returns rank-0 before/after for quick
    # visibility into whether leaks are recoverable at the Python+allocator
    # level vs stuck in CUDA libraries.
    @app.post(
        "/v1/debug/force_gc",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_force_gc():
        from server.worker import broadcast_force_gc
        return broadcast_force_gc()

    # ---- GET /v1/debug/allocations ----
    # Dump torch.cuda.memory._snapshot() for rank 0 — returns the list
    # of live allocations with sizes, so we can see WHAT is leaked.
    @app.get(
        "/v1/debug/allocations",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_allocations():
        import torch
        out: dict[str, Any] = {}
        try:
            snap = torch.cuda.memory._snapshot()
            # Filter to actually-allocated blocks only (state="active_allocated").
            allocs = []
            for seg in snap.get("segments", []):
                if seg.get("device") != 0:
                    continue
                for blk in seg.get("blocks", []):
                    if blk.get("state") == "active_allocated":
                        allocs.append({
                            "size_mb": blk["size"] // (1024 * 1024),
                            "size_bytes": blk["size"],
                            "segment_total_mb": seg["total_size"] // (1024 * 1024),
                            "segment_pool": seg.get("segment_pool_id"),
                        })
            out["rank0_active_allocations"] = allocs
            out["rank0_active_count"] = len(allocs)
            out["rank0_active_total_mb"] = sum(a["size_mb"] for a in allocs)
            # Segment summary: all segments on device 0 regardless of state.
            segs = [
                {
                    "total_mb": s["total_size"] // (1024 * 1024),
                    "allocated_mb": s.get("allocated_size", 0) // (1024 * 1024),
                    "active_mb": s.get("active_size", 0) // (1024 * 1024),
                    "stream": s.get("stream"),
                    "num_blocks": len(s.get("blocks", [])),
                    "pool": s.get("segment_pool_id"),
                    "segment_type": s.get("segment_type"),
                }
                for s in snap.get("segments", [])
                if s.get("device") == 0
            ]
            out["rank0_segments"] = segs
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    # ---- POST /v1/debug/start_history ----
    # Start recording allocation stack traces (rank 0 only). After this,
    # every torch.cuda allocation/free records a Python frame list on
    # the block. Call this BEFORE enqueueing the job you want to trace.
    # Overhead: measurable, but fine for a single diagnostic run.
    @app.post(
        "/v1/debug/start_history",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_start_history():
        import torch
        try:
            torch.cuda.memory._record_memory_history(
                max_entries=100_000,
                context="all",
                stacks="python",
            )
            return {"recording": True}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- POST /v1/debug/stop_history ----
    @app.post(
        "/v1/debug/stop_history",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_stop_history():
        import torch
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
            return {"recording": False}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- GET /v1/debug/dump_snapshot ----
    # Returns the *frames* (Python stacks) for every active_allocated
    # block on rank 0. Requires /v1/debug/start_history to have been
    # called before the allocation happened. The 'frames' list is the
    # call stack where the block was allocated — read it top-to-bottom
    # to find the line that constructed the tensor.
    @app.get(
        "/v1/debug/dump_snapshot",
        dependencies=[Depends(require_bearer)],
    )
    async def debug_dump_snapshot():
        import torch
        out: dict[str, Any] = {}
        try:
            snap = torch.cuda.memory._snapshot()
            blocks = []
            for seg in snap.get("segments", []):
                if seg.get("device") != 0:
                    continue
                for blk in seg.get("blocks", []):
                    if blk.get("state") != "active_allocated":
                        continue
                    frames = blk.get("frames") or blk.get("frame") or []
                    # frames is a list of dicts: {filename, name, line}
                    fmt = [
                        f"{f.get('filename','?')}:{f.get('line','?')} in {f.get('name','?')}"
                        for f in frames
                    ]
                    blocks.append({
                        "size_mb": blk["size"] // (1024 * 1024),
                        "size_bytes": blk["size"],
                        "segment_total_mb": seg["total_size"] // (1024 * 1024),
                        "stack": fmt,
                    })
            out["rank0_blocks"] = blocks
            # Also include device_traces — ordered alloc/free events —
            # truncated to last N events for the 180-ish MB size range.
            traces = []
            for dev_traces in snap.get("device_traces", [])[:1]:
                for ev in dev_traces[-500:]:
                    sz = ev.get("size", 0)
                    # Focus on sizes near our 180 MB target and other >= 1 MB.
                    if sz >= 1024 * 1024:
                        frames = ev.get("frames") or []
                        fmt = [
                            f"{f.get('filename','?')}:{f.get('line','?')} in {f.get('name','?')}"
                            for f in frames[:10]
                        ]
                        traces.append({
                            "action": ev.get("action"),
                            "size_mb": sz // (1024 * 1024),
                            "size_bytes": sz,
                            "addr": ev.get("addr"),
                            "stream": ev.get("stream"),
                            "stack_top": fmt[:5],
                        })
            out["rank0_recent_traces"] = traces
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    return app


# ----- helpers -----


def _validate_params(req: GenerateRequest) -> None:
    if req.frame_num < MIN_FRAMES or req.frame_num > MAX_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"frame_num must be in [{MIN_FRAMES}, {MAX_FRAMES}] "
                f"(measured 4090 ceiling)"
            ),
        )
    if (req.frame_num - 1) % 4 != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"frame_num must satisfy 4n+1 (got {req.frame_num}); "
                f"try {req.frame_num - ((req.frame_num - 1) % 4)}"
            ),
        )
    if req.size != "1280*704":
        raise HTTPException(
            status_code=400,
            detail="size is fixed to '1280*704' in this demo",
        )
    if req.sampling_steps < 1 or req.sampling_steps > 200:
        raise HTTPException(
            status_code=400,
            detail="sampling_steps must be in [1, 200]",
        )


def _job_to_response(
    job: Job,
    video_base_url: str,
    store_position: Optional[int],
) -> JobResponse:
    video_url: Optional[str] = None
    if job.status == STATUS_DONE and job.video_path:
        video_url = (
            f"{video_base_url.rstrip('/')}/v1/generations/{job.id}/video"
            f"?token={job.video_token}"
        )
    return JobResponse(
        job_id=job.id,
        status=job.status,
        prompt=job.prompt,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        queue_position=store_position,
        error=job.error,
        video_path=job.video_path,
        video_url=video_url,
        callback_status=job.callback_status,
        callback_attempts=job.callback_attempts or None,
    )


def _gpu_free_mb() -> list[int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            out.append(free // (1024 * 1024))
        return out
    except Exception:
        return []


def _debug_memory() -> dict[str, Any]:
    """Rank-0 torch allocator + nvml snapshot. Used to diagnose across-request leaks."""
    import torch
    out: dict[str, Any] = {}
    try:
        local_rank = 0
        alloc_mb = torch.cuda.memory_allocated(local_rank) // (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(local_rank) // (1024 * 1024)
        peak_alloc_mb = torch.cuda.max_memory_allocated(local_rank) // (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved(local_rank) // (1024 * 1024)
        free_b, total_b = torch.cuda.mem_get_info(local_rank)
        out["rank0_torch"] = {
            "allocated_mb": alloc_mb,
            "reserved_mb": reserved_mb,
            "peak_allocated_mb": peak_alloc_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "nvml_used_mb": (total_b - free_b) // (1024 * 1024),
            "nvml_free_mb": free_b // (1024 * 1024),
            "nvml_total_mb": total_b // (1024 * 1024),
            # The gap between nvml.used and (torch.reserved + ~ 600 MB
            # baseline CUDA context) is "non-torch" memory: NCCL comm
            # buffers, cuBLAS/cuDNN/flash_attn workspaces, kernel code
            # cache. This is what the caching allocator can never free.
            "non_torch_mb": (
                (total_b - free_b) // (1024 * 1024) - reserved_mb
            ),
        }
        # All devices free/used from NVML (cross-process signal).
        dev_info = []
        for i in range(torch.cuda.device_count()):
            f, t = torch.cuda.mem_get_info(i)
            dev_info.append({
                "device": i,
                "used_mb": (t - f) // (1024 * 1024),
                "free_mb": f // (1024 * 1024),
            })
        out["nvml_all"] = dev_info
        # FULL memory_stats dict (rank 0) — helps pin down which pool /
        # counter grows across requests. Keys suffixed .current tell us
        # the live state; .peak tracks high-water-mark since the last
        # reset_peak_memory_stats(). Bytes converted to MB.
        try:
            stats = torch.cuda.memory_stats(local_rank)
            interesting = {}
            for k, v in stats.items():
                if isinstance(v, int):
                    if any(k.startswith(p) for p in (
                        "allocated_bytes", "reserved_bytes", "active_bytes",
                        "inactive_split_bytes", "requested_bytes",
                    )):
                        interesting[k + "_mb"] = v // (1024 * 1024)
                    elif any(k.startswith(p) for p in (
                        "segment.", "active.", "allocation.",
                        "num_alloc_retries", "num_ooms", "num_sync_all_streams",
                        "num_device_alloc", "num_device_free",
                        "oversize_allocations", "oversize_segments",
                    )):
                        interesting[k] = v
            out["rank0_memory_stats"] = interesting
        except Exception:
            pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out
