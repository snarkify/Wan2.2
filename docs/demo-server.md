# Demo API Server

A long-running HTTP service that wraps Wan2.2 TI2V-5B text-to-video
generation behind a small REST API and a Slack callback. Built for
internal demo use on a 4× RTX 4090 box (`gpu6`); not exposed to the
internet.

Start: `bash scripts/run_demo_server.sh` (see env-var checklist near the
top of the script).

## Architecture

```
torchrun --nproc_per_node=4 -m server.main   (single launch, persistent)
│
├── rank 0   FastAPI + uvicorn (port 8000) + asyncio job queue + worker loop
│              accepts POST /v1/generations
│              dispatches each job to all ranks via dist.broadcast_object_list
│              writes mp4 to disk inside the executor
│              fires callback to /v1/slack/callback (if requested)
│
└── ranks 1..3   idle in dist.broadcast_object_list waiting for the next job;
                 participate in the diffusion forward (Ulysses-4 + FSDP).
```

Choices that matter:

- **Cross-rank dispatch via `dist.broadcast_object_list`**: zero new
  dependencies, reuses the existing torch.distributed group that FSDP
  and Ulysses already need. NCCL watchdog timeout bumped to 24 h via
  `init_process_group(timeout=...)` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`
  so idle broadcasts between jobs don't kill the group.
- **Reload pipeline per request**: each job constructs a fresh
  `WanTI2V`, runs once, drops it. Tried `reuse=True` (keep models in
  memory between jobs and shuffle to/from CPU) but FSDP's `to('cpu')`
  doesn't release the FlatParameter storage on a 24 GB card, so it
  OOMed every second job in VAE decode. Reload-per-request adds ~95 s
  of model-load overhead — tolerable for an 11-minute job.
- **SQLite-backed job store** in `WAN_DEMO_OUTPUT_DIR/jobs.db` so
  pending and completed jobs survive restarts. Startup reconciliation:
  in-flight `running` jobs become `failed("server restarted")` (and
  fire their callbacks); `queued` jobs stay queued and the worker
  picks them up on first idle.
- **Backpressure**: `WAN_DEMO_MAX_QUEUE` (default 3) — POST returns 503
  with `Retry-After: 600` when full.
- **Per-job video token**: `secrets.token_urlsafe(32)` stored on the
  job, accepted as `?token=...` on `GET /video/{job_id}` so external
  HTTP clients (Slack workers in particular) can fetch the mp4 without
  the shared bearer.
- **SSRF guard on callback URLs**: blocks loopback / link-local /
  RFC1918 by default. `WAN_DEMO_ALLOW_PRIVATE_CALLBACK=1` overrides for
  the intra-net case where OpenClaw and the demo server share private
  addresses.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/generations` | Bearer | Submit a job (returns `job_id`, `eta_seconds`, `queue_position`) |
| `GET` | `/v1/generations/{id}` | Bearer | Job status |
| `DELETE` | `/v1/generations/{id}` | Bearer | Cancel queued job (409 if running) |
| `GET` | `/v1/generations/{id}/video` | Bearer **or** `?token=…` | Download mp4 |
| `GET` | `/v1/health` | none | Queue depth, GPU free MB, uptime |
| `POST` | `/v1/slack/callback` | Bearer | Internal: receives our own callback, posts to Slack |
| `GET` | `/v1/debug/memory` | Bearer | torch + nvml memory snapshot (rank 0) |
| `POST` | `/v1/debug/force_gc` | Bearer | Run gc + empty_cache + ipc_collect on all ranks |
| `GET` | `/v1/debug/allocations` | Bearer | Live allocation list with sizes |
| `POST` | `/v1/debug/start_history` / `stop_history` | Bearer | Toggle `torch.cuda.memory._record_memory_history` |
| `GET` | `/v1/debug/dump_snapshot` | Bearer | Active blocks with Python stack traces (needs history on first) |

Frame counts are constrained to **`4n+1` in [5, 141]**; size is fixed
to `1280*704`; `eta_seconds` follows `T(N) ≈ 443 + 2.68·N` measured on
the 4090 box (see `docs/profiling-results.md`).

## Slack integration

OpenClaw owns the user-facing Slack interaction (slash command or
`app_mention`). It calls `/v1/generations` with three things in
`callback_metadata`:

```
{
  "channel":   "<event.channel>",   // "C..." channel ID, not "#name"
  "user":      "<event.user>",      // "U..." user ID, not "<@U...>"
  "thread_ts": "<event.thread_ts ?? event.ts>"
}
```

and sets `callback_url=http://gpu6:8000/v1/slack/callback`. Eleven
minutes later the demo server hits its own endpoint which uploads the
mp4 via `files.upload_v2` and posts `:tada: <@user> your video is ready`
in the originating thread. OpenClaw never has to handle the callback,
download the video, or call Slack — it just submits and reports the
ETA.

Required envs on the demo server:

```
WAN_SLACK_BOT_TOKEN=xoxb-…        # the SAME app token OpenClaw uses
WAN_SLACK_ALLOWED_CHANNELS=C…,C…  # comma-separated allow-list (optional)
```

Bot scopes the app must have: `app_mentions:read` (OpenClaw side),
`chat:write`, `files:write`. The bot must be a *member* of every
channel you intend to post into — `files.upload_v2` returns
`channel_not_found` otherwise (no equivalent of `chat:write.public`
exists for file uploads).

OpenClaw skill prompt for this is in PR notes; the operator just sets
the constants (`BASE_URL`, `BEARER`, `CALLBACK`) and pastes it.

## Operational notes

- **Server restart**: `pkill -9 -f 'torchrun.*server'`, wait for GPUs
  to free, then re-run `scripts/run_demo_server.sh`. The launch script
  exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the
  NCCL settings that keep idle broadcasts alive.
- **Watching jobs**: `curl http://gpu6:8000/v1/health` for queue depth;
  `curl http://gpu6:8000/v1/debug/memory` for rank-0 memory; the
  per-job `mem[…]` lines in `WAN_DEMO_OUTPUT_DIR/server.log` show the
  GPU trajectory inside `_build_and_run`.
- **GPUs not freeing on shutdown**: `nvidia-smi --query-compute-apps=pid
  --format=csv,noheader | xargs -r kill -9` after `pkill`.
- **Crashed mid-job**: SQLite reconciliation marks the run failed on
  next boot and fires a callback (so OpenClaw / the user knows).
- **Bot keeps replying "channel_not_found"**: invite the bot to that
  channel: `/invite @<bot-name>`.

## What we hit during development (and what to remember)

These are the lessons that aren't obvious from the code alone.

### 1. Wan2.2 4090 ceiling at 1280×704 is 141 frames

Measured by binary search; 17 → 41 → 81 → 121 → 141 succeed (peak
~23.1 GB on rank 0 — VAE decode workspace, not diffusion). 153 and 161
OOM during VAE decode by ~50 MB. 81 and 141 are reliable defaults; the
server enforces `4n+1, ≤ 141`.

Generation time on the 4090 follows `T ≈ 443 + 2.68·N` seconds within
±2% across the range we measured. This is what `/v1/generations` returns
in `eta_seconds`.

### 2. Make `--convert_model_dtype` actually take effect under FSDP

The original code only applied `model.to(bf16)` on the non-FSDP branch.
With FSDP wrapping, the cast was silently dropped and the FlatParameter
shard stayed in fp32 — that doubles per-rank weight memory and was the
difference between fitting and OOMing on a 24 GB card. Fix in
`wan/textimage2video.py::_configure_model`: cast before `shard_fn`.

### 3. Per-rank random seeds break Ulysses output

`generate.py` already broadcasts the seed from rank 0 via
`dist.broadcast_object_list` precisely because Ulysses needs every
rank to produce the same initial noise tensor — otherwise
`gather_forward` at the end of `sp_dit_forward` concatenates slices
from different noise realizations and the middle of the video comes
out as noise (the edges happen to match because they're on the
sequence boundaries). The demo server initially passed `seed: -1`
through the broadcast unchanged; each rank then independently hit
`random.randint(0, sys.maxsize)` inside `t2v()` and picked its own
value. Fix: resolve `-1` to a concrete int on rank 0 in
`server/worker.py::_build_kwargs` before the message is broadcast.

This was the bug that produced the symptom "first few frames clean,
rest noisy" / "first and last clean, middle noisy" depending on which
ranks happened to align.

### 4. Asynchronous VAE decode + post-decode `empty_cache()` corrupts tail frames

We added `del self.vae; gc.collect(); torch.cuda.empty_cache()`
immediately after `videos = self.vae.decode(x0)` to free VAE weights
early. But VAE decode enqueues kernels asynchronously and returns to
Python before they complete. `empty_cache()` interleaving with the
in-flight kernels showed up as noisy tail frames. Lesson: **never run
`empty_cache` between an async tensor-producing call and the eventual
sync (`.cpu()` etc.) on its result**. Removed; let the caller's
pipeline teardown handle VAE release after `save_video` has finished
the `.cpu()` transfer.

### 5. FSDP's unsharded all-gather buffer leaks across requests

The big debugging arc. After fixing the obvious things (`del self.vae`,
the video tensor crossing `loop.run_in_executor`, etc.) we still saw
~170 MB / job permanent growth on every rank. `torch.cuda.memory._snapshot()`
with `_record_memory_history()` traced the leaked allocation directly to:

```
torch/distributed/fsdp/_flat_param.py:1381 in _alloc_padded_unsharded_flat_param
torch/distributed/fsdp/_flat_param.py:1354 in unshard
torch/distributed/fsdp/_runtime_utils.py:417 in _pre_forward_unshard
```

This is the `_full_param_padded` buffer FSDP uses as the destination
for all-gather during forward. In eager mode it's normally released
by `_reshard` after each layer, but the FlatParameter object retains
a Python reference to the storage. Under our eval-mode + reuse=False
teardown, that reference survived and the storage stuck around.

`wan/distributed/fsdp.py::free_model` originally only called
`_free_storage(handle.flat_param.data)` — i.e. the *sharded* shard.
We extended it to also walk the unsharded-buffer attributes
(`_full_param_padded`, `_full_prec_full_param_padded`,
`_padded_unsharded_flat_param`, `_full_unsharded_flat_param`) and
`_free_storage` each. After the fix, `mem[driver_after_save_cleanup]`
shows `torch.alloc=8 MB` between jobs, identical across consecutive
runs — the only residual is the GPU-sampler thread's own state.

The forensic path: instrumented per-stage memory snapshots in
`server/worker.py::_mem_snapshot`; `/v1/debug/memory` for live
inspection; `/v1/debug/force_gc` to distinguish "Python-held vs
CUDA-lib-held"; `/v1/debug/allocations` for live block sizes; and
finally `/v1/debug/start_history` + `/v1/debug/dump_snapshot` for
the stack trace that named the culprit.

### 6. NCCL watchdog kills idle process groups

Default NCCL watchdog timeout is 10 minutes. While the demo server is
idle between jobs, ranks 1..3 sit in `dist.broadcast_object_list` —
which NCCL counts as a pending collective. Past 10 minutes of idle the
watchdog SIGABRTs the process group, killing the whole server. Fix:
`init_process_group(timeout=timedelta(hours=24))` plus
`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` in the launch script. Long-term a
cleaner solution is a separate gloo-backed group for control-plane
broadcasts, but it isn't worth the churn for this demo.

### 7. FSDP doesn't release FlatParameter storage on `del self.model`

Plain `del + gc.collect() + empty_cache()` left ~5 GB resident per GPU
because FSDP's internal handles still pointed at the storage. The
project already had a `wan/distributed/fsdp.py::free_model` helper for
this — it walks each FSDP submodule and calls `_free_storage` on the
flat_param. The original code didn't use it; the demo server now does.
Same fix pattern as #5 (extended to also clear unsharded buffers).

### 8. Things that turned out NOT to be the root cause

Documenting these so future debugging doesn't waste time on them:

- "Future / executor pinning the result tensor" — we restructured
  `_build_and_run` to save the video inside the executor instead of
  returning the tensor. Cleaner code, but had no measurable effect on
  the leak; the FSDP unshard buffer was the real culprit.
- "cuBLAS / cuDNN / flash-attn workspace caches" — we tested
  `torch._C._cuda_clearCublasWorkspaces` and `torch.cuda.ipc_collect`;
  they recover < 20 MB total and don't grow per job once stabilized.
- "NCCL communicator buffers accumulate per FSDP re-init" — would have
  been linear like the symptom suggested, but the actual size of the
  alloc (180 MB) was wrong for NCCL buffers, and the snapshot stack
  pointed at FSDP, not NCCL.

Lesson reinforced: the user was right to push past the easy
"acceptable for a demo, restart every 50 jobs" answer. The data was
available all along (`torch.cuda.memory._snapshot` with `frames`)
and the actual fix was 30 lines once we stopped guessing.

## Branch layout

- `profiling-framework`: profiling + tracing infra (CSV, Chrome trace,
  GPU sampler, NCCL stats, cProfile hook). Predates the demo server.
- `demo-server` (this work): adds `server/` package + the FSDP and
  Slack changes above. All commits in `git log demo-server`.

The 4090 leak fixes (FSDP unshard buffer, seed broadcast, async-VAE
ordering) are in the `demo-server` branch. They benefit
non-server callers too — anyone running `python generate.py
--dit_fsdp --ulysses_size 4` would have hit the seed bug and the
leak, just less visibly because the process exits at the end.

## Files added by this branch

```
server/
  __init__.py
  app.py            FastAPI routes, /v1/slack/callback, /v1/debug/*
  callbacks.py      generic webhook delivery (HMAC, retries, SSRF guard)
  config.py         env-var loading, ETA formula, frame-count constraints
  jobstore.py       SQLite-backed Job + JobStore + reconciliation
  main.py           torchrun entrypoint, rank dispatch
  slack.py          Slack chat.postMessage + files.upload_v2 helpers
  worker.py         dispatch loop, memory instrumentation, force_gc broadcast
scripts/
  run_demo_server.sh   torchrun launch wrapper with all env vars
docs/
  demo-server.md       this file
```
