# Demo API Server

A long-running HTTP service that wraps Wan2.2 TI2V-5B text-to-video
generation behind a small REST API and a Slack callback. Built for
internal demo use on a 4× RTX 4090 box (`gpu6`); not exposed to the
internet.

Start: `bash scripts/run_demo_server.sh` (see env-var checklist near the
top of the script).

## Architecture

```
torchrun --nproc_per_node=1 -m server.main   (single launch, persistent)
│
└── rank 0   FastAPI + uvicorn (port 8000) + asyncio job queue + worker loop
              accepts POST /v1/generations
              builds a WanTI2V singleton on first job (~86 s) and reuses
              it for the lifetime of the process (warm path)
              writes mp4 to disk inside the executor
              fires callback to /v1/slack/callback (if requested)
```

The default is now 1 GPU + warm pipeline + fp8. Set `WAN_DEMO_GPUS=4`
to fall back to the legacy 4-GPU FSDP+Ulysses path (see "Why 1 GPU
beat 4" below); when GPUS>1, ranks 1..N-1 sit in
`server.worker.worker_client_loop` waiting for `dist.broadcast_object_list`
broadcasts and participate in the diffusion forward.

Choices that matter:

- **1 GPU + warm pipeline (default)**: 288 s/job warm, 374 s for the
  first job after server start. The 4-GPU FSDP+Ulysses path runs at
  649 s/job — the diffusion forward gets faster with more GPUs but
  per-job FSDP wrap/unflatten + cross-rank communication eats the win,
  and FSDP's unsharded buffers leak across requests (forcing the old
  reload-per-request workaround that paid 95 s of model-load every
  job). With `WanTI2V.__init__`'s world-size-1 guard skipping the
  FSDP wrap, we can keep the pipeline live across requests safely.
  Net: 2.25× speedup vs the old default. See bench summary below.
- **fp8 quantization (default)**: `WAN_DEMO_QUANT=fp8` casts every
  transformer-block `nn.Linear` weight to `torch.float8_e4m3fn` and
  replaces forward with `torch._scaled_mm`. Mirrors ComfyUI's
  `fp8_e4m3fn_fast` path. ~11% per-step speedup on Ada (4090) tensor
  cores. Set `WAN_DEMO_QUANT=bf16` to disable; the speedup is real but
  fp8 is a lossy cast — visual quality is comparable to bf16 on the
  prompts we tested but not byte-identical.
- **Cross-rank dispatch via `dist.broadcast_object_list`** (used only
  when `WAN_DEMO_GPUS>1`): zero new dependencies, reuses the existing
  torch.distributed group that FSDP and Ulysses already need. NCCL
  watchdog timeout bumped to 24 h via `init_process_group(timeout=...)`
  and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` so idle broadcasts between
  jobs don't kill the group.
- **Module-level pipeline singleton in `server/worker.py`**: built
  lazily on the first job, reused across all subsequent jobs. Pipeline
  ownership lives in the worker module, not in `Config` or `JobStore`
  — those are env state and job state respectively; the pipeline is
  worker state. T5 still offloads to CPU after each encode (built-in
  to `WanTI2V.generate`); DiT moves to CPU between requests via
  `offload_model=True` so VAE decode has the 24 GB it needs for the
  141-frame ceiling.
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
to `1280*704`; `eta_seconds` follows `T(N) ≈ 191 + 1.20·N` warm (1 GPU
+ fp8) plus a one-time `+86 s` cold-start bonus while the singleton is
still being built. The 141-frame ceiling holds at world=1: VAE decode
is rank-0-only either way, so the OOM boundary doesn't move with GPU
count.

## Bench: 1-GPU + warm + fp8 vs. 4-GPU FSDP+Ulysses

Single 4090 (gpu6), TI2V-5B, 81 frames, 1280×704, 50 steps,
`offload_model=True`, prompt "A cat walking in a sunlit meadow", same
seed. Three back-to-back generations from a single Python process so
gen 1 covers cold-kernel-compile + warmup and gens 2–3 are warm
steady-state. Raw artifacts on gpu6 at
`bench_artifacts/stats_warm_fp8_v2.json` and
`bench_artifacts/run_warm_fp8_v2.log`.

| config                          | gen 1 | gen 2 | gen 3 | warm mean | peak alloc |
|---------------------------------|------:|------:|------:|----------:|-----------:|
| 1 GPU + fp8 + warm              | 288 s | 289 s | 288 s |   288 s   |   23.1 GB  |
| 4 GPU + bf16 + reload-per-job   |       |       |       |   649 s   |            |

Pipeline constructor: 86 s (one-time, paid by the first job after
server start). Steady-state per-step: 5.75 s/step.

The 4-GPU number is from the previous production path measured in
`docs/profiling-results.md`. The speedup comes from three things
stacking: skipping FSDP wrap/unflatten + MixedPrecision overhead at
world=1, reusing the loaded pipeline across jobs, and fp8 `_scaled_mm`
on the transformer blocks. fp8 alone bought ~11%; the rest is the
warm + 1-GPU restructuring.

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

### 9. The 4-GPU path was the wrong default

Most of the bug history above (#5, #7, the per-request reload in #2)
exists because we were running FSDP+Ulysses across 4 GPUs and paying
the FSDP overhead for it. Once we benchmarked the 1-GPU path with a
warm pipeline + fp8 on a single 4090, it landed at 288 s/job vs the
4-GPU path's 649 s — 2.25× faster. The 4-GPU diffusion forward IS
faster than 1-GPU per step, but per-job FSDP wrap/unflatten +
cross-rank communication overwhelms the diffusion savings, and it's
the FSDP wrap that creates all the across-request leak surface area
in the first place.

The world-size-1 guard in `wan/textimage2video.py::_configure_model`
demotes `dit_fsdp` and `use_sp` to `False` when there's only one
process. That removes the FSDP code path entirely (so the leaks in #5
and #7 can't recur) and lets `server/worker.py` keep the pipeline
alive across requests as a module-level singleton. The 4-GPU path is
still reachable via `WAN_DEMO_GPUS=4` for anyone who wants it; the
old reload-per-request behavior in #2 is the right call there.

fp8 quantization (`WAN_DEMO_QUANT=fp8`, default) adds another ~11%
on top via `torch._scaled_mm` on Ada tensor cores. fp8_e4m3fn weights
survive the CPU↔GPU round-trip cleanly (validated across 3
back-to-back gens with `offload_model=True` — peak alloc stays at
23.1 GB, no drift), so it composes with the warm-singleton path.

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
