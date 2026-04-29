# Demo Server Runbook (gpu6)

Operational tasks: how to start, stop, restart, inspect, and recover the
Wan2.2 demo server. For architecture, design, and bench numbers see
[`demo-server.md`](demo-server.md).

The server runs on `gpu6` in a `tmux` session called `demo-server`,
listening on `:8000`. Token: `demo-secret` (development; change for any
non-internal use).

## First-time setup on a fresh box

The launcher reads its config from environment variables. Rather than
typing them on every restart (and forgetting one — see [Common gotcha
#1](#1-slack-callbacks-fail-with-http-500)), keep them in a single
gitignored file at the repo root.

```bash
ssh gpu6
cd ~/boyu/Wan2.2
cp .env.example .env.demo-server
chmod 600 .env.demo-server
$EDITOR .env.demo-server         # fill in the secrets
```

`.env.example` lists every variable the launcher reads, with a brief
note on each. The `.gitignore` protects `.env.*` (with an explicit
`!.env.example` exception so the template stays tracked).

Required values for our gpu6 deployment:

```
WAN_DEMO_TOKEN=demo-secret
WAN_DEMO_CKPT_DIR=/home/ubuntu/boyu/Wan2.2/Wan2.2-TI2V-5B
WAN_DEMO_OUTPUT_DIR=/home/ubuntu/boyu/Wan2.2/demo_output
WAN_DEMO_ALLOW_PRIVATE_CALLBACK=1
WAN_SLACK_BOT_TOKEN=xoxb-…
WAN_SLACK_ALLOWED_CHANNELS=C070HJ0QP1U,C0ARZCSRXM0,C048K9W8X1V
```

Defaults for `WAN_DEMO_GPUS=1` and `WAN_DEMO_QUANT=fp8` are baked into
`scripts/run_demo_server.sh`; override only if you want the legacy
4-GPU FSDP+Ulysses path or full bf16 precision.

## Start

```bash
ssh gpu6 'tmux new-session -d -s demo-server -x 200 -y 50 \
  "cd /home/ubuntu/boyu/Wan2.2 && \
   source .venv/bin/activate && \
   set -a && source .env.demo-server && set +a && \
   bash scripts/run_demo_server.sh 2>&1 | tee demo_output/server.log; \
   sleep 86400"'
```

`set -a … set +a` auto-exports every variable read from the env file —
no need to keep the launcher in sync as new vars are added.

The trailing `sleep 86400` keeps the tmux pane alive for a day after
the launcher exits, so you can read the last log lines on a crash
instead of finding the session already gone.

**First job after start pays the 86 s pipeline-ctor cold start** (T5
+ VAE + DiT load + fp8 conversion). Every subsequent job is warm
(~290 s for 81 frames).

## Verify it came up

```bash
ssh gpu6 'curl -s http://localhost:8000/v1/health | python3 -m json.tool'
```

Healthy output:

```json
{
  "model_loaded": true,
  "queue_size": 0,
  "active_jobs": 0,
  "uptime_s": 12.3,
  "gpu_free_mb": [23656, 23718, 23718, 23718]
}
```

Watch the live log in tmux:

```bash
ssh gpu6 tmux attach -t demo-server   # Ctrl-b d to detach
```

Or tail without attaching:

```bash
ssh gpu6 tail -f /home/ubuntu/boyu/Wan2.2/demo_output/server.log
```

## Stop

**Always check for running jobs first.** A kill mid-generation marks
the job `failed("server restarted")` on next startup but the user (or
OpenClaw) has to resubmit.

```bash
ssh gpu6 'curl -s http://localhost:8000/v1/health | python3 -m json.tool'
# active_jobs should be 0 before you proceed
ssh gpu6 'tmux kill-session -t demo-server'
```

Or, if the user can wait, drain the queue first by pausing submissions
upstream and letting `active_jobs` reach 0.

## Restart cleanly

Same dance: stop → start. The pipeline gets re-built (86 s warmup)
and any in-flight job at kill time is reconciled to `failed` on
startup. `queued` jobs survive the restart and resume.

A bash one-liner if you're doing a no-op restart:

```bash
ssh gpu6 'tmux kill-session -t demo-server 2>/dev/null; sleep 2; \
  tmux new-session -d -s demo-server -x 200 -y 50 \
  "cd /home/ubuntu/boyu/Wan2.2 && source .venv/bin/activate && \
   set -a && source .env.demo-server && set +a && \
   bash scripts/run_demo_server.sh 2>&1 | tee demo_output/server.log; \
   sleep 86400"'
```

## Inspect state

### Recent jobs

```bash
ssh gpu6 'cd /home/ubuntu/boyu/Wan2.2 && .venv/bin/python -c "
import sqlite3
conn = sqlite3.connect(\"demo_output/jobs.db\")
for r in conn.execute(\"SELECT substr(id,1,8), status, callback_status, substr(prompt,1,60) FROM jobs ORDER BY created_at DESC LIMIT 10\"):
    print(r)
"'
```

### Full record for one job

```bash
JOB=606e3786-be89-4acf-8d55-b1605ee2d0fc
ssh gpu6 "curl -s http://localhost:8000/v1/generations/$JOB \
  -H 'Authorization: Bearer demo-secret' | python3 -m json.tool"
```

### Memory + GPU snapshot

```bash
ssh gpu6 'curl -s http://localhost:8000/v1/debug/memory \
  -H "Authorization: Bearer demo-secret" | python3 -m json.tool'
ssh gpu6 nvidia-smi
```

## Submit a test job

```bash
ssh gpu6 'curl -s -X POST http://localhost:8000/v1/generations \
  -H "Authorization: Bearer demo-secret" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"a cat walking in a sunlit meadow\", \"frame_num\": 81}"'
```

Returns `{"job_id": "...", "status": "queued", "queue_position": 0, "eta_seconds": 374}`.

Poll status:

```bash
ssh gpu6 'curl -s http://localhost:8000/v1/generations/<id> \
  -H "Authorization: Bearer demo-secret"'
```

When `status=done`, download:

```bash
ssh gpu6 'curl -s http://localhost:8000/v1/generations/<id>/video \
  -H "Authorization: Bearer demo-secret" -o /tmp/out.mp4'
```

## Common gotchas

### 1. Slack callbacks fail with HTTP 500

Symptom: `callback_status=failed`, `callback_last_error=HTTP 500` in
the jobs table. Server log shows `RuntimeError: WAN_SLACK_BOT_TOKEN
is not set; cannot post to Slack` from `server/slack.py:_token`.

Cause: server was restarted without sourcing `.env.demo-server` (or
the env file is missing `WAN_SLACK_BOT_TOKEN`). The diffusion
finishes, the callback handler tries to upload to Slack, fails
because the token isn't set, the demo-API retry loop tries 4 times
(2 s, 10 s, 60 s backoff) then gives up.

Fix: confirm `.env.demo-server` has the token, restart the server
with the `set -a; source …; set +a` pattern, manually re-fire any
orphaned callbacks (next gotcha).

### 2. Re-fire a stranded callback

Job is `done`, mp4 is on disk, but Slack never got it because the
callback retries failed (e.g. gotcha #1 above).

```bash
JOB=606e3786-be89-4acf-8d55-b1605ee2d0fc
ssh gpu6 "cd /home/ubuntu/boyu/Wan2.2 && .venv/bin/python -c '
import sqlite3, json, sys
conn = sqlite3.connect(\"demo_output/jobs.db\")
row = conn.execute(\"SELECT prompt, callback_metadata_json FROM jobs WHERE id=?\", (\"$JOB\",)).fetchone()
print(json.dumps({\"job_id\": \"$JOB\", \"status\": \"done\", \"prompt\": row[0], \"metadata\": json.loads(row[1])}))
' > /tmp/payload.json
curl -s -X POST http://localhost:8000/v1/slack/callback \
  -H 'Authorization: Bearer demo-secret' \
  -H 'Content-Type: application/json' \
  -d @/tmp/payload.json"
```

A successful re-post returns `{"ok": true, "files": [...]}`. The
job's `callback_status` in the DB stays `failed` (we don't update
it from the manual path) — that's only a record of the original
auto-retry failure.

### 3. OOM mid-job

Symptom: server log shows `torch.OutOfMemoryError`, job ends
`failed`. Common at `frame_num > 141` or if another process is
holding GPU 0.

Diagnosis:

```bash
ssh gpu6 nvidia-smi
ssh gpu6 'curl -s http://localhost:8000/v1/debug/memory -H "Authorization: Bearer demo-secret" | python3 -m json.tool'
```

If another process is on GPU 0 (rare; we set `CUDA_VISIBLE_DEVICES=0`
implicitly via the launcher), kill it. Otherwise the request was
just too big — drop `frame_num` toward the validated 81-frame point.

### 4. Pipeline ctor takes much longer than 86 s

Cold T5 load from disk is the long pole (`models_t5_umt5-xxl-enc-bf16.pth`,
~11 GB, sequential `torch.load`). On first boot after a system
reboot, the page cache is cold and load can take 2–3 min instead
of ~10 s. That's normal — the next restart benefits from page cache
and lands at the bench number (~86 s).

### 5. Server is running but tmux session is gone

The launcher detaches `torchrun` cleanly when its parent shell exits
on a non-tmux session. If `tmux ls` shows no `demo-server` but
`pgrep -f server.main` shows it running, the process is fine; just
the foreground log capture is lost. Either let it finish in-flight
jobs and restart, or attach to the orphaned process via
`/proc/<pid>/fd/1` for live logs.

## Where to look when things break

| Symptom | First place to look |
|---|---|
| 500 from `/v1/slack/callback` | `demo_output/server.log` for `RuntimeError`, then `.env.demo-server` for the missing var |
| Job stuck in `running` after restart | jobs.db `error` column should say `server restarted`; reconciliation may have failed if the DB was open by another process. Restart again. |
| `model_loaded:false` on `/v1/health` | Pipeline ctor failed. Check log for `Traceback` near `_get_or_build_pipeline` or `WanTI2V.__init__`. |
| Slack `channel_not_found` | Bot isn't a member of the target channel. `/invite @bot` in Slack. |
| ETA way off | `eta_seconds` formula is fit to the warm 1-GPU + fp8 path; if you flipped to `WAN_DEMO_QUANT=bf16` or `WAN_DEMO_GPUS=4`, the formula is wrong but the server still works. |

## Related docs

- [`demo-server.md`](demo-server.md) — architecture, API, bug history,
  and bench tables. Read this first if you're new to the codebase.
- [`profiling-results.md`](profiling-results.md) — the underlying
  performance investigation that motivated the 1-GPU + warm + fp8
  default.
- `.env.example` (repo root) — the env-var checklist. Copy to
  `.env.demo-server` on the box and fill in.
