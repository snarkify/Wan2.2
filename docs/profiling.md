# Profiling & Tracing Framework

A zero-overhead profiling framework for Wan2.2 video generation pipelines, adapted from the [jolt-cpp-harness profiling design](https://snarkify.github.io/jolt-cpp-harness/profiling-tracing-design.html).

## Quick Start

```bash
# Single GPU with profiling
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --frame_num 17 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --t5_cpu --offload_model True \
  --profile_dir ./profile_output

# Multi-GPU (requires 80GB+ GPUs like A100/H100)
torchrun --nproc_per_node=4 generate.py \
  --task t2v-A14B \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp --t5_cpu --ulysses_size 4 \
  --profile_dir ./profile_output

# Post-processing
python scripts/profiling/aggregate_csv.py ./profile_output
python scripts/profiling/merge_traces.py ./profile_output
python scripts/profiling/summarize.py ./profile_output

# View timeline in browser
# Open profile_output/trace_rank0.json (or merged_trace.json) at https://ui.perfetto.dev
```

## Architecture

Three independent measurement layers, all gated by `--profile_dir` / `WAN_PROFILE_DIR`. When disabled, overhead is ~50-100ns per span (single boolean check + no-op return).

### Layer 1: Stopwatch (CSV timing records)

Wall-clock + CUDA event GPU timing per code region. Produces append-only CSV files:

```
run_id,rank,name,step,wall_ms,gpu_ms,timestamp
cc129d54e473,0,text_encoding,-1,198835.460,198835.234,1776252612.428502
cc129d54e473,0,step,0,1145.230,1145.100,1776252815.712
cc129d54e473,0,model_forward_cond,0,410.920,564.200,1776252815.713
...
```

Per-rank output files (`timing_rank0.csv`, `timing_rank1.csv`, ...).

### Layer 2: Chrome Trace Format Timeline

Streaming JSON events for visualization in [Perfetto](https://ui.perfetto.dev). Supports Begin/End spans, Counter events (GPU memory), and Instant markers. Per-rank files, mergeable via post-processing.

### Layer 3: PyTorch Profiler (optional)

Wraps `torch.profiler.profile` for kernel-level CUPTI tracing. Opt-in via `WAN_PROFILE_TORCH=1`. Exports Chrome Trace Format that can be merged with Layer 2 traces for a unified Perfetto view showing both application-level spans and GPU kernel activity.

## Configuration

### Primary interface

```bash
python generate.py --profile_dir ./profile_output ...
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WAN_PROFILE_DIR` | (unset) | Fallback if `--profile_dir` not given. Enables profiling when set. |
| `WAN_PROFILE_RUN_ID` | auto-generated | Tag for multi-run benchmarks |
| `WAN_PROFILE_TRACE` | `1` | Enable Chrome Trace output |
| `WAN_PROFILE_TORCH` | `0` | Enable torch.profiler (heavyweight) |
| `WAN_PROFILE_SYNC` | `1` | Force `cuda.synchronize` on macro spans |
| `WAN_PROFILE_MODEL_DETAIL` | `0` | Per-transformer-block spans (adds ~10K spans) |

## Instrumentation Strategy

The framework uses a **hybrid approach**: PyTorch hooks for module-level spans (zero pipeline edits) and minimal manual spans for loop-level structure that hooks cannot capture.

### Automatic spans (via hooks)

These are registered by `setup_profiling(pipeline)` and require no edits to pipeline files:

| Span | Hook Type | Modules |
|---|---|---|
| `model_forward/{attr}` | `register_forward_hook` | WanModel, WanModel_S2V, WanAnimateModel |
| `text_encoding` | `register_forward_hook` | T5EncoderModel |
| `vae_encode` | `functools.wraps` | Wan2_1_VAE, Wan2_2_VAE (not nn.Module) |
| `vae_decode` | `functools.wraps` | Wan2_1_VAE, Wan2_2_VAE |

### Manual spans (via StepProfiler)

~6-8 lines per pipeline, wrapping the diffusion loop:

```python
from wan.profiling import profiled_loop

with profiled_loop() as loop:
    for step_idx, t in enumerate(tqdm(timesteps)):
        with loop.step(step_idx) as spans:
            with spans.span("model_forward_cond"):
                noise_pred_cond = model(x, t=timestep, **arg_c)[0]
            with spans.span("model_forward_uncond"):
                noise_pred_uncond = model(x, t=timestep, **arg_null)[0]
            with spans.span("guidance_merge"):
                noise_pred = uncond + scale * (cond - uncond)
            with spans.span("scheduler_step"):
                x0 = scheduler.step(...)
```

### Span hierarchy

```
pipeline_generate
|-- text_encoding              (hook)
|-- vae_encode                 (hook, I2V/S2V/Animate only)
|-- model_forward/{attr}       (hook, fires on every forward call)
|-- diffusion_loop             (profiled_loop)
|   +-- step                   (loop.step)
|       |-- model_prepare      (spans.span, T2V/I2V only)
|       |-- model_forward_cond (spans.span)
|       |-- model_forward_uncond (spans.span, skipped when guide_scale <= 1)
|       |-- guidance_merge     (spans.span, skipped when guide_scale <= 1)
|       +-- scheduler_step     (spans.span)
|-- vae_decode                 (hook)
+-- gpu_memory counters        (at init, per-step, end)
```

### Optional model-level detail

Set `WAN_PROFILE_MODEL_DETAIL=1` to add spans inside `WanModel.forward()`:

```
patch_embedding, time_embedding, text_embedding,
block_0, block_1, ..., block_31,
head, unpatchify
```

This adds ~100 spans per forward call x 2 (cond+uncond) x 50 steps = 10,000 spans per generation.

## GPU Timing Strategy

CUDA operations are asynchronous. The framework uses two timing strategies to avoid pipeline stalls:

| Span Level | Strategy | Sync Cost |
|---|---|---|
| Top-level (text_encoding, vae_decode) | Wall-clock + CUDA events, sync in `__exit__` | Negligible (few times) |
| Loop-interior (step, model_forward_*, etc.) | CUDA events, **deferred batch resolution** | Near-zero |

**Deferred resolution**: Inside `profiled_loop()`, all spans push CUDA start/end event pairs to a global buffer without synchronizing. At the loop exit, one `torch.cuda.synchronize()` resolves all pending events in batch. This avoids ~200 sync points per generation.

Hooks are also loop-aware: when inside `profiled_loop()`, hooks automatically use deferred mode.

## Post-Processing Scripts

### Aggregate CSV

```bash
python scripts/profiling/aggregate_csv.py ./profile_output
```

Reads `timing_rank*.csv`, filters warmup (step 0), computes median/mean/std/p95 per span, outputs `summary.csv` and a formatted table.

### Merge Traces

```bash
python scripts/profiling/merge_traces.py ./profile_output
```

Merges per-rank `trace_rank*.json` files (and optionally `torch_chrome_trace_rank*.json` from Layer 3) into a single `merged_trace.json` for Perfetto. Uses `__trace_sync__` events for cross-rank timestamp alignment.

### Summarize

```bash
python scripts/profiling/summarize.py ./profile_output
```

Generates a human-readable markdown report: total time, per-phase breakdown (%), per-step statistics, peak memory.

## Example Profiling Results

Collected on RTX 4090 (24GB), TI2V-5B model, 1280x704, 17 frames, 50 steps:

```
Total generation time: 476s

Phase Breakdown:
  text_encoding     397s  (83.4%)  -- T5-XXL on CPU (--t5_cpu)
  diffusion_loop     57s  (12.0%)
  vae_decode          5s   (1.1%)

Per-Step (median, excluding warmup):
  step                1,130ms wall
  model_forward_cond    411ms wall /  564ms GPU
  model_forward_uncond  565ms wall /  565ms GPU
  scheduler_step        153ms wall /    1ms GPU
  guidance_merge          1ms wall /    0ms GPU

Peak GPU memory: 23,022 MB
```

Key observations:
- T5 on CPU dominates (83%) due to CPU-bound text encoding. Moving T5 to GPU would shift the bottleneck to the diffusion loop.
- `model_forward_cond` wall time (411ms) < GPU time (564ms) because the CPU returns from the async kernel launch before the GPU finishes. The deferred CUDA events capture the true GPU time.
- `scheduler_step` is 153ms wall but only 1ms GPU — it's a CPU-bound solver computation.

## Hardware Requirements & Caveats

### GPU memory requirements by model

| Model | Params | Min Single-GPU VRAM | Notes |
|---|---|---|---|
| TI2V-5B | 5B (1 model) | ~24GB | Fits on RTX 4090 at low frame counts (17 frames). OOM at 41+ frames on 24GB. |
| T2V-A14B | 14B x 2 models | ~80GB | Requires A100/H100. Does NOT fit on RTX 4090/5090 (28GB model > 24/32GB card). |
| I2V-A14B | 14B x 2 models | ~80GB | Same as T2V-A14B. |
| S2V-14B | 14B (1 model) | ~80GB | Untested on consumer GPUs. |

### Known limitations

1. **14B models on consumer GPUs**: The T2V/I2V 14B models store each 14B DiT as ~28GB in bf16. This exceeds the 24GB of RTX 4090 and the 32GB of RTX 5090 even with model offloading, because the full model must be on GPU during forward pass. FSDP also fails because it loads the full model to GPU before sharding.

2. **RTX 5090 (32GB) estimates**: A single 5090 could fit 28GB model params but leaves only ~4GB for activations — likely insufficient. 2x RTX 5090 with FSDP should work (14GB model per card, 18GB headroom). 4x RTX 5090 would be comfortable for full resolution.

3. **T5 on CPU**: Using `--t5_cpu` avoids GPU memory pressure from the 11GB T5-XXL encoder but makes text encoding very slow (200s per encoding on CPU vs. seconds on GPU). Profile data will be dominated by this phase.

4. **VAE classes are not nn.Module**: `Wan2_1_VAE` and `Wan2_2_VAE` are plain wrapper classes. The framework uses `functools.wraps` method decoration instead of `register_forward_hook`.

5. **Wan 2.1 VAE in Wan 2.2 repo**: The 14B models (T2V, I2V, S2V, Animate) use `Wan2.1_VAE.pth`. Only TI2V-5B uses the newer `Wan2.2_VAE.pth` with z_dim=48. This is intentional — the 14B models were trained with the 2.1 VAE.

6. **Profiling overhead inside loops**: Hook-created spans inside `profiled_loop()` automatically use deferred CUDA event resolution. Outside the loop (e.g., text_encoding, vae_decode), spans synchronize immediately.

## File Structure

```
wan/profiling/
    __init__.py              # Public API: trace_span, profiled_loop, StepProfiler, flush
    _config.py               # Lazy ProfileConfig from --profile_dir / env vars
    _event_buffer.py         # Global CUDA event buffer, deferred batch resolution
    _stopwatch.py            # CudaTimedSpan + StopwatchRecorder (CSV)
    _tracer.py               # Chrome Trace Format streaming JSON writer
    _torch_profiler.py       # Optional torch.profiler wrapper
    _hooks.py                # Auto-discovery hook registration
    _memory.py               # GPU memory snapshots

scripts/profiling/
    aggregate_csv.py         # Warmup filtering, median/p95 aggregation
    merge_traces.py          # Merge per-rank + torch.profiler traces
    summarize.py             # Human-readable markdown report
```

## Modified Files

- `generate.py` — `--profile_dir` argument, `setup_profiling()`, `trace_span("pipeline_generate")`, `record_memory()`, `flush()`
- `wan/text2video.py` — StepProfiler loop instrumentation
- `wan/image2video.py` — StepProfiler loop instrumentation
- `wan/speech2video.py` — StepProfiler loop + multi-clip span
- `wan/textimage2video.py` — StepProfiler in both `t2v()` and `i2v()` methods
- `wan/animate.py` — StepProfiler loop + clip counter for `while True` loop
- `wan/modules/model.py` — Optional model-detail spans (gated by `WAN_PROFILE_MODEL_DETAIL=1`)
