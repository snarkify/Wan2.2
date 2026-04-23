# Profiling Results — 4x RTX 5090 (32GB)

## Models Tested

### TI2V-5B (Text/Image-to-Video, 5B params)

**Works.** Single GPU, no FSDP needed.

| Config | Frames | Resolution | Result | Time |
|---|---|---|---|---|
| `--offload_model True` | 81 | 1280×704 | Success | ~4.5 min |
| `--offload_model True` | 121 | 1280×704 | OOM | — |

The 5B model (single DiT, ~10GB bf16) fits on one 32GB GPU alongside T5 (~11GB) using offload_model to swap them sequentially. 81 frames at 720p is the practical maximum — 121 frames OOMs during a linear algebra operation.

**Command:**
```bash
python generate.py --task ti2v-5B --size 1280*704 --frame_num 81 \
  --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True \
  --profile_dir ./profile_output \
  --prompt "..."
```

### T2V-A14B (Text-to-Video, 14B params × 2 DiTs)

**Does not work on 32GB GPUs.** Explored multiple approaches:

| Approach | What happened |
|---|---|
| FSDP1 (`--dit_fsdp --t5_fsdp`) | OOM during init: `_full_param_padded` pre-allocates full model size (26.6GB) on top of shards (6.6GB) = 33GB > 31.4GB |
| No FSDP, `--offload_model True --convert_model_dtype` | Works at 5 frames only. Each GPU holds full 26.6GB model, ~4GB left for activations |
| FSDP2 (`fully_shard`) | Loading works (13.78GB for two sharded DiTs). 17 frames generates successfully. 81 frames OOMs during forward — FSDP2 all-gathers ~23 transformer blocks simultaneously despite `reshard_after_forward=True`, accumulating 15.66GB of unsharded params |

The 14B model stores two DiTs (low_noise + high_noise, 26.6GB each in bf16). FSDP2 was the most promising — model sharding works correctly, but the forward pass all-gather behavior needs investigation. This work is preserved on the `fsdp2-14b-32gb` branch. See `docs/fsdp2-32gb-status.md` for the full investigation.

## Collected Traces

### TI2V-5B, 81 frames, 1280×704, RTX 5090

**Location**: `~/Downloads/wan2.2_ti2v5b_81f_v4/` (with detailed trace guide at `trace_guide.md`)

Open `trace_rank0.json` in [Perfetto UI](https://ui.perfetto.dev).

#### End-to-end timeline (~241s)

```
pipeline_init (63s)
├── load_t5        61s   T5-XXL from .pth (torch.load, 11GB, slow)
├── load_vae        2s   Wan2.2 VAE (0.5GB)
└── load_dit       <1s   5B DiT from safetensors (memory-mapped, instant)

pipeline_generate (241s)
├── t5_to_gpu       6s   PCIe transfer, 11GB CPU→GPU
├── text_encoding  <1s   Two T5 forward passes (prompt + negative)
├── t5_free        <1s   del + gc (was 10s with old CPU offload)
├── dit_to_gpu      5s   PCIe transfer, 10GB CPU→GPU
├── diffusion_loop 213s  50 denoising steps
│   └── step_N      4.3s each (median, excluding warmup)
│       ├── model_forward_cond   1.6s wall / 2.1s GPU
│       ├── model_forward_uncond 2.1s wall / 2.1s GPU
│       ├── guidance_merge       <1ms
│       └── scheduler_step       526ms wall / 1ms GPU (CPU-bound)
├── dit_free       <1s   del + gc (was 10s with old CPU offload)
└── vae_decode     15s   Latent→pixels (120× decompression)
```

#### Key observations

- **Diffusion dominates (88%)**: 100 DiT forward passes (50 steps × 2 for classifier-free guidance). Each pass processes ~7K tokens through 30 transformer layers.
- **T5 loading is the init bottleneck (61s)**: Uses `torch.load()` on a .pth file — sequential deserialization + double allocation (construct model then copy weights). Converting to safetensors would reduce this to <5s.
- **Scheduler is CPU-bound**: 526ms wall but 1ms GPU per step. The UniPC solver runs on CPU.
- **Free vs offload saves ~20s**: Replacing `model.cpu()` (PCIe transfer back) with `del + gc.collect()` eliminated the 10s T5 offload and 10s DiT offload.
- **model_forward_cond wall < GPU time**: CPU returns from async kernel launch before GPU finishes (1.6s vs 2.1s). model_forward_uncond wall ≈ GPU time because by then CPU has caught up.

### Earlier Traces (for reference)

| Trace | Location | Notes |
|---|---|---|
| TI2V-5B, 5 frames, no FSDP | `~/Downloads/wan2.2_profile/` | First successful run. Peak 29.6GB. |
| T2V-A14B, 17 frames, FSDP2 | `~/Downloads/wan2.2_fsdp2_17f/` | FSDP2 working at 17 frames. Shows clean sharding. |
| TI2V-5B, 81 frames, kernel traces | `~/Downloads/wan2.2_ti2v5b_kernel_trace/` | Merged L2 + L3 traces (app-level spans + CUDA kernel events for T5 encoding, step_2, and VAE decode phases). |

## Optimization experiments

### JIT compilation (`torch.compile`)

**Works.** ~20% speedup on diffusion loop.

| Mode | Step 1 (warmup) | Steady-state step | Diffusion (50 steps) |
|---|---|---|---|
| Eager | 4.3s | 4.25s | ~213s |
| JIT (first run) | 16.6s (compile) | 3.41s | ~183s |
| JIT (cached) | 8.2s | 3.41s | ~174s |

**Command:**
```bash
TORCHINDUCTOR_CACHE_DIR=/workspace/.compile_cache \
python generate.py --task ti2v-5B --size 1280*704 --frame_num 81 \
  --ckpt_dir /workspace/Wan2.2-TI2V-5B \
  --offload_model True --compile_model \
  --profile_dir ./profile_output \
  --prompt "..."
```

**Cache behavior**: Keyed on tensor shapes + dtypes. Same prompt/seed → cache hit. Different `--frame_num` or `--size` → recompile (first time for that value). Cache persists at `/workspace/.compile_cache/` (~19MB per shape config).

**Profiling integration**: CUDA events are unreliable inside compiled regions (graph reordering breaks start/end event pairing). The profiling framework detects `_compile_active` and falls back to wall-clock only for `model_forward_cond`, `model_forward_uncond`, and `model_forward/*` hook spans. All other spans (T5, VAE, scheduler, step_N) keep full wall + GPU timing.

### AOTInductor (ahead-of-time compilation)

**Blocked** on the current model code. Investigation preserved for future reference.

Attempted `torch.export.export()` + `torch._inductor.aoti_compile_and_package()` on the TI2V-5B DiT. Fails at `wan/modules/model.py:57` in `rope_apply`:

```python
for i, (f, h, w) in enumerate(grid_sizes.tolist()):  # unbacked symints
    seq_len = f * h * w
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        ...
    ])
```

The `.tolist()` creates unbacked symbolic ints that trigger `GuardOnDataDependentSymNode` when used as tensor shapes. This is a fundamental incompatibility: AOT requires the full graph to trace, while JIT falls back to eager on graph breaks.

**Effort to fix**: 4 hours (best case, annotations only) to 1 week (worst case, multiple rewrites). Main blocker is the rope_apply rewrite to avoid `.tolist()`. Additional potential blockers in `flash_attention`, list IO, and runtime asserts.

**Expected payoff vs JIT**: ~5-10% additional speedup (170-175s vs 183s) + zero warmup cost per run. Only worth it for production serving at fixed shapes.

**Decision**: Not pursued. JIT `torch.compile` provides most of the benefit with no model changes needed.

### FSDP + Ulysses on 4× RTX 5090 (TI2V-5B)

**Both work and compose.** With the FSDP cleanup fix below, 161 frames at 1280×704 fits comfortably.

| Config | Frames | Peak GB (rank 0) | Diffusion | Total |
|---|---|---|---|---|
| 1 GPU baseline | 81 | 27.0 | 213s | 310s |
| 4 GPU FSDP | 81 | 23.3 | 212s | 310s |
| 4 GPU FSDP + Ulysses | 81 | 27.1 | **131s** (1.6×) | 230s |
| 4 GPU FSDP + Ulysses | **161** | **24.3** | 233s | 351s |
| 4 GPU Ulysses only | 81 | — | — | OOM at T5+fp32 DiT (21.8+10.8>32GB) |

**Memory decomposition (1 GPU baseline, 81 frames, fp32 DiT):**

| Stage | allocated | notes |
|---|---|---|
| VAE loaded | 2.7 GB | |
| T5 on GPU | 13.5 GB | T5-XXL bf16 = 10.8 GB |
| T5 freed | 2.7 GB | back to VAE |
| DiT on GPU | 21.8 GB | fp32 DiT = 19.1 GB (passing `--convert_model_dtype` halves this) |
| Diffusion step peak | 27.0 GB | activations ≈ 5.2 GB |
| VAE decode peak | 23.1 GB | **chunked** — ~16 GB workspace, roughly frame-independent |

**Key findings:**
- **FSDP shards DiT cleanly**: 19.1 GB → 4.8 GB per GPU (4×).
- **Ulysses reduces activations** ~48–62% (not the full 4× — FSDP all-gather buffers and cross-attn don't shard).
- **Diffusion peak stays low** with FSDP+Ulysses: 10.5 GB at 81f, 12.9 GB at 161f. Plenty of room for more frames on the diffusion path.
- **VAE decode is the ceiling**, not diffusion. It runs on rank 0 only, doesn't use Ulysses, but is internally chunked so it scales sub-linearly with frames.
- **Ulysses alone infeasible with fp32 DiT** — full DiT (19.1 GB) + T5 on GPU (10.8 GB) > 32 GB. Pair with FSDP or `--convert_model_dtype`.

**FSDP cleanup bug (fixed):**

`del self.model` did not release FSDP-sharded weights. `after_dit_free` showed 7.8 GB still allocated vs 2.9 GB on single-GPU baseline — the 4.9 GB FlatParameter storage was pinned by FSDP's internal handles. This caused VAE decode to OOM.

Fix: call `free_model(self.model)` (from `wan/distributed/fsdp.py:39`) before `self.model = None`. It walks FSDP submodules and calls `_free_storage(m._handle.flat_param.data)` to explicitly resize storage to 0 bytes. After fix: `after_dit_free` drops to 3.0 GB on the FSDP-only run.

Note: when `use_sp=True` (Ulysses), `after_dit_free` still shows ~7.9 GB retained — likely from `sp_attn_forward`/`sp_dit_forward` monkey-patch closures or persistent all-to-all NCCL buffers, separate from the FlatParameter. The run still fits because VAE decode's chunked workspace is smaller than that residual + budget.

**Reproduce:**

```bash
CKPT=/workspace/Wan2.2-TI2V-5B
PROMPT='A cat walking through a sunlit meadow'
OUT=/workspace/profile_outputs
export WAN_PROFILE_FLUSH_INTERVAL=1  # persist CSV row-by-row (survives crashes)

# 1 GPU baseline
python generate.py --task ti2v-5B --size '1280*704' --frame_num 81 \
  --ckpt_dir $CKPT --offload_model True \
  --profile_dir $OUT/baseline_1gpu_81f --prompt "$PROMPT"

# 4 GPU FSDP only
torchrun --nproc_per_node=4 --master_port=29500 generate.py \
  --task ti2v-5B --size '1280*704' --frame_num 81 \
  --ckpt_dir $CKPT --offload_model True --dit_fsdp \
  --profile_dir $OUT/fsdp_only_81f --prompt "$PROMPT"

# 4 GPU FSDP + Ulysses, 81 frames
torchrun --nproc_per_node=4 --master_port=29500 generate.py \
  --task ti2v-5B --size '1280*704' --frame_num 81 \
  --ckpt_dir $CKPT --offload_model True --dit_fsdp --ulysses_size 4 \
  --profile_dir $OUT/fsdp_ulysses_81f --prompt "$PROMPT"

# 4 GPU FSDP + Ulysses, 161 frames (enabled by the fix)
torchrun --nproc_per_node=4 --master_port=29500 generate.py \
  --task ti2v-5B --size '1280*704' --frame_num 161 \
  --ckpt_dir $CKPT --offload_model True --dit_fsdp --ulysses_size 4 \
  --profile_dir $OUT/fsdp_ulysses_161f --prompt "$PROMPT"
```

**Read results**: memory rows are written to `timing_rank0.csv` with the format
`run_id,rank,memory/<label>,-1,<allocated_mb>,<peak_mb>,<timestamp>`.
Labels emitted per stage: `init_start`, `after_load_{t5,vae,dit}`, `after_t5_to_gpu`, `after_text_encoding`, `after_t5_free`, `after_dit_to_gpu`, `step_N` (per diffusion step), `after_diffusion_loop`, `after_dit_free`, `after_vae_decode`, `pipeline_end`.

Quick one-liner for peak + timing:
```bash
awk -F, '/memory\// {if ($6+0>mx) mx=$6+0} \
  /memory\/init_start/ {st=$7} /memory\/pipeline_end/ {en=$7} \
  /memory\/after_dit_to_gpu/ {d0=$7} /memory\/after_diffusion_loop/ {d1=$7} \
  END {printf "peak=%.0fMB total=%.1fs diffusion=%.1fs\n", mx, en-st, d1-d0}' \
  $OUT/<run>/timing_rank0.csv
```

## GPU utilization sampler

Background thread polls NVML on a fixed interval (default 100 ms) and writes one CSV row per metric alongside the regular timing rows. Also emitted as a Chrome Trace `gpu_sampler` counter so Perfetto plots all series under the span timeline.

**Metrics** (one row each per tick, `name` column carries the metric, `wall_ms` carries the value, `gpu_ms` is always `-1`):

| name | meaning | source |
|---|---|---|
| `gpu/util` | % of sample window any kernel was running (NVML rolling ~1 s avg) | `nvmlDeviceGetUtilizationRates().gpu` |
| `gpu/mem_bw_pct` | % of sample window memory controller was busy — **bandwidth**, not capacity | `nvmlDeviceGetUtilizationRates().memory` |
| `gpu/power_w` | power draw in watts | `nvmlDeviceGetPowerUsage` |
| `gpu/sm_mhz` | SM clock in MHz | `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` |
| `gpu/temp_c` | GPU temperature in Celsius | `nvmlDeviceGetTemperature` |
| `gpu/mem_used_mb` | resident memory in MB (all processes on the GPU) | `nvmlDeviceGetMemoryInfo().used` |
| `gpu/throttle_reasons` | NVML bitmask; 0 = no throttling | `nvmlDeviceGetCurrentClocksThrottleReasons` |

**Enable/disable** (on by default when profiling is enabled):

| Env var | Default | Effect |
|---|---|---|
| `WAN_PROFILE_GPU_SAMPLER` | `1` | set to `0` to disable the sampler entirely |
| `WAN_PROFILE_GPU_SAMPLE_MS` | `100` | polling interval in ms |
| `WAN_PROFILE_GPU_SAMPLER_RANKS` | `0` | rank filter: `0` (rank 0 only), `all`, or comma list e.g. `0,2` |

**Read results** — mean + max utilization and peak memory for rank 0:

```bash
awk -F, '$3=="gpu/util"       {n++; s+=$5; if($5>mx)mx=$5} \
         $3=="gpu/mem_used_mb"{if($5>mm)mm=$5} \
         $3=="gpu/power_w"    {if($5>pw)pw=$5} \
         END{printf "util mean=%.1f%% max=%.0f%% | mem peak=%.0fMB | power peak=%.0fW (n=%d)\n", \
             s/(n?n:1), mx, mm, pw, n}' \
  $OUT/<run>/timing_rank0.csv
```

To visualize over time, open `trace_rank0.json` in Perfetto — the `gpu_sampler` counter lane shows every series aligned with diffusion spans.

**Caveats**:

- `gpu/util` and `gpu/mem_bw_pct` are rolling ~1 s averages inside NVML, not instantaneous. Sub-second spikes will be smoothed out; oversampling below 500 ms mostly buys alignment, not resolution.
- `gpu/mem_bw_pct` is memory-BANDWIDTH utilization (% of time any memory read/write was in flight). It is NOT "% of VRAM used" — use `gpu/mem_used_mb` for capacity.
- `gpu/mem_used_mb` comes from NVML and reports all resident memory on the physical GPU, including other processes. On a shared host this will be higher than the process-local figure from `record_memory`.
- Under `CUDA_VISIBLE_DEVICES` the sampler resolves NVML handles by UUID (via `torch.cuda.get_device_properties(local_rank).uuid`) because `nvmlDeviceGetHandleByIndex` takes PHYSICAL indices and does not honour the visibility mask. Using `local_rank` directly would silently sample the wrong GPU.
- By default only rank 0 samples, to avoid 4× redundant rows under DDP/FSDP. Set `WAN_PROFILE_GPU_SAMPLER_RANKS=all` to diagnose per-GPU imbalance (e.g. VAE decode on rank 0 only, Ulysses all-to-all skew).

## NCCL collective stats

`all_to_all` and `all_gather` from `wan/distributed/util.py` are wrapped with timing + byte tracking. Each call emits a `collective/<op>` CSV row with wall + GPU time and byte count; `flush()` appends aggregate `collective_total/<op>` and `collective_total/<op>_bytes` rows.

**Example** (17-frame FSDP+Ulysses-4 run, rank 0):

```
collective_total/all_to_all       count=12000  wall=1023ms   gpu=12899ms   bytes=81.1 GB
collective_total/all_gather       count=100    wall=12.5ms   gpu=23.2ms    bytes=338 MB
```

12.9 s of GPU time in all-to-all on rank 0 ≈ 17% of diffusion GPU time — a direct answer to "why is Ulysses only 1.6× speedup instead of 4×".

One-liner:
```bash
awk -F, '$3 ~ /^collective_total\// {print}' $OUT/<run>/timing_rank0.csv
```

## Memory fragmentation, PCIe bytes, attention backend

- **`memory_frag/<label>`** rows accompany every `memory/<label>` row. `wall_ms` = `reserved − allocated` (MB); `gpu_ms` = `reserved` (MB). Large values mean the caching allocator is holding freed blocks unreusable by the next alloc.
- **`t5_to_gpu` / `dit_to_gpu` spans** carry a `bytes` metadata field in the trace JSON. Divide by `wall_ms` to get effective PCIe bandwidth. RTX 5090 PCIe 5.0 x16 tops out at ~63 GB/s.
- **`env/attn_backend/<name>`** one-shot row records the attention path (`flash_attn_3`, `flash_attn_2`, or `sdpa`) selected at profile init.

## CPU cProfile (opt-in)

The scheduler loop is CPU-heavy (~100–500 ms wall per step, 1 ms GPU). To see what dominates, set `WAN_PROFILE_CPU_SCHEDULER=1` and rerun. Stats accumulate across all 50 `scheduler.step()` calls and dump to `<profile_dir>/cpu_scheduler_rank0.prof`.

```bash
WAN_PROFILE_CPU_SCHEDULER=1 python generate.py ...   # run with profiling
python -m pstats $OUT/<run>/cpu_scheduler_rank0.prof \
  -c 'sort cumulative' -c 'stats 15' -c quit
```

**Example** (UniPC on 17 frames, 50 steps, 4.8 s scheduler CPU total):
- `multistep_uni_c_bh_update`: **2.48 s (51%)** — the UniPC corrector is the hotspot
- `torch.tensor(...)` (247 calls): **2.33 s** — scalar-to-CUDA conversions inside UniPC, a clear optimization target

Overhead: cProfile adds ~30–50% CPU time to the wrapped region. Don't leave it on for steady-state benchmarking. Any region can be profiled the same way — wrap it in `with cpu_profile("name"):` and enable via `WAN_PROFILE_CPU_NAME=1`.

## Variance driver

Wall numbers from a single run have ±5% noise (warmup, thermal, OS jitter). `scripts/profiling/variance_run.py` runs a command N times and reports mean ± std:

```bash
python scripts/profiling/variance_run.py \
    --name fsdp_uly_81f --runs 3 --skip-first \
    --out-root /workspace/profile_outputs \
    -- \
    torchrun --nproc_per_node=4 --master_port=29500 generate.py \
      --task ti2v-5B --size '1280*704' --frame_num 81 \
      --ckpt_dir /workspace/Wan2.2-TI2V-5B --offload_model True \
      --dit_fsdp --ulysses_size 4 --prompt 'repro'
```

Output:
```
=== Aggregate ===
  peak_mb     : 27123.4 ± 32.1 (n=2)
  total_s     : 229.5 ± 1.4 (n=2)
  diffusion_s : 131.2 ± 0.9 (n=2)
```

`--skip-first` discards run 0 as warmup (cache-cold, thermal low).
