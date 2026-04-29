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

---

# Demo Server Architecture Investigation — gpu6 (4× RTX 4090, 24 GB, PCIe-PHB, no NVLink)

The demo server originally inherited the Wan2.2 reference codebase's 4-GPU FSDP+Ulysses path. On gpu6's hardware (consumer 4090s, no NVLink, all GPU↔GPU comm goes through the CPU host bridge), this path is **net negative** for the 5B model. This section captures the investigation that overturned it and the validated single-GPU + warm + fp8 path that's now ~2.25× faster.

## TL;DR — gpu6 paths

| Path | 81f wall time | Per-step | Δ vs old prod |
|---|---:|---:|---:|
| Demo 4-GPU FSDP+Ulysses bf16 (old prod) | 649 s | 10.30 s | — |
| Demo 1-GPU bf16 (just `--nproc_per_node=1`) | 414 s | 5.55 s | −36% |
| **Demo 1-GPU fp8 warm (current prod)** | **293 s warm / 382 s cold** | **5.76 s** | **−55%** |
| ComfyUI 1-GPU bf16 warm (reference) | 291 s | 5.80 s | — |
| ComfyUI 1-GPU fp8_fast warm (reference) | 259 s | 5.20 s | — |

Demo 1-GPU per-step (5.55 s) ≈ ComfyUI 1-GPU per-step (5.80 s). The pipeline is fine; the parallelism strategy was the regression.

## The hardware mismatch

```
nvidia-smi topo -m on gpu6:
       GPU0   GPU1   GPU2   GPU3
GPU0    X    PHB    NODE   NODE
GPU1   PHB    X     NODE   NODE
GPU2   NODE  NODE    X     PHB
GPU3   NODE  NODE   PHB     X

NVLink: empty (4090 is consumer; no NVLink support)
PCIe gen.current=1 idle, gen.max=4 (ramps under load)
```

**PHB** = PCIe Host Bridge (CPU). Every GPU↔GPU NCCL collective makes a round-trip through the CPU. Realistic NCCL aggregate ~5–10 GB/s on this topology vs ~900 GB/s on NVSwitch-equipped servers (H100/A100 SXM) where the upstream Wan2.2 code was designed.

## Why FSDP+Ulysses is net-negative for the 5B model on this hardware

Per inference (50 steps × CFG 2× = 100 forwards), the 4-GPU path does:

- **3000 FSDP all-gathers** (30 transformer blocks × 100 forwards) reconstructing the full ~167 MB block weights on each rank
- **12,000 Ulysses all-to-alls** (4 per attention block × 30 blocks × 100 forwards)

Each collective is a sync point and PCIe round-trip. With ~100 ms aggregate per-block on PHB-PCIe, that's hundreds of seconds of pure comm. The 4-way compute speedup from sequence parallelism (~1.5–2× realistic) does not pay for the comm tax.

Reference codebase pays this tax happily because (a) the 14B model literally doesn't fit on 1 GPU, and (b) NVLink-class hardware makes collectives essentially free. Neither premise holds for the 5B on consumer 4090s.

## fp8 attempts on the demo server

**Attempt 1: fp8 + FSDP1 (failed).** Naive quantization of all 300 `nn.Linear` modules (4.9 B params → fp8) before FSDP1 wrapping. Failed at FSDP wrap with `ValueError: Must flatten tensors with uniform dtype`. FSDP1 flattens parameters per wrapped unit; our wrap policy wraps each transformer block as one unit; inside a block we have fp8 Linears + bf16 norms → mixed → fail.

**Attempt 2: fp8 + 1-GPU + warm (worked).** With FSDP gone (world-size-1 short-circuit added to `_configure_model`), fp8 conversion runs cleanly. 3 back-to-back 81f generations stable at 287.7 / 288.7 / 288.1 s, peak 23.1 GB. Pipeline ctor 85.9 s one-time.

The demo server's fp8 path is **storage-only** (weights cast to fp8_e4m3fn but matmul still bf16). Same gotcha as ComfyUI's `fp8_e4m3fn` vs `fp8_e4m3fn_fast` toggle — only `_fast` engages 4090 fp8 tensor cores via `torch._scaled_mm`. Storage-only fp8 saves 37% steady-state VRAM but no compute. The wall-clock win on the demo server came entirely from eliminating the ~85 s reload-per-request, *not* from fp8 acceleration.

## Memory ceiling on 4090

Single 4090, fp8 warm, 81f at 1280×704: **23.1 GB peak**. With `offload_model=True` (T5 to CPU after encode, DiT to CPU before VAE decode):

- Resident steady-state: ~7.5 GB (DiT fp8 2.5 + VAE 0.5 + cached buffers)
- Diffusion peak: ~15 GB (DiT on GPU + activations × CFG 2)
- VAE decode peak: 23.1 GB

A first run with `offload_model=False` OOMed at 22.5 GB allocated trying to alloc 218 MB in `rope_apply`. **`offload_model=True` is mandatory** for 81f on a single 4090.

The 4090 OOMs at ~141 frames at 1280×704 (VAE decode peak hits 24 GB ceiling).

## Validated production architecture

`docs/demo-server.md` covers the full architecture. Summary:

1. `--nproc_per_node=1` — drop FSDP, drop Ulysses
2. Module-level `_PIPELINE` singleton in `server/worker.py` — one ctor at startup, reuse forever
3. `WAN_DEMO_QUANT=fp8` default (storage-only on 4090)
4. `offload_model=True` — mandatory for 24 GB headroom
5. Drop the rank>0 worker loop when world_size=1

Result: 288 s/job vs 649 s today. 2.25× speedup. Frees 3 GPUs for parallel jobs (run 4 instances for 4× concurrency).

---

# Cross-platform: H100 80GB single-GPU vs RTX 4090

After the demo-server refactor landed, we benchmarked the same pipeline on a vast.ai H100 80GB SXM-class instance to characterize the 4090→H100 gap and the headroom H100 unlocks for longer videos.

## All-config benchmark — 81 frames @ 1280×704, 50 steps, warm

| Backend | Hardware | Wall time | Per-step | Frames/s | Real-time factor* |
|---|---|---:|---:|---:|---:|
| **ComfyUI fp8_e4m3fn_fast** | **1× H100 80GB** | **84.3 s** | 1.69 s | **0.961** | **25.0×** |
| Demo server fp8 warm | 1× H100 80GB | 93.4 s | 1.87 s | 0.868 | 27.7× |
| ComfyUI bf16 | 1× H100 80GB | 90.1 s | 1.80 s | 0.900 | 26.7× |
| ComfyUI fp8_e4m3fn (storage) | 1× H100 80GB | 90.3 s | 1.81 s | 0.897 | 26.8× |
| Demo server bf16 warm | 1× H100 80GB | 102.3 s | 2.07 s | 0.792 | 30.3× |
| ComfyUI fp8_e4m3fn_fast | 1× 4090 24GB | 259.0 s | 5.20 s | 0.313 | 76.7× |
| Demo server fp8 warm | 1× 4090 24GB | 293.0 s | 5.76 s | 0.276 | 86.8× |
| ComfyUI bf16 | 1× 4090 24GB | 290.8 s | 5.80 s | 0.278 | 86.2× |
| ComfyUI fp8_e4m3fn (storage) | 1× 4090 24GB | 288.7 s | — | 0.281 | 85.6× |
| Demo 1-GPU bf16 (cold) | 1× 4090 24GB | 414.0 s | 5.55 s | 0.196 | 122.7× |
| Demo 4-GPU FSDP+Ulysses bf16 | 4× 4090 24GB | 649.4 s | 10.30 s | 0.125 | 192.3× |

\* Real-time factor = wall_time / video_duration (3.375 s @ 24 fps source for 81 frames)

## Cross-platform speedup ratios

| Path | 4090 → H100 |
|---|---:|
| ComfyUI fp8_fast | 259 → 84 s = **3.07×** |
| ComfyUI bf16 | 291 → 90 s = **3.23×** |
| Demo fp8 warm | 293 → 93 s = **3.14×** |
| Demo bf16 warm | (≈291) → 102 s = **2.85×** |

Consistent **~3×** across configs. Notable: the per-step ratio matches the wall-time ratio, so this is real per-step compute speedup, not init/load amortization.

## H100-specific observations

**1. The "fp8 win" gap shrinks on H100.** On 4090: fp8_fast vs bf16 saves 32 s (-11%). On H100: only 6 s (-6.4%). H100 is bandwidth-limited at bf16, so speeding up matmul (fp8 tensor cores) doesn't help proportionally.

**2. Storage-only fp8 ties bf16 on H100 too** (90.1 vs 90.3 s) — same conclusion as 4090. Storage-only fp8 saves memory but not time.

**3. Pipeline ctor is faster on H100** — 50–58 s vs 86 s on 4090 (~40% faster) thanks to faster storage and HBM bandwidth.

**4. Demo server vs ComfyUI gap also shrinks on H100** — 9 s gap (93 vs 84) vs 34 s on 4090. Reload tax / offload differences matter less when everything is fast.

## Frame-count headroom on H100

| Frames | Wall time | Per-step | Peak alloc | Notes |
|---:|---:|---:|---:|---|
| 81 | 93 s | 1.87 s | 24.0 GB | baseline |
| 401 | 805 s | 15.2 s | 33.0 GB | first probe — succeeded |
| **721** | **2154 s** | **43.1 s** | **53.2 GB** | **30 s of video at 24 fps** |

Memory scales sub-linearly with frame count: ~28 MB/frame extra peak alloc beyond baseline. By that scaling H100 could plausibly fit 1500+ frames before hitting the 80 GB ceiling. **But see the next section — the model can't usefully generate that long.**

## ComfyUI install on H100 — gotchas

If reproducing this bench, three traps to avoid:

1. **kijai's `WanVideoModelLoader` requires single-file DiT.** Wan2.2 ships sharded; merge in-memory with `safetensors.torch.load_file()` + `.clone()` *before* `os.remove()` of the shards (mmap retains the inode on `safe_open`-style readers, blocking actual disk free until process exit). On a small `/workspace`, do `os.remove()` *before* the merged write to make room for the output.

2. **VHS_VideoCombine** node lives in a separate repo (`Kosinkadink/ComfyUI-VideoHelperSuite`). The bench script's workflow uses it for mp4 output. Easy miss, fast fail with `missing_node_type`.

3. **flash_attn 2** has no prebuilt wheel for torch 2.11+cu128+py312. The Wan2.2 codebase calls `flash_attention()` directly which `assert FLASH_ATTN_2_AVAILABLE`s. The wrapping `attention()` has a clean SDPA fallback but isn't on the hot path. Patch `flash_attention()` to fall back to `torch.nn.functional.scaled_dot_product_attention` when neither FA2 nor FA3 is available — PyTorch 2.11's bundled flash backend is competitive on H100. See the diff in `wan/modules/attention.py` under the world-size-1 fixes for an example.

---

# The training-horizon wall — why long Wan2.2 videos look static

We pushed the H100's frame-count headroom (memory permits) toward a 30-second clip and hit a model-quality wall, not a memory wall.

## The 30-second probe

Same setup as the 81/401-frame benches (fp8 warm, 1280×704, 50 steps, seed 42), `--frames 721` (=`4×180+1` ≈ 30.04 s @ 24 fps):

- Wall time 2154 s, peak alloc 53 GB, per-step 43 s — pipeline executed cleanly.
- **But the output is essentially static.** Frames 0, 360, 720 show the same composition with only minute frame-to-frame differences.
- A multi-scene narrative prompt (5 scenes: office desk → paper airplane → drawing → rain → bright park) collapsed into a single visual that loosely resembles the *last* described scene, with no scene cuts or transitions.
- Frame 0 also shows OOD denoising artifacts (pixelated glitches in one half of the image).

## Two stacked failures

**Failure 1 — past training distribution.** Wan2.2 TI2V-5B's `ti2v_5B.frame_num = 121` config is the trained/recommended max (set in `wan/configs/wan_ti2v_5B.py`). 721 frames is **~6× past training horizon**. The model "stretches" — produces minute frame-to-frame variations but no real scene evolution. The pixelated artifact is classic out-of-distribution denoising failure.

**Failure 2 — multi-scene prompts don't work in any current text-to-video model.** Wan2.2 (and Sora, Veo, Cog…) treat the entire prompt as describing **one scene**, not a script with cuts. Markers like "Scene 1… Scene 2…" get averaged into a single visual setting; the most-emphasized or last-described scene typically dominates. Multi-scene narratives must be generated as separate clips and stitched.

## Implications

1. **The `MAX_FRAMES=141` cap on the demo server is doing real work**, not just being defensive. Beyond ~121 frames, output quality degrades sharply regardless of available memory. The cap should stay even on H100 deployments.

2. **The H100's frame-count *memory* headroom does not translate into useful long-form generation.** 80 GB unlocks the *ability* to render 1000+ frames in a single pass; the *model* can't keep them dynamic.

3. **For long-form output, three patterns work:**
   - **Multi-clip stitching**: generate 5× ≤121-frame clips with separate single-scene prompts, cross-fade with ffmpeg. Same total wall-clock as one monolithic run, but with actual scene changes.
   - **Single-scene continuous-motion at ≤121 frames**: rich continuous motion in one setting (camera dolly, weather change, slow object motion). The jellyfish prompt is closer to this pattern than the multi-scene dream prompt and the output reflects it.
   - **i2v chaining** (Wan2.2 paper's recommended pattern for long videos): generate a 121-frame clip; use the last frame as image conditioning for the next clip's prompt; concatenate. Requires the i2v branch which is in the codebase but not wired to the demo server.

## Concrete artifacts on H100

`/workspace/Wan2.2/bench_artifacts/`:
- `stats_h100_fp8.json`, `stats_h100_bf16.json` — 81f warm matrix
- `stats_h100_401.json` — 401f probe (succeeded, 805 s, 33 GB peak)
- `stats_h100_30s.json`, `stats_h100_30s_dream.json` — 721f / 30 s runs (jellyfish + multi-scene dream prompt)
- `h100_30s_jelly/warm_fp8_gen1.mp4`, `h100_30s_dream/warm_fp8_gen1.mp4` — output mp4s
- `h100_comfy_bench/runs.jsonl` — ComfyUI matrix bench (bf16 / fp8 / fp8_fast at 81f)

