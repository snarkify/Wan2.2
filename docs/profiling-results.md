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
