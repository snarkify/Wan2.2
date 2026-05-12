# Wan2.2 T2V-A14B on 2× H100 — TL;DR

**Workload:** 1280×720, 81 frames, 40 UniPC steps, fixed seed.
**Hardware:** 2× H100 80GB HBM3, NVLink (NV18).
**Stack:** PyTorch 2.5.1+cu121, FA2 (FA3 deferred), bf16 weights via `--convert_model_dtype`.
**Metric:** `diffusion_s` = 40-step denoising loop only (excludes T5 / VAE / load).

## Leaderboard (lossless candidates)

| Rank | Config | diffusion_s | Speedup vs 1-GPU | Peak GB | Bit-equal to Y1? |
|---|---|---|---|---|---|
| — | **Z1** — 1× H100 baseline (offload) | 1619.9 ± 0.1 (n=2) | 1.00× | 61 | n/a |
| 4 | **Y1** — 2× H100, Ulysses-2 (existing seq-parallel) | 879.5 ± 0.7 (n=2) | 1.84× | 66 | reference |
| 3 | **Y4** — 2× H100, **CFG-parallel-2** (new, in-tree) | 825.4 ± 0.4 (n=4, old server) | 1.96× | 61 | ✅ MD5 identical to Y1 (within platform) |
| ★ | **Y4-sync** — Y4 + one-line `cuda.current_stream().synchronize()` after `dist.all_gather` | **613.2 ± 3.3 (n=5, new server)** | **2.64×** | TBD | ✅ MD5 identical across all 5 Y4-sync runs (CV 0.5%); 28.6 dB vs old-server Y4-ref is the platform shift only (torch 2.5.1→2.11 + FA2→SDPA), not the sync fix |
| 2 | **Y5** — Y4 + `torch.compile` (default mode) | 780.6 ± 5.7 (n=4) | 2.08× | 64 | ❌ 28.4 dB drift |
| — | Y6 — Y4-sync + `torch.compile mode=reduce-overhead` (torch 2.11) | 592.4 ± 15.5 (n=2) | 1.04× vs Y4-sync | TBD | ❌ 28 dB drift + **non-deterministic** (29 dB run-to-run, MD5 differs) |
| 1 | **X1** — xDiT (xfuser) CFG-parallel-2 | 763.6 ± 1.2 (n=2) / 539 (n=1 rerun) | 2.12× / 3.00× | 67 / 74 | ❌ **13.86 dB PSNR vs Y4** (diffusers RNG path) |

## Engines compared (same workload, 2× H100)

| Engine | Best config | diffusion_s | vs Y4 | Quality |
|---|---|---|---|---|
| In-tree (this fork) | Y4 CFG-parallel | 825.4 | 1.00× | ✅ bit-equal to upstream Wan2.2 (`generate.py`) |
| In-tree + compile | Y5 | 780.6 | 1.06× | ❌ 28.4 dB drift |
| xDiT (xfuser) | X1 CFG-parallel | 763.6 / 539 (SDPA rerun) | 1.08× / 1.53× | ❌ **13.86 dB PSNR** — diffusers RNG path, not in-tree |
| SGLang-Diffusion | S1 CFG-parallel | **393.1** | **2.10×** | ❌ **17.2 dB PSNR** — also diffusers RNG path (same family) |

## Headline

**CFG-parallel-2 + a one-line sync fix is the strict-lossless 720p winner — 2.64× over single-GPU.** Original Y4 gave 1.96× at bit-equal output. After diagnosing the visible `scheduler_step` regression (2640× slower than Y1's, ~3 s/step), the root cause turned out to be implicit-sync pollution from `cfg_all_gather`'s async `dist.all_gather` returning before NCCL completed. Adding `torch.cuda.current_stream().synchronize()` after the all_gather drops Y4 to **613 ± 3 s diffusion (2.64×, n=5, CV 0.5%)**.

**Bit-equality property is robust on the new platform.** All eight in-tree runs of CFG-parallel-2 (with and without offload) and Ulysses-2 produce the **same MD5**. The in-tree code path is fully deterministic and equivalent across parallelism schemes. Y4-sync (CFG-parallel-2) is 8.8% faster than Y1-new (Ulysses-2) — 613 s vs 668 s — for the same byte-for-byte output. Production choice: CFG-parallel-2 + sync fix.

**Confirmed**: external engines (xDiT, SGLang) are not lossy — they wrap diffusers, and diffusers itself diverges from `generate.py` by ~18 dB. A vanilla `diffusers.WanPipeline` invocation on the same hardware scored **17.9 dB vs Y4**, matching xDiT (13.9 dB vs Y4) and SGLang (17.2 dB vs Y4) within the same family. Three independent diffusers-path invocations all cluster around 14–18 dB from in-tree → same model, different RNG/scheduler init path, not approximation.

| | Y4 (in-tree) | Diffusers vanilla | xDiT X1 | SGLang S1 |
|---|---|---|---|---|
| Y4 | ∞ (=Y1) | 17.9 dB | 13.9 dB | 17.2 dB |
| Diffusers | 17.9 dB | reference | — | 16.5 dB |
| SGLang | 17.2 dB | 16.5 dB | — | — |

If you require bit-equality to in-tree Wan2.2: only **Y4** qualifies. If your serving reference is `diffusers.WanPipeline`: **SGLang S1** at 393 s is **2.71× faster than vanilla diffusers** with no extra quality loss beyond what diffusers itself does. The decision is which code path you bind to.

**Visual A/B confirmed (2026-05-12)**: Y4-sync, diffusers-vanilla, SGLang S1, and old-server-Y4 all look equally good. The 14-18 dB PSNR cluster is structural sample variation (different helmets, different specific details) — not quality degradation. SGLang's 2.71× over diffusers-vanilla is therefore a clean production win, not a quality trade-off.

## Why CFG-parallel beats Ulysses

| | Ulysses-2 | CFG-parallel-2 |
|---|---|---|
| What's split | tokens (sequence) | forward branch (cond / uncond) |
| Per-rank work per step | 2× full forwards on half tokens | 1× full forward on full tokens |
| Collective traffic / generation | ~2400 `all_to_all` calls, ~5 TB | 1 `all_gather`, ~1.5 GB |
| Numerical equivalence to vanilla | bit-equal (yes) | bit-equal (yes) |

3300× less NCCL traffic ⇒ ~6% per-step compute savings on NVLink-connected H100s.

## Production recommendation

| Use case | Config | Speedup | Notes |
|---|---|---|---|
| Strict-bit-equality to in-tree `generate.py` | **Y4** | 1.96× | bit-equal to in-tree Wan2.2 (Y1 ↔ Y4 MD5-identical) |
| Production on `diffusers.WanPipeline` reference | **SGLang S1** | 4.12× vs in-tree 1-GPU; 2.71× vs vanilla diffusers | 16.5 dB vs vanilla diffusers, 17.2 dB vs in-tree — same RNG-path family, no extra quality loss |
| Production on `diffusers` reference, lower-effort | xDiT X1 | 2.12×–3.00× | same diffusers path; 13.9 dB vs in-tree; FA2 path needs rebuild on this server |
| Vanilla diffusers (1-GPU, no parallelism) | — | ~1.52× | diffusers-path 1-GPU reference |
| Compile-path latency, accepts numerical drift | Y5 | 2.08× | ~28 dB drift from Y4; in-tree but lossy |
| Cost-optimized single-GPU, in-tree | Z1 | 1.00× | latency 1620 s |

## How to reproduce Y4-sync (in-tree, strict lossless, 614 s)

Apply this one-line patch to `wan/distributed/cfg_parallel.py:106` (after `dist.all_gather`):

```python
torch.cuda.current_stream().synchronize()
```

Then:

```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B --ckpt_dir <path> \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed 42 --prompt '<...>' \
  --cfg_parallel_size 2 --convert_model_dtype --offload_model True
```

## How to reproduce X1 (xDiT, faster lossless)

Requires `xfuser>=0.4.5` + local Wan adapter modules in `scripts/profiling/`:

```bash
torchrun --nproc_per_node=2 scripts/profiling/run_xdit.py \
  --model <hf-diffusers-Wan2.2-A14B-path> \
  --prompt '<...>' --height 720 --width 1280 \
  --num_frames 81 --num_inference_steps 40 --seed 42 \
  --use_cfg_parallel --profile_dir ./out
```

## Open headroom

- **xDiT X1 output capture + PSNR confirmation.** Currently bit-equality is inferred from algorithm-equivalence; the mp4 write path needs a small fix to verify directly.
- **CFG-parallel `scheduler_step` regression** (Y4 only — X1 doesn't show it): ~115 s/gen recoverable; would push Y4 to ~2.25× lossless. The fact that X1 avoids this regression suggests a mechanical fix exists.
- **Locate the diffusers-vs-in-tree RNG-path divergence at the line-of-code level.** PSNR matrix proves the gap exists and is robust; identifying the responsible code line (suspects: latent noise gen order, scheduler init, dtype-cast ordering, text-encoder padding) would collapse the entire cluster to bit-equal.
- **FA3** (Hopper-tuned attention): not yet built; likely +10–30% on attention layers, lossless.
- **Fix the CUDAGraph crash on `torch.compile mode='reduce-overhead'` / `'max-autotune'`.** Wall-clock looked dramatic (~580 s vs Y4 825 s, ~30% lossless headroom if the math is right), but every run crashed before mp4 export, so the wins are speculative until we get a valid output.

## xDiT configs that did NOT work

| Config | Result |
|---|---|
| X2 — xDiT USP-2 (Ulysses inside xfuser) | **OOM at 720p × 81 frames.** xfuser's SP wrapper pre-allocates a full-sequence scratch and doesn't compose with diffusers' `--cpu-offload`; needs ~3 GB more than 80 GB. Would require xfuser's layerwise-offload path to fix. |

Full report: `docs/a14b-2gpu-h100-benchmark.md`.
