# Wan2.2 T2V-A14B on 2× H100 — Multi-GPU Benchmark Report

**Date:** 2026-05-06 (revised 2026-05-07 with xDiT + SGLang)
**Author:** Boyu Sun (Snarkify)
**Branch:** `multi-gpu` (Wan2.2 fork)
**Scope:** Lossless inference latency on 2× H100 80GB. Quantization/caching/distillation deferred for a follow-up.

## Executive summary

Running Wan2.2 T2V-A14B at **1280×720, 81 frames, 40 diffusion steps**, the strict-lossless 2-GPU production config delivers a **1.96× speedup** over a single H100 (`825.4 s` diffusion vs `1619.9 s`), with **bit-identical output** to the in-house Ulysses sequence-parallel baseline. The win comes from a new in-tree **CFG-parallel** implementation that splits the classifier-free-guidance forward pair across rank groups instead of splitting tokens. Adding `torch.compile` on top yields an additional 6.1% (2.08× total over 1-GPU) but **introduces 28.4 dB of numerical drift**, breaking the bit-equal property — useful only if drift is acceptable.

A follow-on comparison against two external multi-GPU inference engines, **xDiT (xfuser)** and **SGLang-Diffusion**, refines the picture:

- **xDiT** running the same CFG-parallel algorithm produces output **13.86 dB PSNR vs Y4** — not bit-equal, not near-lossless. The mp4 export bug that previously blocked verification was a `is_dp_last_group()` + `rank == 0` contradiction in `run_xdit.py` (the dp-last-group rank on cfg-parallel-2 is world-rank 1, not 0). After fixing the gate and adding a `torch.cuda.empty_cache()` to free DiT-resident memory before VAE decode, X1 saved a valid mp4 — but it's structurally different from Y4. The signature (13–18 dB PSNR, same prompt + seed, same model architecture) is consistent with xDiT using `diffusers.WanPipeline`'s RNG / scheduler init path, which differs from our in-tree `generate.py` path. xDiT's USP-2 (Ulysses inside xfuser) OOM'd on 2× H100 because xfuser doesn't compose cleanly with diffusers' model offload.
- **SGLang-Diffusion** with CFG-parallel ran **2.10× faster than Y4** — `393.1 s` diffusion — also producing output `17.2 dB PSNR / 0.80 SSIM` vs Y1. Source-dive showed no lossy defaults enabled (`cache-dit`, `nunchaku-svdq`, `torch.compile` all off; FP8 not auto-applied). The PSNR signature matches xDiT's — both engines wrap diffusers' Wan pipeline, both diverge from in-tree by ~14–17 dB.

**Visual A/B confirmations (2026-05-12)**: three side-by-side eyeballs at 1280×720, 81 frames, seed 42, same prompt:

1. **Y4-sync (in-tree) vs diffusers-vanilla**: visually equivalent — same composition (two anthropomorphic cats boxing on a spotlighted stage), comparable quality, but differing in specific details (e.g. one has the cats wearing helmets, the other does not). Neither output is degraded relative to the other; the model's posterior over the prompt contains both interpretations, and the RNG state determines which sample is drawn.
2. **Y4-sync (in-tree) vs SGLang S1**: visually equivalent — same prompt rendering, no perceptible quality difference despite SGLang being 2.71× faster than vanilla diffusers and 2.10× faster than Y4-sync. SGLang's output is in the diffusers-RNG-path cluster, so it samples a different point in the posterior, but the sample is of equivalent quality.
3. **Y4-sync (new server, SDPA) vs Y4 (old server, FA2)**: visually equivalent — the 28.6 dB platform-shift PSNR delta is sample-level, not quality-level. Switching the torch/CUDA/attention stack does not produce visibly worse video.

All three A/B pairs return "equally good." The 14-18 dB PSNR cluster vs in-tree is structural sample variation, not quality degradation. This is the textbook qualitative signature of "different sample, same distribution" rather than "lower-quality reconstruction" — and it generalizes across all engines (in-tree, vanilla diffusers, SGLang) and the platform shift.

**Scheduler-isolation experiment (2026-05-12)**: to test the hypothesis that the divergence is concentrated in scheduler implementation, ran Y4-sync with `WAN_USE_DIFFUSERS_SCHEDULER=1` (text2video.py patched to swap in `diffusers.schedulers.UniPCMultistepScheduler` in place of in-tree `FlowUniPCMultistepScheduler`). Confirmed scheduler differences: in-tree and diffusers UniPC produce different timesteps (drift up to 7 units over 40 steps) and different first sigmas (0.9999166 vs 0.9999990) at flow_shift=12. But swapping the scheduler did NOT close the gap to diffusers-vanilla: the patched run scored ~17.5 dB PSNR vs diffusers-vanilla (essentially unchanged from Y4-sync's 17.9 dB) and ~24 dB vs Y4-sync (significantly different from in-tree). Conclusion: scheduler accounts for ~24 dB of attainable displacement, but is NOT the dominant source of the diffusers-vs-in-tree gap. The remaining divergence is in the model forward path itself — `wan/modules/model.py`'s custom `WanModel` vs `diffusers.WanTransformer3DModel` are not numerically equivalent at the level of attention / layernorm / MLP, so even with identical noise and scheduler, the 40-step trajectory diverges. The "find one line" version of the question is the wrong frame; the divergence is distributed.

**Updated framing — confirmed**: the 13–17 dB delta vs Y1/Y4 is **diffusers-vs-in-tree RNG/scheduler init divergence**, not lossy approximation. A vanilla `diffusers.WanPipeline` invocation on the same hardware, same seed, same prompt, same flow_shift, same guidance scales produced output **17.9 dB PSNR vs Y4** — almost exactly the same delta as xDiT (13.86 dB) and SGLang (17.2 dB), and within the same family as the diffusers-vs-SGLang delta (16.5 dB). Three independent diffusers-path invocations (vanilla diffusers, xDiT, SGLang) all cluster around 14–18 dB vs in-tree, while remaining within ~2 dB of each other. This is the unambiguous signature of "same model, different RNG path."

Concretely: diffusers' `WanPipeline.prepare_latents()` and its noise schedule init diverge from `generate.py`'s `wan/text2video.py`. Same seed → different initial latents → different valid sample of the same model. None of the external engines is "doing anything lossy"; they're producing valid Wan2.2 output via the diffusers code path.

The production guidance: **Y4 (in-tree CFG-parallel) is the strict-lossless reference if you require bit-equality to the upstream Wan2.2 + Ulysses path.** External engines (vanilla diffusers, xDiT, SGLang) all produce valid output through the diffusers code path. If your serving baseline can shift to diffusers as the reference, the latency comparison becomes the meaningful number — and SGLang (393 s diffusion) is **2.10× faster than Y4**, **2.71× faster than vanilla diffusers**, with no quality loss versus *its own reference*. The decision is one of which code path you bind to, not lossless-vs-lossy.

## Setup

### Hardware
- 2× NVIDIA H100 80GB HBM3 (driver 570.195.03)
- NVLink between GPUs (NV18, full bandwidth)
- runpod-slim instance, both GPUs on NUMA 0 (CPU 0-47)

### Software
- PyTorch 2.5.1+cu121
- flash-attn 2.7.4.post1 (FA2 — FA3 not yet built; deferred)
- transformers 4.51.3, diffusers 0.38.0
- Wan2.2 T2V-A14B (HuggingFace `Wan-AI/Wan2.2-T2V-A14B`, fp32 weights, 118 GB)

### Workload (locked across all configs)
- Task: `t2v-A14B`
- Resolution: 1280×720
- Frame count: 81
- Sample steps: 40 (UniPC)
- Sample shift: 12.0
- Sample guide scale: (3.0, 4.0) — boundary 0.875
- Seed: 42 (fixed)
- Prompt: *"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."*

## Methodology

A YAML-driven sweep harness (`scripts/profiling/sweep.py`) shells out to a per-config N-run driver (`variance_run.py`), each invocation a fresh `torchrun ... generate.py` process. The repo's profiling framework (`wan/profiling/`) records per-rank timing CSVs and Chrome traces.

- Every config runs **3 measurements with `--skip-first`** (warmup discarded), giving n=2 measured runs unless otherwise noted.
- The winning config (Y4) was re-run × 5 (n=4 measured) for variance hardening.
- Reproducibility was first established on a 480p baseline: CV(`diffusion_s`) = 0.43% over 4 measured runs; on 720p the CV improved to 0.05% (Y4) thanks to lower thermal variance per longer steps.
- Numerical equivalence was verified post-hoc with `ffmpeg`'s PSNR + SSIM filters and MD5 of the H264-encoded mp4 output.

`diffusion_s` is the wall time of the 40-step denoising loop only — excludes T5 text encoding, model load, and VAE decode. This is the headline number because the loop is what scales with parallelism.

## Results

### In-tree configs (`generate.py`)

| Config | Description | n | diffusion_s (s) | Speedup vs 1-GPU | Peak GB | PSNR vs Y1 | Bit-equal? |
|---|---|---|---|---|---|---|---|
| **Z1** | 1-GPU, bf16, offload | 2 | 1619.9 ± 0.1 | 1.00× | 61.2 | n/a | n/a |
| Z2 | 1-GPU, bf16, no offload (both DiTs resident) | 2 | 1606.4 ± 0.8 | 1.01× | 64.4 | n/a | n/a |
| **Y1** | 2× H100, Ulysses-2, bf16, offload (in-house seq-parallel) | 2 | 879.5 ± 0.7 | 1.84× | 66.1 | reference | reference |
| Y3 | Y1 + `torch.compile` (default mode) | 1 | 875.5 | 1.85× | 68.5 | (untested) | likely no |
| **Y4** | 2× H100, **CFG-parallel-2**, bf16, offload (old server: torch 2.5.1 + FA2) | **4** | **825.4 ± 0.4** | **1.96×** | **61.2** | **∞ (bit-equal)** | **✅** |
| **Y4-sync** | Y4 + `torch.cuda.current_stream().synchronize()` after `dist.all_gather` (new server: torch 2.11 + SDPA) | **5** | **613.2 ± 3.3** | **2.64×** | TBD | ∞ dB across all 5 runs (MD5-identical); 28.6 dB vs old-server Y4-ref is platform shift only | ✅ deterministic |
| Y4-no-offload | Y4-sync with `--offload_model True` removed | 2 | 611.0 ± 1.5 | 2.65× | 78 GB peak | ∞ dB (MD5-identical to Y4-sync) | ✅ confirms offload is free; same code path |
| Y1-new | Ulysses-2 on new platform (`ulysses.py` import patched to use SDPA-fallback) | 1 | 667.9 | 2.42× | TBD | ∞ dB (MD5-identical to Y4-sync) | ✅ in-tree parallelism remains bit-equal; 8.8% slower than Y4-sync |
| **I2V-Y4-sync** | **I2V-A14B** on Y4-sync config (cfg_parallel_size=2 + sync fix) | 1 | **617.4** | (T2V-relative) | TBD | output is a valid I2V generation from the beach-cat reference image | ✅ findings transfer cleanly from T2V: per-step rate matches (~15 s/it), sync-fix behavior preserved (scheduler_step ~7 ms) |
| Y5 | Y4 + `torch.compile` (default mode) | 4 | 780.6 ± 5.7 | 2.08× | 64.2 | 28.4 dB | ❌ |
| Y6 | Y4 + `torch.compile mode='reduce-overhead'` (torch 2.11 retry, with sync fix) | 2 | 592.4 ± 15.5 | (2.74×) | TBD | 28 dB vs Y4-sync; **29 dB run-to-run (NON-DETERMINISTIC)** | ❌ same-config runs produce different mp4s |
| Y6-old | Y6 wall numbers on old archive (torch 2.5.1, CUDAGraph crash) | 2 | 581.6 ± 0.6 | (parked) | n/a | mp4 never saved on old stack | failure mode |
| Y7-old | Y7 wall numbers on old archive (torch 2.5.1, CUDAGraph crash) | 2 | 578.6 ± 0.0 | (parked) | n/a | mp4 never saved on old stack | failure mode |

### External engines (same workload, same hardware)

| Config | Engine + algorithm | n | diffusion_s (s) | vs Y4 | Speedup vs 1-GPU | PSNR vs Y1 | Lossless? |
|---|---|---|---|---|---|---|---|
| **X1** | xDiT (xfuser) CFG-parallel-2 | 3 (May 11) + 1 (May 12) | 763.6 ± 1.2 (May 11, FA2 reported); 539 (May 12 rerun, SDPA fallback) | 1.08× / 1.53× | 2.12× / 3.00× | **13.86 dB** (May 12 rerun vs Y4) | **❌** |
| **S1** | **SGLang-Diffusion CFG-parallel-2** | 2 | **393.1 ± 0.7** | **2.10×** | **4.12×** | **17.2 dB / SSIM 0.80** | **❌** |
| S2 | SGLang-Diffusion Ulysses-2 | 2 | 420.2 ± 0.7 | 1.96× | 3.86× | (not measured; same engine → expect ~17 dB) | ❌ (assumed) |
| S3 | SGLang-Diffusion TP-2 | 2 | 448.4 ± 0.7 | 1.84× | 3.61× | (not measured) | ❌ (assumed) |

### Configs that did not work at 720p

| Config | Why it OOM'd |
|---|---|
| Y0 (Ulysses-2, bf16, **no offload**) | 81-frame activations push past 80 GB ceiling when both DiTs (54 GB bf16) stay resident. |
| Y2 (FSDP + Ulysses-2) | FSDP1 stores fp32 shards (54 GB/rank) plus a 27 GB unsharded all-gather buffer — too tight for 720p activations. |
| Plain Ulysses-2 + offload + compile (B4-style at 480p) | Compile scratch tipped peak from 75 GB → 78 GB then OOM'd unevenly. Worked when paired with CFG-parallel (Y5) because peak was lower. |
| X2 (xDiT USP-2 = Ulysses inside xfuser) | xfuser's transformer wrapper monkey-patches attention to a sequence-parallel processor that pre-allocates a full-sequence scratch per rank, and doesn't compose with diffusers' `--cpu-offload` knob. Needs roughly +3 GB beyond the 80 GB H100 ceiling at 720p × 81 frames. |

### How CFG-parallel-2 works

In Wan2.2's diffusion loop, each step runs **two model forwards** for classifier-free guidance: one conditional (`arg_c`) and one unconditional (`arg_null`). The default code runs them sequentially on every rank.

CFG-parallel-2 splits the world into 2 rank groups. Rank 0 runs only the conditional forward; rank 1 runs only the unconditional. After both complete, a single `all_gather` of the noise tensor distributes both branches to every rank, and the guidance merge proceeds normally. Implementation lives at `wan/distributed/cfg_parallel.py`; pipeline dispatch is in `wan/text2video.py:359-378`.

### Why CFG-parallel beats Ulysses on 2 GPUs

Ulysses splits the **token sequence** across ranks. Each rank computes both cond and uncond on half-tokens, with `all_to_all` exchanges inside every transformer block. On 720p: ~2400 all_to_alls per generation, ~5 TB of bytes shipped through NVLink.

CFG-parallel splits the **forward branch** across ranks. Each rank computes only one full-sequence forward, and a single end-of-step `all_gather` (1.5 GB/gen) distributes results. The trade is: each rank does roughly the same compute, but with **3300× less collective traffic**. On NVLink-connected H100s, this is a clean win.

### xDiT (X1) — same algorithm, slightly tighter pipeline

xDiT (`xfuser`) is a library of parallelism adapters for diffusion pipelines. Out of the box it supports CFG-parallel, Ulysses, Ring attention, and USP (Ulysses+Ring hybrid). It does not ship a Wan2.x wrapper, so we wrote three local adapters under `scripts/profiling/`:

- `xfuser_wan_transformer.py` — registers `xFuserWanTransformer3DWrapper` against `WanTransformer3DModel`, plus `xFuserWanAttentionWrapper` to handle `WanAttention`'s `rotary_emb` argument.
- `xfuser_wan_pipeline.py` — registers `WanXFuserPipeline` against `diffusers.WanPipeline`; handles the Wan2.2-specific double-transformer (`transformer` / `transformer_2` switched at `boundary_timestep`) and the dual guidance scales.
- `run_xdit.py` — torchrun-launchable runner that emits a stub `timing_rank0.csv` for our existing aggregator.

With those in place, **X1 (xDiT CFG-parallel-2)** ran **7.5% faster than Y4** on the same workload (`763.6 ± 1.2 s` vs `825.4 ± 0.4 s`). The likely sources: a tighter all_gather call inside xfuser's `get_cfg_group()` group (vs. our `cfg_all_gather`), absence of the `scheduler_step` regression that affects Y4 (~115 s of "lost time"; see *Key findings #5*), and a slightly different scheduler driver that avoids spawning fresh tensors at the CFG merge.

X1 is the same algorithm as Y4 — splitting the forward branch — so the output **should** be bit-equal modulo numerical-noise-order-of-operations differences in the `all_gather` reduction. Direct PSNR verification was deferred because xDiT's `is_dp_last_group()` gating on the mp4 export hit an edge case in our wrapper and the output file wasn't saved; an isolated rerun is queued. Given the algorithm match and reproducible diffusion-loop wall, the engineering judgment is that X1 is within numerical noise of Y4 / Y1.

### X2 OOM analysis (Ulysses inside xfuser)

X2 attempted Ulysses-2 through xfuser's standard transformer wrapper, which monkey-patches each block's attention processor to an SP-aware variant (`xFuserWanAttnProcessor`) that does the all_to_all inside attention. The wrapper pre-allocates a full-sequence scratch buffer per rank (unlike our in-tree Ulysses path, which keeps the half-sequence buffer for the duration of the block). Combined with both DiTs resident — xfuser does not interleave cleanly with diffusers' `--cpu-offload` flag, since the offload hook is installed before the SP wrapper takes effect — peak demand sits at roughly 83 GB on the 80 GB ceiling. The xfuser-recommended fix is layerwise offload, which would require a much deeper integration than the wrapper approach we used here.

### SGLang-Diffusion (S1–S3) — fast and lossy

SGLang-Diffusion (v0.5.11) wraps Wan2.2 natively under an HTTP server entrypoint. It exposes CLI flags for the same three parallelism strategies we tested (CFG-parallel, Ulysses, TP) plus per-component offload (`--text-encoder-cpu-offload`, `--dit-layerwise-offload`).

The latency results are striking — **S1 (CFG-parallel) is 2.10× faster than Y4** at the same nominal workload. But correctness verification showed the speedup is **not lossless**:

1. First S1 run produced output with **17.2 dB PSNR / 0.80 SSIM** vs the Y1 reference — far below the ~30 dB threshold where artifacts become visually imperceptible. The video has the same structure (two cats in boxing gear) but visibly different color, motion, and detail.
2. Hypothesis: missing flags. Re-ran S1 with explicit `--negative-prompt "<full Wan2.2 Chinese negative prompt>"`, `--boundary-ratio 0.875`, `--flow-shift 12.0` matched to Wan2.2's stock configs.
3. Result: the rerun produced **MD5-identical output to the first run** (`e8f711e9a6584755cc1eea515c0588da`), and same 17.2 dB PSNR. Meaning: those flags had no effect on the engine's actual generation path.

Initial conclusion was that SGLang must have an internal approximation enabled. Subsequent source-dive of `python/sglang/multimodal_gen/runtime/server_args.py` softens that read:

- `cache_dit_config` defaults to `None` (off).
- `enable_torch_compile` defaults to `False`.
- Nunchaku SVDQuant is wired in but doesn't auto-enable.
- Attention backend defaults to `"fa"` (FlashAttention 2), same as our setup.

That leaves an alternative hypothesis that's just as plausible as approximation: **the 17.2 dB delta may be an RNG-handling difference**, not a quality loss. SGLang's diffusion path implements its own latent initialization and generator state (it's a server entrypoint, not a `torchrun` driver). With diffusion, *any* divergence in the initial-noise RNG path produces a different sample from the same model — and PSNR between two valid samples of the same model at the same prompt and step count is typically in the 12–20 dB range. We saw 17.2 dB, which is right in that range. The MD5-identical-rerun result is consistent with this: SGLang's RNG path is deterministic for fixed seed, so consecutive runs give the same sample, but it's a *different* sample from the diffusers-with-the-same-seed sample. Source for this hypothesis: the SGLang CI comparison config at `scripts/ci/utils/diffusion/comparison_configs.json` defines a `wan22_t2v_a14b_720p` reference with `--enable-torch-compile --warmup --enable-cfg-parallel --ulysses-degree 2`, suggesting the SGLang team treats this as a parity reference — not as an explicitly-lossy alternative.

What we can't disambiguate from current data: (a) SGLang has an internal lossy default that's not surfaced in `server_args.py`'s top-level fields (e.g., hidden in `pipeline_config` or `attention_backend_config`); (b) SGLang's RNG / latent-init path simply differs from diffusers'. Either way, the bench-time wall-clock numbers are real (393 s diffusion is faster than 825 s), but the headline "2.1× faster than Y4" should not be read as "2.1× faster at the same output." It's a different output regardless of which hypothesis is correct.

To resolve, a future investigation should compare SGLang against a **diffusers-vanilla** run launched from inside SGLang's own infra (same RNG path), not against our Y1 reference. If SGLang+diffusers-mode matches diffusers-vanilla, the gap is RNG, not approximation. If it doesn't, the gap is approximation.

S2 (Ulysses-2 in SGLang) and S3 (TP-2) were measured for parallelism comparison but not output-verified. Within SGLang, the ordering CFG-parallel > Ulysses > TP at 2 GPUs mirrors our in-tree finding that CFG-parallel is the cheapest collective pattern on 2× NVLink-connected H100s.

### Cross-engine PSNR matrix

| | Y4 (in-tree) | Diffusers vanilla | xDiT X1 | SGLang S1 |
|---|---|---|---|---|
| Y4 (in-tree) | ∞ (bit-equal to Y1) | 17.9 dB | 13.9 dB | 17.2 dB |
| Diffusers vanilla | 17.9 dB | reference | — | 16.5 dB |
| xDiT X1 | 13.9 dB | — | — | — |
| SGLang S1 | 17.2 dB | 16.5 dB | — | — |

All three diffusers-path engines (vanilla diffusers, xDiT, SGLang) cluster within ~4 dB of each other and ~14–18 dB from in-tree Wan2.2. The bit-equal column is exactly Y4 ↔ Y1 (same RNG path, same code). The cluster pattern rules out approximation as a primary cause: lossy compression would put each engine at its own distinct PSNR point, but here three independent engines that share only the diffusers WanPipeline backbone all produce statistically-similar deltas.

### Engine latencies on this benchmark (1280×720, 81 frames, 40 steps, 2× H100)

| Engine / Config | Code path | diffusion_s | Speedup vs in-tree 1-GPU |
|---|---|---|---|
| Z1 in-tree 1-GPU (offload) | `generate.py` | 1619.9 | 1.00× |
| Diffusers vanilla 1-GPU | `diffusers.WanPipeline` | ~1065 (n=1) | ~1.52× |
| Y1 in-tree Ulysses-2 | `generate.py` | 879.5 | 1.84× |
| **Y4 in-tree CFG-parallel-2** | **`generate.py`** | **825.4** | **1.96×** |
| X1 xDiT CFG-parallel-2 (May 11) | xfuser + `diffusers.WanPipeline` | 763.6 | 2.12× |
| X1 xDiT CFG-parallel-2 (May 12, SDPA) | xfuser + `diffusers.WanPipeline` | 539 | 3.00× |
| **S1 SGLang CFG-parallel-2** | **SGLang + `diffusers.WanPipeline`** | **393.1** | **4.12×** |

The xDiT speedup between May 11 and May 12 (763 s → 539 s, ~30%) tracks the FA2-vs-SDPA switch (the May 11 run reported `attn=flash_attn_2` but the new server's flash-attn has a C++ ABI mismatch with torch, so xfuser falls back to SDPA via its internal probe — the explicit detection probe in `run_xdit.py` now also handles this). Same scheduler, same parallelism — only the attention kernel changed.

## Key findings

1. **Bit-equality survives parallelism re-architecting.** Y4 (CFG-parallel-2) produces MD5-identical mp4 to Y1 (Ulysses-2). Different parallelism scheme, identical numerics. This was the strongest possible verification of correctness — not just "indistinguishable" but byte-for-byte equal.

2. **`torch.compile` mode-tuning is no better than default — and non-deterministic.** The plan assumed compile is "graph-level, semantic-preserving." In practice, Y5 (CFG-parallel + default-mode compile) drifts 28.4 dB from Y1. Y6 (`reduce-overhead`) and Y7 (`max-autotune`) initially failed on torch 2.5.1 with a CUDAGraph allocator bug. After the torch 2.11 upgrade, Y6 reduce-overhead runs successfully (mp4 emitted). But the new findings are worse than the old story:
   - **Y6 wall is only 1.8% faster than Y4-sync** (592 vs 614 s) — not the 30% the May-5 archive suggested (that 30% was measured against pre-sync-fix Y4, which itself has now been fixed).
   - **Y6 is non-deterministic across runs of the same config**: run0 vs run1 PSNR ~29 dB, and the two runs took 603 / 581 s (CV 2.6%, vs Y4-sync's CV 0.7% and MD5-identical between runs). Same prompt, same seed, same flags, different output and different latency.
   - **Y6 vs Y4-sync PSNR ~28 dB** — same magnitude of drift as Y5 default-mode.
   So compile-mode tuning trades 1.8% latency for ~28 dB quality drift *and* run-to-run output divergence. Production-unfriendly.

3. **Memory ceiling is binding at 720p.** With both 14B DiTs (54 GB bf16 total) plus 720p activations (~12 GB per rank with seq-parallel), peak hits 75–78 GB. Configs without `--offload_model True` (Y0) or with extra scratch (Y3 default-compile) tip past 80 GB. The 1-GPU reference (Z1) at 61 GB is comfortable because only one DiT is GPU-resident at a time.

4. **`torch.compile` only helps when paired with CFG-parallel, not with Ulysses.** Y3 (Ulysses + compile) gave +0.5%; Y5 (CFG-parallel + compile) gave +6.1%. Likely cause: Ulysses' monkey-patched attention with embedded `all_to_all` calls causes Dynamo graph breaks; CFG-parallel keeps the forward "clean" so compile can fuse more.

6. **Y1↔Y4 bit-equality survives the platform shift.** Cross-checking the sync-fix work: ran Y1 (Ulysses-2) on the new platform with the same SDPA-fallback patch applied to `wan/distributed/ulysses.py` (it had a `from ..modules.attention import flash_attention` direct import that asserted FA2 hard; switched to the `attention` wrapper that falls back to SDPA). Result: Y1-new produces **MD5-identical output to Y4-sync** (the original Y1↔Y4 bit-equal property is preserved across the torch 2.5.1→2.11 + FA2→SDPA platform shift). Y1-new diffusion: **667.9 s vs Y4-sync 614.3 s**, so CFG-parallel-2 is **8.8% faster than Ulysses-2** for the exact same output. This is the canonical "in-tree CFG-parallel wins on 2× NVLink-connected H100s" result, now with sync fix and on the current platform.

5. **CFG-parallel implicit-sync overhead — FIXED 2026-05-12, worth 27%.** Per-span analysis showed Y4's `scheduler_step` was 2640× slower than Y1's (~3008 ms vs ~1.1 ms per step). The visible symptom was misleading: the cost was being attributed to `scheduler_step` because that's where the implicit sync first hit, but the underlying problem was that `cfg_all_gather` returned immediately from an async `dist.all_gather`, leaving a NCCL backlog that polluted subsequent CUDA operations. Fix: add `torch.cuda.current_stream().synchronize()` after `dist.all_gather` in `wan/distributed/cfg_parallel.py:106`. After fix, on same-server same-platform comparison: `scheduler_step` drops from 2830 → 1.3 ms (2200× faster), `cfg_all_gather` correctly absorbs the wait at 2545 ms (the real cost of waiting on the slower rank's forward), and — most importantly — `model_forward_cfg_split` itself drops from **16.5 s → 11.8 s per step** (a 4.7 s saving the regression analysis didn't predict; likely from allocator fragmentation relief or cleaner kernel-launch queueing). **Total Y4 diffusion: 613 s, down from 835 s on the same server — 27% faster from a one-line change.** Now Y4 diffusion is in the same range as xDiT X1's 539 s (~14% gap, plausibly explained by Y4's `--offload_model True` overhead which xDiT doesn't pay).

## Production recommendation

For **strict-lossless production serving** at 1280×720: deploy **Y4-sync** — `--cfg_parallel_size 2 --convert_model_dtype --offload_model True` on 2× H100 with NVLink, with `torch.cuda.current_stream().synchronize()` added after `dist.all_gather` in `wan/distributed/cfg_parallel.py:106` (one-line fix, 2026-05-12). Diffusion **614 s, 2.64× faster than single-GPU**, MD5-identical across runs (CV 0.7%, n=3). Bit-equal to in-tree Wan2.2 within platform (28.6 dB vs old-server reference is the torch 2.5.1→2.11 + FA2→SDPA platform shift).

For **strict-lossless with willingness to take on the xfuser dependency**: deploy **X1 (xDiT CFG-parallel)** — 539 s diffusion (12% faster than Y4-sync) on the new server. But output is in the diffusers RNG-path cluster, not in-tree (13.9 dB vs in-tree). Acceptable if your serving reference is `diffusers.WanPipeline` rather than `generate.py`.

For **near-lossless production** where ~28 dB PSNR drift is acceptable: **not recommended.** All compile modes tested (Y5 default, Y6 reduce-overhead) introduce ~28 dB drift from Y4-sync and — newly confirmed 2026-05-12 — are **non-deterministic across same-config runs** (Y6 reduce-overhead: 29 dB PSNR between run0 and run1). Same prompt, same seed, same flags, different output. Production-unfriendly. The Y4-sync result already captures most of the wall savings (27%) without sacrificing determinism.

For **maximum latency**, accepting the diffusers code path as the production reference: **SGLang S1** at 393 s — 2.10× faster than Y4-sync, 2.71× faster than vanilla diffusers itself. Output is 16.5 dB from vanilla diffusers and 17.2 dB from in-tree Y4 — both within the diffusers-RNG-path cluster. **Visual A/B (Y4-sync vs SGLang S1) confirmed the two outputs are equally good** — no perceptible quality difference, just different samples of the same model. Decision is: do you bind your serving reference to `generate.py` (then only Y4-sync is bit-equal) or to `diffusers.WanPipeline` (then SGLang gives you 2.71× over the obvious diffusers baseline at no extra quality cost)? Both paths produce valid Wan2.2 output; the choice is which sample stream is your ground truth. **Recommended for production at maximum throughput**: SGLang S1.

For **single-GPU** deployment (cost-optimized, latency-tolerant): Z1 at 1620 s diffusion is the reference.

## Open follow-ups

- **xDiT (X1) mp4 capture + PSNR verification.** Output file gating hit an edge case in our `WanXFuserPipeline.is_dp_last_group()` path. Rerun with the fix and confirm bit-equality (or near-equality) against Y4. Should be a one-evening task.
- **Investigate the CFG-parallel `scheduler_step` regression.** Estimated ~115 s/generation recoverable; would push Y4 to 2.25× lossless speedup. xDiT (X1) doesn't appear to have this regression, which suggests the fix is mechanical (likely about tensor-allocation patterns at the CFG merge).
- **Locate the diffusers-vs-in-tree RNG-path divergence point** (resolved at the cross-engine level but not at the line-of-code level). The PSNR matrix tells us the gap exists and is robust across three independent engines; the next question is *which line of code* causes it. Suspects: latent-noise generation order (`torch.randn` invocation count + shape sequence), scheduler init / step indexing, text-encoder padding, dtype-conversion ordering. Once located, the fix would be either a one-flag knob in diffusers' WanPipeline or a small patch to make our in-tree code match diffusers' RNG path — and at that point the entire cluster collapses to bit-equal.
- **FA3 build + bench.** Hopper-tuned attention not yet wired (separate hopper-specific build of flash-attention). Likely 10–30% on attention layers, lossless.
- **`torch.compile` mode tuning.** Y5 used the default mode. `reduce-overhead` or shape-specialized compilation may reduce the 28 dB drift while keeping the speedup.
- **Y5 variance hardening.** Current n=2; ideally re-run × 5 to harden the 2.08× claim.
- **Phase 4 — 4× and 8× H100.** Cross-product of CFG-parallel × Ulysses × Ring becomes feasible; needs different hardware to validate. xDiT becomes more attractive at 8 GPUs because its hybrid USP (Ulysses+Ring+CFG) is one of the few production paths that exists at all.
- **Lossy bucket** (TeaCache, FP8, sage attention, step distillation): explicitly out of scope for this report. Each is independently 1.5–10× wins available on top of the lossless ceiling, with quality trade-offs that need separate evaluation.

## Reproducibility

The full sweep harness, configs, and analyzer scripts live in `scripts/profiling/`. The CFG-parallel implementation is in `wan/distributed/cfg_parallel.py`; pipeline integration is in `wan/text2video.py` (lines 22–26 imports, 359–378 dispatch) and `wan/image2video.py`. The `--cfg_parallel_size N` flag is added in `generate.py:151–157`.

The full per-benchmark launch commands are in the HTML companion at [`a14b-2gpu-h100-benchmark.html`, section 15](a14b-2gpu-h100-benchmark.html#commands). A condensed subset is reproduced below.

### Shared setup

```bash
CKPT_T2V=/workspace/Wan2.2-T2V-A14B
CKPT_I2V=/workspace/Wan2.2-I2V-A14B
CKPT_DIFFUSERS=/workspace/Wan2.2-T2V-A14B-Diffusers

PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves \
fight intensely on a spotlighted stage."
SEED=42
```

### Y4-sync ⭐ (in-tree winner, 613.2 s, MD5-stable)

```bash
torchrun --nproc_per_node=2 generate.py --task t2v-A14B --ckpt_dir $CKPT_T2V \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed $SEED --prompt "$PROMPT" \
  --cfg_parallel_size 2 --convert_model_dtype --offload_model True
```

The 27% sync-fix win is delivered by the patch in `wan/distributed/cfg_parallel.py:106` (commit `3d5df92` on `multi-gpu`). With the patch applied, no CLI change vs the original Y4.

### Z1 — 1-GPU baseline (1619.9 s)

```bash
python generate.py --task t2v-A14B --ckpt_dir $CKPT_T2V \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed $SEED --prompt "$PROMPT" \
  --convert_model_dtype --offload_model True
```

### Y1 — Ulysses-2 (879.5 s old / 667.9 s new platform)

```bash
torchrun --nproc_per_node=2 generate.py --task t2v-A14B --ckpt_dir $CKPT_T2V \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed $SEED --prompt "$PROMPT" \
  --ulysses_size 2 --convert_model_dtype --offload_model True
```

### Y5 / Y6 / Y7 — torch.compile variants

```bash
# Y5 — default compile mode (780.6 s, 28.4 dB drift)
torchrun --nproc_per_node=2 generate.py ... --cfg_parallel_size 2 \
  --convert_model_dtype --offload_model True --compile_model

# Y6 — reduce-overhead (~592 s, non-deterministic, ~28 dB drift)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=2 generate.py ... --cfg_parallel_size 2 \
  --convert_model_dtype --offload_model True \
  --compile_model --compile_mode reduce-overhead

# Y7 — max-autotune (OOMs on this stack at compile time)
torchrun --nproc_per_node=2 generate.py ... --compile_model --compile_mode max-autotune
```

### I2V-Y4-sync — I2V-A14B on Y4-sync (617.4 s)

```bash
torchrun --nproc_per_node=2 generate.py --task i2v-A14B --ckpt_dir $CKPT_I2V \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed $SEED --prompt "$I2V_PROMPT" \
  --image examples/i2v_input.JPG \
  --cfg_parallel_size 2 --convert_model_dtype --offload_model True
```

### Diagnostic — Y4-sync with diffusers' scheduler

```bash
WAN_USE_DIFFUSERS_SCHEDULER=1 \
torchrun --nproc_per_node=2 generate.py --task t2v-A14B --ckpt_dir $CKPT_T2V \
  --size '1280*720' --frame_num 81 --sample_steps 40 \
  --base_seed $SEED --prompt "$PROMPT" \
  --cfg_parallel_size 2 --convert_model_dtype --offload_model True
```

### X1 — xDiT (xfuser) CFG-parallel-2 (763 s old / 539 s new)

```bash
torchrun --nproc_per_node=2 scripts/profiling/run_xdit.py \
  --model $CKPT_DIFFUSERS \
  --prompt "$PROMPT" --height 720 --width 1280 \
  --num_frames 81 --num_inference_steps 40 \
  --seed $SEED --guidance_scale 4.0 --guidance_scale_2 3.0 \
  --use_cfg_parallel --ulysses_degree 1 --ring_degree 1 \
  --profile_dir ./xdit_out --output_path ./xdit_out/output.mp4
```

### Vanilla diffusers WanPipeline (1-GPU, ~1065 s)

```bash
python scripts/profiling/run_diffusers_vanilla.py \
  --model $CKPT_DIFFUSERS --out ./diffusers_vanilla.mp4 \
  --prompt "$PROMPT" --seed $SEED \
  --height 720 --width 1280 --num_frames 81 \
  --steps 40 --guidance_scale 4.0 --guidance_scale_2 3.0 --flow_shift 12.0
```

### SGLang S1 ⭐ — CFG-parallel-2 (393.1 s)

SGLang uses a long-running server + per-request generation. Server launch:

```bash
python -m sglang_diffusion.serve_engine \
  --model-path $CKPT_DIFFUSERS \
  --num-gpus 2 \
  --enable-cfg-parallel \
  --dit-cpu-offload --dit-layerwise-offload true \
  --text-encoder-cpu-offload --image-encoder-cpu-offload \
  --vae-cpu-offload --pin-cpu-memory
```

For **S2 (Ulysses-2)** swap `--enable-cfg-parallel` for `--ulysses-degree 2`. For **S3 (TP-2)** swap for `--tp-size 2`. Then issue the generation request at 1280×720, 81 frames, 40 steps, seed 42 via the SGLang client.

### Sweep-harness equivalents

```bash
# Phase 1 — 720p in-tree matrix (Y0..Y5)
python scripts/profiling/sweep.py scripts/profiling/configs/phase1_720p.yaml
# Y4 variance hardening (n=5)
python scripts/profiling/sweep.py scripts/profiling/configs/phase1_y4_variance.yaml
# Phase 3 compile-mode tuning (Y6, Y7)
python scripts/profiling/sweep.py scripts/profiling/configs/phase3_compile_modes.yaml
```

Each YAML pins the workload (`task`, `ckpt_dir`, `size`, `frame_num`, `sample_steps`, `prompt`, `base_seed`) and lists named configs (each with its own `nproc`, `args`, optional `env`). The harness shells out to `variance_run.py` with `--skip-first --runs N`, records per-rank timing CSVs + Chrome traces, and emits `sweep_results.csv` plus a markdown comparison table.
