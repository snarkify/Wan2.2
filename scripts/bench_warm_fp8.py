"""Path B warm-perf bench harness.

Drives the same `WanTI2V.generate()` path the demo server's worker uses
(`server/worker.py::_run_job`), but without the FastAPI / asyncio /
JobStore surface. Goal: get a clean per-gen wall-time + peak-alloc
number we can hold constant across Path B's three phases (fp8-fast,
sageattention, torch.compile).

Reuses the production singleton pattern: build the pipeline once, run
N consecutive `generate()` calls with `reuse=True` and
`offload_model=True`, save one mp4 per gen, dump JSON.

Honored env vars (forward-compatible across phases):
    WAN_DEMO_QUANT     bf16 | fp8 | fp8_fast (fp8_fast lands in Phase 1)
    WAN_DEMO_ATTN      flash | sage          (Phase 2)
    WAN_DEMO_COMPILE   0 | 1                  (Phase 3)
    WAN_DEMO_TEACACHE_THRESH  float          (Phase 4 spike; unset = off)

CLI args mirror the canonical bench config from
`docs/path-b-acceptance.md`. The defaults are the bench config; pass
`--quant fp8_fast` (or set `WAN_DEMO_QUANT` directly) to switch.

Run on gpu6 from a separate tmux session; do NOT collide with the
production demo-server (port 29600 / output dir). This harness picks
its own torch.distributed master port and writes only into the bench
output dir.

Typical use:
    source .env.demo-server                # exports WAN_DEMO_CKPT_DIR etc.
    python scripts/bench_warm_fp8.py --num-gens 3 --quant fp8

Output:
    bench_artifacts/<branch>_<ts>_<config>.json   (per-gen wall, mean step,
                                                    peak alloc, env)
    <out_dir>/gen_<i>.mp4                          (one per gen)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


_CANONICAL_PROMPT = "A cat walking in a sunlit meadow"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _maybe_init_dist() -> None:
    """If launched under `torchrun`, init the process group exactly like
    the demo server does. If launched as plain `python`, do nothing —
    `WanTI2V` handles world=1 without a process group."""
    if "RANK" not in os.environ:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _mem_snapshot(label: str) -> dict[str, int]:
    """Mirrors server.worker._mem_snapshot but returns the numbers so we
    can stash them in JSON instead of just logging."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    alloc = torch.cuda.memory_allocated(local_rank) // (1024 * 1024)
    reserved = torch.cuda.memory_reserved(local_rank) // (1024 * 1024)
    peak_alloc = torch.cuda.max_memory_allocated(local_rank) // (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(local_rank) // (1024 * 1024)
    free_b, total_b = torch.cuda.mem_get_info(local_rank)
    used_mb = (total_b - free_b) // (1024 * 1024)
    snap = {
        "label": label,
        "alloc_mb": alloc,
        "reserved_mb": reserved,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "nvml_used_mb": used_mb,
    }
    logging.info(
        "mem[%s] alloc=%dMB reserved=%dMB peak_alloc=%dMB peak_reserved=%dMB nvml.used=%dMB",
        label, alloc, reserved, peak_alloc, peak_reserved, used_mb,
    )
    return snap


def _build_pipeline(ckpt_dir: str, init_on_cpu: bool = False):
    """Mirror of server.worker._get_or_build_pipeline, minus the global
    singleton (this harness owns the lifetime; one process = one bench
    run, no need for module-level state).

    `init_on_cpu=True` keeps the DiT on CPU at construction so T5+DiT
    don't both occupy 4090 VRAM at the same time. Required for the
    bf16 quality-reference path (T5-bf16 ~10 GB + DiT-bf16 ~10 GB +
    overhead exceeds 24 GB if both are GPU-resident). Production uses
    `init_on_cpu=False` because fp8 DiT is half the size."""
    from wan import configs as wan_configs
    from wan.textimage2video import WanTI2V

    cfg = wan_configs.WAN_CONFIGS["ti2v-5B"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    use_fsdp = world_size > 1
    use_sp = world_size > 1

    logging.info(
        "building WanTI2V: rank=%d world=%d quant=%s attn=%s compile=%s init_on_cpu=%s ckpt=%s",
        rank, world_size,
        os.environ.get("WAN_DEMO_QUANT", "bf16"),
        os.environ.get("WAN_DEMO_ATTN", "flash"),
        os.environ.get("WAN_DEMO_COMPILE", "0"),
        init_on_cpu, ckpt_dir,
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    t0 = time.perf_counter()
    pipeline = WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=use_fsdp,
        use_sp=use_sp,
        t5_cpu=False,
        init_on_cpu=init_on_cpu,
        convert_model_dtype=True,
    )
    elapsed = time.perf_counter() - t0
    logging.info("pipeline built in %.2fs", elapsed)

    # Phase 3: apply torch.compile to the DiT singleton when
    # WAN_DEMO_COMPILE=1. Mirrors `server/worker.py:_get_or_build_pipeline`.
    # The compile is lazy (Dynamo+Inductor capture happens on first
    # forward), so the wall time of gen 1 absorbs the compile cost.
    compile_env = os.environ.get("WAN_DEMO_COMPILE", "0").strip().lower()
    if compile_env in ("1", "true", "yes", "on"):
        from wan.distributed.compile_util import (
            compile_dit, resolve_compile_mode,
        )
        mode = resolve_compile_mode(os.environ.get("WAN_DEMO_COMPILE_MODE"))
        logging.info("applying torch.compile mode=%s", mode)
        compile_dit(pipeline, mode=mode, enabled=True)

    return pipeline, elapsed


def _run_one_gen(
    pipeline,
    *,
    prompt: str,
    size: tuple[int, int],
    frame_num: int,
    seed: int,
    sampling_steps: int,
    save_path: str,
    teacache_enabled: bool = False,
) -> dict[str, Any]:
    """Run a single generate() and return per-gen stats. Save mp4 inline,
    drop the tensor before returning (mirrors the inline-save fix in
    server/worker.py that closed a 180 MB/job leak)."""
    import gc
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    enter = _mem_snapshot("enter_run_job")
    torch.cuda.reset_peak_memory_stats(local_rank)

    if teacache_enabled:
        # Reset cnt/accumulators/residuals so each gen starts fresh.
        # Without this, gen N+1 inherits gen N's even/odd parity offset
        # and residuals — the first ~2 forwards of gen N+1 would skip
        # against stale state from gen N.
        from wan.distributed.teacache import reset_teacache
        reset_teacache(pipeline.model)

    t0 = time.perf_counter()
    result = pipeline.generate(
        input_prompt=prompt,
        size=size,
        frame_num=frame_num,
        seed=seed,
        sampling_steps=sampling_steps,
        offload_model=True,
        reuse=True,
    )
    torch.cuda.synchronize(local_rank)
    wall = time.perf_counter() - t0

    after_gen = _mem_snapshot("after_generate")

    if rank == 0 and result is not None and save_path:
        from wan.utils.utils import save_video
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(
            tensor=result[None],
            save_file=save_path,
            fps=24,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    # Drop tensor before peak-after-save snapshot so the post-save peak
    # only counts retained state (catches across-request drift).
    result = None  # noqa: F841
    gc.collect()
    torch.cuda.empty_cache()
    after_save = _mem_snapshot("after_save_inline")

    out = {
        "wall_s": wall,
        "mean_step_s": wall / sampling_steps,
        "enter": enter,
        "after_generate": after_gen,
        "after_save": after_save,
    }
    if teacache_enabled:
        from wan.distributed.teacache import teacache_stats
        out["teacache"] = teacache_stats(pipeline.model)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", default=os.environ.get("WAN_DEMO_CKPT_DIR"),
                   help="Wan2.2-TI2V-5B checkpoint dir. Default: $WAN_DEMO_CKPT_DIR")
    p.add_argument("--out-dir", default=os.environ.get(
        "WAN_DEMO_BENCH_OUT_DIR", "./bench_artifacts/path-b"),
                   help="Where to write per-gen mp4s. JSON always lands in ./bench_artifacts/.")
    p.add_argument("--prompt", default=_CANONICAL_PROMPT,
                   help=f"Prompt. Default: {_CANONICAL_PROMPT!r}")
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--num-gens", type=int, default=3,
                   help="Total gens. Gen 1 is cold (pays pipeline build); "
                        "warm steady-state is mean(gen 2..N).")
    p.add_argument("--quant",
                   choices=["bf16", "fp8", "fp8_fast"],
                   default=None,
                   help="Sets WAN_DEMO_QUANT for the pipeline build. "
                        "If omitted, uses the env var (or 'bf16' default).")
    p.add_argument("--attn", choices=["flash", "sage"], default=None,
                   help="Sets WAN_DEMO_ATTN. Phase 2+; ignored otherwise.")
    p.add_argument("--compile", dest="compile_flag",
                   choices=["0", "1"], default=None,
                   help="Sets WAN_DEMO_COMPILE. Phase 3+; ignored otherwise.")
    p.add_argument("--teacache-thresh", type=float, default=None,
                   help="Enable TeaCache step-caching at this rescaled-rel-L1 "
                        "accumulator threshold. Recommended 0.15-0.25 on 5B. "
                        "Unset = caching off. Disable model-level compile "
                        "when this is on (data-dependent Python branch).")
    p.add_argument("--teacache-use-ret-steps", action="store_true",
                   help="Use the e0/_RET polynomial variant with longer "
                        "warmup (5 steps). Slightly higher quality at the "
                        "same threshold; slightly less speedup.")
    p.add_argument("--tag", default=None,
                   help="Optional run tag appended to the JSON filename.")
    p.add_argument("--init-on-cpu", action="store_true",
                   help="Keep DiT on CPU at construction (move to GPU only "
                        "during diffusion). Required for bf16 on 24 GB GPUs "
                        "where T5+DiT both bf16 don't co-fit; production "
                        "fp8 path uses False since fp8 DiT is ~9 GB.")
    args = p.parse_args(argv)

    if not args.ckpt:
        print("ERROR: --ckpt or WAN_DEMO_CKPT_DIR required", file=sys.stderr)
        return 2

    if args.quant is not None:
        os.environ["WAN_DEMO_QUANT"] = args.quant
    if args.attn is not None:
        os.environ["WAN_DEMO_ATTN"] = args.attn
    if args.compile_flag is not None:
        os.environ["WAN_DEMO_COMPILE"] = args.compile_flag

    # Match production server allocator config — across-request peak
    # numbers are not comparable without it.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _maybe_init_dist()

    quant = os.environ.get("WAN_DEMO_QUANT", "bf16")
    attn = os.environ.get("WAN_DEMO_ATTN", "flash")
    compile_on = os.environ.get("WAN_DEMO_COMPILE", "0")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_root = Path("./bench_artifacts").resolve()
    bench_root.mkdir(parents=True, exist_ok=True)

    branch = _git_branch().replace("/", "-")
    sha = _git_sha()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_tag = f"q{quant}_a{attn}_c{compile_on}_f{args.frames}_s{args.steps}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    json_name = f"{branch}_{ts}_{config_tag}{tag_suffix}.json"
    json_path = bench_root / json_name

    pipeline, build_s = _build_pipeline(args.ckpt, init_on_cpu=args.init_on_cpu)

    teacache_enabled = args.teacache_thresh is not None
    if teacache_enabled:
        if compile_on in ("1", "true", "yes", "on"):
            logging.warning(
                "TeaCache + torch.compile: the data-dependent Python "
                "branch in the cached forward forces graph breaks; "
                "expect compile speedup to be largely lost. Spike "
                "should run with --compile 0."
            )
        from wan.distributed.teacache import enable_teacache
        enable_teacache(
            pipeline.model,
            thresh=args.teacache_thresh,
            num_steps=2 * args.steps,  # CFG: cond + uncond per step
            use_ret_steps=args.teacache_use_ret_steps,
        )

    gens: list[dict[str, Any]] = []
    for i in range(1, args.num_gens + 1):
        save_path = str(out_dir / f"gen_{i}.mp4")
        logging.info("=== gen %d/%d ===", i, args.num_gens)
        stats = _run_one_gen(
            pipeline,
            prompt=args.prompt,
            size=(args.width, args.height),
            frame_num=args.frames,
            seed=args.seed,
            sampling_steps=args.steps,
            save_path=save_path,
            teacache_enabled=teacache_enabled,
        )
        stats["index"] = i
        stats["save_path"] = save_path
        gens.append(stats)
        logging.info(
            "gen %d wall=%.2fs mean_step=%.3fs peak_alloc=%dMB",
            i, stats["wall_s"], stats["mean_step_s"],
            stats["after_generate"]["peak_alloc_mb"],
        )

    warm = [g for g in gens if g["index"] >= 2]
    summary = {
        "host": socket.gethostname(),
        "timestamp": ts,
        "git_branch": branch,
        "git_sha": sha,
        "env": {
            "WAN_DEMO_QUANT": quant,
            "WAN_DEMO_ATTN": attn,
            "WAN_DEMO_COMPILE": compile_on,
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "teacache": {
            "enabled": teacache_enabled,
            "thresh": args.teacache_thresh,
            "use_ret_steps": args.teacache_use_ret_steps,
        },
        "config": {
            "prompt": args.prompt,
            "size": [args.width, args.height],
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "num_gens": args.num_gens,
            "ckpt": args.ckpt,
        },
        "pipeline_build_s": build_s,
        "gens": gens,
        "warm_mean_wall_s":
            sum(g["wall_s"] for g in warm) / len(warm) if warm else None,
        "warm_mean_step_s":
            sum(g["mean_step_s"] for g in warm) / len(warm) if warm else None,
        "cold_extra_s": (gens[0]["wall_s"] - warm[0]["wall_s"]) if warm else None,
        "out_dir": str(out_dir),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    logging.info("wrote %s", json_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
