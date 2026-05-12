#!/usr/bin/env python3
"""xDiT (xfuser) runner for Wan2.2 T2V-A14B.

Companion to xfuser_wan_pipeline.py — registers the WanXFuserPipeline wrapper,
parses xfuser CLI args, initializes the parallel runtime, runs inference, and
emits a stub timing CSV that variance_run.py / analyze_sweep.py can consume.

Launch via torchrun. Example:
    torchrun --nproc_per_node=2 scripts/profiling/run_xdit.py \\
      --model /workspace/Wan2.2-T2V-A14B-Diffusers \\
      --prompt 'two cats' --height 720 --width 1280 \\
      --num_frames 81 --num_inference_steps 40 --seed 42 \\
      --ulysses_degree 2 --ring_degree 1 \\
      --profile_dir /workspace/profile_outputs/xdit_run \\
      --output mp4
"""

import argparse
import os
import sys
import time
import uuid

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video

# Ensure our wrapper modules are importable; they register via import side-effect.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xfuser_wan_transformer  # noqa: F401 — registers xFuserWanTransformer3DWrapper
import xfuser_wan_pipeline  # noqa: F401 — registers WanXFuserPipeline

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    initialize_runtime_state,
    is_dp_last_group,
)
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister
from diffusers import WanPipeline


def parse_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser(description="xDiT Wan2.2 runner")
    xFuserArgs.add_cli_args(parser)
    parser.add_argument("--profile_dir", type=str, required=True,
                        help="Where to write timing CSV stub.")
    parser.add_argument("--guidance_scale_2", type=float, default=None,
                        help="Wan2.2 high-noise guidance scale (low_noise = "
                             "--guidance_scale).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="MP4 output path; default <profile_dir>/output.mp4")
    return parser.parse_args()


def emit_timing_csv(profile_dir: str, run_id: str, t_init_start: float,
                    t_init_end: float, t_diff_start: float, t_diff_end: float,
                    t_pipeline_end: float, peak_mb: float, attn_backend: str):
    """Write a stub timing_rank0.csv compatible with variance_run.read_run()."""
    os.makedirs(profile_dir, exist_ok=True)
    csv = os.path.join(profile_dir, "timing_rank0.csv")
    init_ms = (t_init_end - t_init_start) * 1000.0
    gen_ms = (t_pipeline_end - t_init_end) * 1000.0
    diff_ms = (t_diff_end - t_diff_start) * 1000.0
    with open(csv, "w") as f:
        f.write("run_id,rank,name,step,wall_ms,gpu_ms,timestamp\n")
        f.write(f"{run_id},0,memory/pipeline_init,-1,0,0,{t_init_start:.6f}\n")
        if attn_backend:
            f.write(f"{run_id},0,env/attn_backend/{attn_backend},-1,0,0,{t_init_start:.6f}\n")
        f.write(f"{run_id},0,pipeline_init,-1,{init_ms:.3f},-1,{t_init_end:.6f}\n")
        f.write(f"{run_id},0,diffusion_loop,-1,{diff_ms:.3f},-1,{t_diff_end:.6f}\n")
        f.write(f"{run_id},0,pipeline_generate,-1,{gen_ms:.3f},-1,{t_pipeline_end:.6f}\n")
        f.write(f"{run_id},0,memory/pipeline_end,-1,{peak_mb:.3f},{peak_mb:.3f},{t_pipeline_end:.6f}\n")


def detect_attn_backend() -> str:
    # Catch ImportError (parent of ModuleNotFoundError) to handle the case
    # where flash-attn is installed but its C++ extension fails to load due
    # to a torch ABI mismatch — at runtime xfuser falls back to SDPA, and so
    # should this probe.
    try:
        import flash_attn_interface  # noqa: F401
        return "flash_attn_3"
    except (ImportError, OSError):
        pass
    try:
        import flash_attn  # noqa: F401
        return "flash_attn_2"
    except (ImportError, OSError):
        pass
    return "sdpa"


def main():
    args = parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    parallel_config = engine_config.parallel_config

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    t_init_start = time.time()

    # xfuser handles dist.init + parallel groups via the engine_config.
    # Pipeline wrapper takes care of the rest internally.
    pipeline = xFuserPipelineWrapperRegister.get_class(WanPipeline).from_pretrained(
        args.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device)

    # Initialize xfuser runtime state (sets per-rank seq-parallel/CFG groups).
    initialize_runtime_state(pipeline, engine_config)

    rank = get_world_group().rank if dist.is_initialized() else 0

    t_init_end = time.time()

    # Warm one-line: ensure FA backend has been touched
    attn_backend = detect_attn_backend()

    # Generate
    seed = args.seed if args.seed is not None else 42
    generator = torch.Generator(device=device).manual_seed(seed)

    t_diff_start = time.time()
    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        generator=generator,
        output_type="np",
        return_dict=True,
    )
    t_diff_end = time.time()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Only the dp-last-group rank actually decoded the VAE; on CFG-parallel-2
    # that is world-rank 1 (cfg_rank == cfg_world - 1), not world-rank 0.
    # Previously this gate was `is_dp_last_group() and rank == 0`, which is
    # unsatisfiable when cfg_world > 1 — so the mp4 never got written.
    if is_dp_last_group() and output.frames is not None and output.frames[0] is not None:
        out_path = args.output_path or os.path.join(args.profile_dir, "output.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        export_to_video(output.frames[0], out_path, fps=16)
        print(f"[run_xdit] saved {out_path}", flush=True)

    t_pipeline_end = time.time()

    if rank == 0:
        run_id = f"xdit_{uuid.uuid4().hex[:12]}"
        emit_timing_csv(
            args.profile_dir, run_id,
            t_init_start, t_init_end, t_diff_start, t_diff_end, t_pipeline_end,
            peak_mb, attn_backend,
        )
        print(
            f"[run_xdit] init={t_init_end - t_init_start:.1f}s "
            f"diffusion={t_diff_end - t_diff_start:.1f}s "
            f"total={t_pipeline_end - t_init_start:.1f}s "
            f"peak={peak_mb:.0f}MB attn={attn_backend}", flush=True,
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
