"""Vanilla diffusers WanPipeline runner — same prompt/seed as Y1.

Goal: settle the SGLang 17 dB delta question. If this run produces a video
similar (PSNR >= 50 dB) to the Y1 reference (generate.py output), then
diffusers and in-tree produce equivalent output, and SGLangs 17 dB delta
must be something SGLang specific. If this run also differs ~17 dB from Y1,
then 17 dB is the diffusers-vs-in-tree RNG/scheduler-path delta, and
SGLang is just inheriting that.
"""
import argparse, time, os, torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--guidance_scale_2", type=float, default=3.0)
    ap.add_argument("--flow_shift", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda", 0)

    print(f"[diffusers] loading {args.model}")
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    # Match Wan2.2 scheduler defaults
    if hasattr(pipe.scheduler.config, "flow_shift"):
        pipe.scheduler = pipe.scheduler.__class__.from_config(
            pipe.scheduler.config, flow_shift=args.flow_shift
        )
    print(f"[diffusers] loaded in {time.time()-t0:.1f}s")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[diffusers] generating: seed={args.seed} steps={args.steps} "
          f"size={args.width}x{args.height} frames={args.num_frames}")

    t1 = time.time()
    out = pipe(
        prompt=args.prompt,
        negative_prompt=NEG_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        generator=gen,
        output_type="np",
    )
    t2 = time.time()
    print(f"[diffusers] generated in {t2-t1:.1f}s")

    export_to_video(out.frames[0], args.out, fps=16)
    print(f"[diffusers] saved {args.out}")
    print(f"[diffusers] peak mem: {torch.cuda.max_memory_allocated(device)/1024**3:.1f} GB")


if __name__ == "__main__":
    main()
