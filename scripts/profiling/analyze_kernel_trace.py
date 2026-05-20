"""Track-A kernel-level analysis of a torch.profiler Chrome trace.

Reads a trace produced by run_kernel_profile.sh, buckets CUDA kernels into
semantic categories (attention math, weight matmul, layer-norm, etc.), and
prints a per-category breakdown so we can compare with the analytical
decomposition from the report.

Usage:
    python scripts/profiling/analyze_kernel_trace.py <trace.json[.gz]>

Categories (matched against kernel name with regex):
    attention_math     scaled_dot_product, flash_attn*, attention kernels
    matmul             aten::mm, addmm, matmul, linear, bmm, baddbmm
    layernorm          layer_norm, rms_norm
    activation         silu, gelu, relu (etc.)
    elementwise        add, mul, div, sub, sigmoid (etc.)
    memory             empty, copy_, to, contiguous, fill_
    reshape            view, transpose, permute, reshape, expand
    rope_freq          rope / freq / position-embedding
    other              catch-all
"""
import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


CATEGORIES = [
    # name              priority  patterns (case-insensitive substring or regex)
    ("attention_math",  [
        r"flash[_-]?attn",
        r"flash_fprop",                      # cuDNN flash-style attention forward
        r"scaled_dot_product",
        r"native_sdpa",                      # cuDNN sm90 SDPA kernels
        r"_efficient_attention",
        r"_attention_forward",
        r"cudnn_generated.*sdpa",
        r"FlashFwd",
        r"fmha",
        r"softmax",                          # part of attention path
    ]),
    ("matmul",          [
        r"\baten::mm\b",
        r"\baten::addmm\b",
        r"\baten::bmm\b",
        r"\baten::baddbmm\b",
        r"\baten::matmul\b",
        r"\baten::linear\b",
        r"\bnvjet",                          # NVIDIA's modern GEMM (sm90+)
        r"xmma_gemm",                        # cuBLAS xmma
        r"gemm",
        r"cublas",
        r"cutlass",
        r"\bmma_kernel\b",
        r"WGMMA",                            # Hopper warp-group MMA (raw)
    ]),
    ("layernorm",       [
        r"layer_norm",
        r"rms_?norm",
        r"native_layer_norm",
    ]),
    ("rope_freq",       [
        r"rope", r"rotary", r"freq", r"sin_cos", r"position_embedding",
    ]),
    ("activation",      [
        r"\bsilu\b", r"\bswish\b", r"\bgelu\b", r"\brelu\b", r"\bsigmoid\b",
        r"\btanh\b",
    ]),
    ("elementwise",     [
        r"\baten::add\b", r"\baten::add_\b",
        r"\baten::mul\b", r"\baten::mul_\b",
        r"\baten::div\b", r"\baten::sub\b",
        r"\baten::neg\b",
        # PyTorch elementwise GPU kernel templates (the actual CUDA kernel
        # names that show up under cat="kernel"):
        r"elementwise_kernel",
        r"vectorized_elementwise",
        r"unrolled_elementwise",
        r"cudaLaunchKernel",
    ]),
    ("reshape_view",    [
        r"\baten::view\b", r"\baten::transpose\b", r"\baten::permute\b",
        r"\baten::reshape\b", r"\baten::expand\b", r"\baten::unsqueeze\b",
        r"\baten::squeeze\b", r"\baten::contiguous\b",
    ]),
    ("memory",          [
        r"\baten::empty\b", r"\baten::empty_like\b",
        r"\baten::zeros\b", r"\baten::ones\b",
        r"\baten::copy_\b", r"\baten::to\b",
        r"\baten::fill_\b", r"\baten::clone\b",
        r"Memcpy", r"Memset",
    ]),
    ("collectives",     [
        r"nccl", r"all_gather", r"all_reduce", r"all_to_all", r"broadcast",
    ]),
]


def categorize(name: str) -> str:
    """Return the first category whose patterns match the kernel name."""
    lname = name
    for cat, pats in CATEGORIES:
        for p in pats:
            if re.search(p, lname, re.IGNORECASE):
                return cat
    return "other"


def load_trace(path: Path):
    """Load a Chrome trace JSON (optionally .gz). Return the events list."""
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="Chrome trace JSON (from torch.profiler)")
    ap.add_argument("--top", type=int, default=20,
                    help="show the N heaviest individual kernel names")
    ap.add_argument("--gpu-only", action="store_true",
                    help="aggregate only kernels with cat='kernel' (true GPU work), "
                         "excluding CPU-side aten::* calls")
    args = ap.parse_args()

    print(f"Loading {args.trace} ...", file=sys.stderr)
    events = load_trace(args.trace)
    print(f"  {len(events):,} events", file=sys.stderr)

    # Bucket durations by category.
    # CUDA kernel events typically have cat="kernel" and dur in microseconds.
    # CPU-side aten::* events have cat="cpu_op".
    cat_us = defaultdict(int)
    kernel_us = defaultdict(int)   # name -> us, for top-N table
    total_us = 0

    for e in events:
        ph = e.get("ph")
        if ph != "X":   # only complete (X) events have durations
            continue
        cat = e.get("cat", "")
        if args.gpu_only and cat != "kernel":
            continue
        # Only count GPU kernels or CPU ops with measurable duration
        if cat not in ("kernel", "cpu_op", "gpu_user_annotation"):
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        name = e.get("name", "?")
        bucket = categorize(name)
        cat_us[bucket] += dur
        kernel_us[name] += dur
        total_us += dur

    total_ms = total_us / 1000.0
    print()
    print(f"=== Per-category breakdown ({'GPU kernels only' if args.gpu_only else 'all events'}) ===")
    print(f"  total measured: {total_ms:>10.1f} ms")
    print()
    print(f"{'category':<20} {'ms':>12} {'%':>7}    bar (40-wide)")
    print("-" * 80)
    # Order by descending time
    ordered = sorted(cat_us.items(), key=lambda kv: -kv[1])
    for cat, us in ordered:
        ms = us / 1000.0
        pct = 100.0 * us / total_us if total_us else 0
        bar = "█" * int(pct * 40 / 100)
        print(f"{cat:<20} {ms:>12,.2f} {pct:>6.2f}%   {bar}")

    print()
    print(f"=== Top {args.top} individual kernels by total time ===")
    print(f"{'kernel':<70} {'ms':>10} {'%':>7}")
    print("-" * 92)
    top = sorted(kernel_us.items(), key=lambda kv: -kv[1])[:args.top]
    for name, us in top:
        ms = us / 1000.0
        pct = 100.0 * us / total_us if total_us else 0
        # Truncate very long kernel names
        disp = name if len(name) <= 68 else name[:65] + "..."
        print(f"{disp:<70} {ms:>10,.2f} {pct:>6.2f}%")

    # Comparison against the analytical prediction (from the HTML report § Track A)
    print()
    print("=== Comparison vs. analytical prediction (1280×720, 81 frames, in-tree DiT) ===")
    pred = {
        "attention_math": 65.5,
        "matmul":         12.3,
        "layernorm":       2.0,   # subset of "LN+residual+activation" 8.4%
        "activation":      0.5,
        "elementwise":     5.9,   # rest of "LN+residual+activation" + modulation
        "memory":          5.0,
        "reshape_view":    3.0,
        "rope_freq":       2.0,
        "collectives":     0.0,   # 1-GPU run, no collectives
        "other":           3.8,
    }
    print(f"{'category':<20} {'predicted %':>14} {'measured %':>14} {'delta':>10}")
    print("-" * 64)
    for cat, _ in CATEGORIES + [("other", None)]:
        p = pred.get(cat, 0)
        m = 100.0 * cat_us.get(cat, 0) / total_us if total_us else 0
        d = m - p
        sign = "+" if d >= 0 else ""
        print(f"{cat:<20} {p:>12.1f} % {m:>12.1f} % {sign}{d:>8.1f}")

    print()
    # Show the top kernels still landing in "other" so we can refine the
    # category regex on iteration.
    print()
    other_kernels = [
        (name, us) for name, us in kernel_us.items() if categorize(name) == "other"
    ]
    other_kernels.sort(key=lambda kv: -kv[1])
    if other_kernels:
        print(f"=== Top kernels still in 'other' (refine the regex if any are large) ===")
        for name, us in other_kernels[:10]:
            ms = us / 1000.0
            pct = 100.0 * us / total_us if total_us else 0
            disp = name if len(name) <= 68 else name[:65] + "..."
            print(f"  {disp:<70} {ms:>8.1f} ms  {pct:>5.2f}%")

    print()
    print("Interpretation:")
    print("  - attention_math ≈ predicted (65 %) confirms the bottleneck is attention.")
    print("  - matmul above predicted suggests FFN dominates the weight-op budget;")
    print("    if matmul %% is high, FP8 quantization is the high-leverage next step.")
    print("  - elementwise + memory high → kernel-fusion or compile would help.")
    print("  - any category materially above predicted is a candidate for targeted")
    print("    optimization.")


if __name__ == "__main__":
    main()
