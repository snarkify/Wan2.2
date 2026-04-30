#!/usr/bin/env bash
# Path B visual A/B scorer.
#
# Usage:
#   scripts/score_path_b_ab.sh <candidate_mp4> <reference_mp4> [tag]
#
# Runs ffmpeg PSNR + SSIM filters on the two clips, parses the per-frame
# stats files for min PSNR / min SSIM, and prints a one-line summary
# plus a structured TSV row to stdout. Per-frame stats files are kept
# next to the candidate so they can be inspected if a frame fails.
#
# Pass criteria (per docs/path-b-acceptance.md, Phase 1 thresholds):
#   - mean PSNR >= 32 dB
#   - min  PSNR >= 28 dB
#   - mean SSIM >= 0.95
#   - min  SSIM >= 0.92
#
# Exit code: 0 on PASS, 1 on FAIL, 2 on usage / setup error.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <candidate_mp4> <reference_mp4> [tag]" >&2
    exit 2
fi

CANDIDATE="$1"
REFERENCE="$2"
TAG="${3:-$(basename "$CANDIDATE" .mp4)}"

if [ ! -f "$CANDIDATE" ] || [ ! -f "$REFERENCE" ]; then
    echo "ERROR: missing input file. candidate=$CANDIDATE reference=$REFERENCE" >&2
    exit 2
fi

DIR="$(dirname "$CANDIDATE")"
PSNR_LOG="$DIR/psnr_${TAG}.log"
SSIM_LOG="$DIR/ssim_${TAG}.log"

# PSNR. ffmpeg writes per-frame numbers to PSNR_LOG and the aggregate
# mean appears on stderr. Capture both.
PSNR_STDERR=$(
    ffmpeg -hide_banner -loglevel info \
        -i "$CANDIDATE" -i "$REFERENCE" \
        -lavfi "psnr=stats_file=${PSNR_LOG}" -f null - 2>&1 \
        || true
)
SSIM_STDERR=$(
    ffmpeg -hide_banner -loglevel info \
        -i "$CANDIDATE" -i "$REFERENCE" \
        -lavfi "ssim=stats_file=${SSIM_LOG}" -f null - 2>&1 \
        || true
)

# ffmpeg PSNR aggregate line:  "[Parsed_psnr_0 @ 0x...] PSNR y:42.06 ... average:42.06 min:38.20 max:..."
MEAN_PSNR=$(echo "$PSNR_STDERR" | awk '/PSNR.*average/ { for (i=1;i<=NF;i++) if ($i ~ /^average:/) { sub("average:","",$i); print $i; exit } }')
# Per-frame log has rows like: "n:1 mse_avg:0.20 ... psnr_avg:55.12 ..."
MIN_PSNR=$(awk '/psnr_avg:/ { for (i=1;i<=NF;i++) if ($i ~ /^psnr_avg:/) { sub("psnr_avg:","",$i); v=$i+0; if (NR==1 || v<m) m=v } } END { if (m == "") print "nan"; else printf "%.4f", m }' "$PSNR_LOG")

# ffmpeg SSIM aggregate: "[Parsed_ssim_0 @ ...] SSIM Y:0.987 ... All:0.987 (18.93)"
MEAN_SSIM=$(echo "$SSIM_STDERR" | awk '/SSIM.*All:/ { for (i=1;i<=NF;i++) if ($i ~ /^All:/) { sub("All:","",$i); print $i; exit } }')
# Per-frame SSIM log: "n:1 Y:0.987 U:... V:... All:0.987 (18.93)"
MIN_SSIM=$(awk '/All:/ { for (i=1;i<=NF;i++) if ($i ~ /^All:/) { sub("All:","",$i); v=$i+0; if (NR==1 || v<m) m=v } } END { if (m == "") print "nan"; else printf "%.4f", m }' "$SSIM_LOG")

# Decide pass/fail using awk for floating-point comparisons (bash can't).
PASS=$(awk -v mp="$MEAN_PSNR" -v np="$MIN_PSNR" -v ms="$MEAN_SSIM" -v ns="$MIN_SSIM" \
    'BEGIN {
        ok = 1;
        if (mp+0 < 32.0) ok = 0;
        if (np+0 < 28.0) ok = 0;
        if (ms+0 < 0.95) ok = 0;
        if (ns+0 < 0.92) ok = 0;
        print ok ? "PASS" : "FAIL";
    }')

# Pretty header on first invocation per dir.
SUMMARY_TSV="$DIR/scores.tsv"
if [ ! -f "$SUMMARY_TSV" ]; then
    printf "tag\tmean_psnr\tmin_psnr\tmean_ssim\tmin_ssim\tverdict\tcandidate\treference\n" > "$SUMMARY_TSV"
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TAG" "$MEAN_PSNR" "$MIN_PSNR" "$MEAN_SSIM" "$MIN_SSIM" "$PASS" "$CANDIDATE" "$REFERENCE" \
    >> "$SUMMARY_TSV"

# Stdout: one human-readable line.
printf "[%s] mean_psnr=%s min_psnr=%s mean_ssim=%s min_ssim=%s -> %s\n" \
    "$TAG" "$MEAN_PSNR" "$MIN_PSNR" "$MEAN_SSIM" "$MIN_SSIM" "$PASS"

[ "$PASS" = "PASS" ] || exit 1
