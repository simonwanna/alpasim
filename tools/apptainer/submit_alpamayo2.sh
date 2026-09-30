#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

# Submit one two-GPU job using existing environments and scene/model assets.
# Usage: bash submit_alpamayo2.sh OUTPUT_ROOT [closed_loop.py arguments]
# Set SBATCH_ACCOUNT if required; RUN_TIMEZONE controls date/time folder names.
# The default 24-step run has a 25-minute launcher limit and 30-minute allocation.
set -euo pipefail
umask 007

if [[ ${1:-} != --worker ]]; then
    if [[ $# -lt 2 || $1 = --help ]]; then
        echo "Usage: bash $0 OUTPUT_ROOT [closed_loop.py arguments]"
        echo "Supply --project, --image, --cache-dir, --scene, --seed-session and --checkpoint."
        exit 2
    fi
    output_root=$1
    shift
    [[ $output_root = /* ]] || { echo "OUTPUT_ROOT must be absolute" >&2; exit 2; }
    for arg in "$@"; do
        case ${arg%%=*} in
            --output-dir|--policy|--policy-gpu|--renderer-gpu|--run|--help|-h)
                echo "This single-job wrapper owns $arg" >&2
                exit 2 ;;
        esac
    done
    repo_root=$(cd "$(dirname "$0")/../.." && pwd)
    stamp=$(TZ="${RUN_TIMEZONE:-UTC}" date +%F/%H-%M-%S%z)
    out="$output_root/$stamp"
    mkdir -p "$(dirname "$out")"
    mkdir -m 2770 "$out"
    printf '%q ' bash "$repo_root/tools/apptainer/submit_alpamayo2.sh" "$output_root" "$@" > "$out/submission.txt"
    printf '\n' >> "$out/submission.txt"
    git -C "$repo_root" rev-parse HEAD > "$out/code-revision.txt"
    echo "Outputs: $out"
    # Absolute log paths exist before submission, so Slurm creates no repo logs.
    if job=$(sbatch --parsable --job-name=alpamayo2-loop --partition=gpu \
        --ntasks=1 --gpus=2 --gpus-per-task=2 --cpus-per-task=8 --time=00:30:00 \
        --chdir="$repo_root" --output="$out/slurm.log" --error="$out/slurm.log" \
        "$repo_root/tools/apptainer/submit_alpamayo2.sh" --worker "$repo_root" "$out" "$@"); then
        printf '%s\n' "$job" > "$out/job-id.txt"
        echo "Submitted job: $job"
        echo "View: tail -f '$out/slurm.log'"
        echo "Stop: scancel ${job%%;*}"
    else
        status=$?
        printf 'submission_exit_status=%s\n' "$status" > "$out/job-summary.txt"
        exit "$status"
    fi
    exit 0
fi

: "${SLURM_JOB_ID:?The worker must run through sbatch}"
repo_root=$2
out=$3
shift 3
[[ -d $out && ! -e $out/run && ! -e $out/job-summary.txt ]] || {
    echo "Worker needs a new run directory" >&2; exit 2;
}
started=$(date -u +%FT%TZ)
SECONDS=0
step_pid=
finish() {
    status=$?
    trap - EXIT
    {
        printf 'job_id=%s\nstarted_utc=%s\nfinished_utc=%s\n' "$SLURM_JOB_ID" "$started" "$(date -u +%FT%TZ)"
        printf 'elapsed_seconds=%s\nexit_status=%s\n' "$SECONDS" "$status"
        echo 'timing_scope=preflight, model loading, simulation and export; excludes queue time'
    } > "$out/job-summary.txt"
    cat "$out/job-summary.txt"
    exit "$status"
}
stop_job() {
    if [[ -n $step_pid ]]; then
        kill -TERM "$step_pid" 2>/dev/null || true
        wait "$step_pid" || true
    fi
    exit "$1"
}
trap finish EXIT
trap 'stop_job 143' TERM
trap 'stop_job 130' INT
srun --ntasks=1 --gpus=2 python3 -u "$repo_root/tools/apptainer/closed_loop.py" \
    --steps 24 --approach-steps 6 --renderer-seed 42 \
    --timeout 1500 --export-timeout 600 \
    "$@" --output-dir "$out/run" \
    --policy alpamayo2 --policy-gpu 0 --renderer-gpu 1 --run &
step_pid=$!
wait "$step_pid"
