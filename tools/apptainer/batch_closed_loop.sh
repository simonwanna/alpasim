#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

# Submit as a Slurm array with sbatch, supplying the account, one GPU per task,
# time, and log path on its command line. The run manifest is tab-separated with
# this header row, then one row per run; array task N runs data row N:
#
#   name  scene  command  steps  approach  seed  prompt
#
# name: short label used in the output directory ([A-Za-z0-9._-]).
# scene: absolute USDZ path. command: straight, left, or right.
# steps: total control steps. approach: recorded steps before policy handover.
# seed: fixed renderer seed, or '-' to let the renderer choose.
# prompt: positive video-model prompt, or '-' to keep the seed session prompt.
#
# Outputs go to OUTPUT_ROOT/<manifest name>-<array job>/<task>-<name>/.

set -euo pipefail
umask 007

if [[ $# -ne 9 ]]; then
    echo "Usage: $0 REPO PROJECT RUNS_TSV IMAGE CACHE OUTPUT_ROOT CHECKPOINT TOKENIZER SEED_SESSION" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

repo_root=$1
project=$2
manifest=$3
image=$4
cache=$5
output_root=$6
checkpoint=$7
tokenizer=$8
seed_session=$9
array_job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
task_id=${SLURM_ARRAY_TASK_ID:-0}

[[ -f $repo_root/tools/apptainer/closed_loop.py ]] || {
    echo "Missing AlpaSim checkout at: $repo_root" >&2
    exit 2
}
[[ $task_id =~ ^[0-9]+$ ]] || { echo "Invalid array task ID: $task_id" >&2; exit 2; }
[[ -f $manifest ]] || { echo "Missing run manifest: $manifest" >&2; exit 2; }
mapfile -t lines < "$manifest"
header=$'name\tscene\tcommand\tsteps\tapproach\tseed\tprompt'
[[ ${lines[0]:-} = "$header" ]] || {
    echo "Manifest header must be: ${header//$'\t'/<TAB>}" >&2
    exit 2
}
(( task_id + 1 < ${#lines[@]} )) || {
    echo "Array task $task_id is outside the $(( ${#lines[@]} - 1 )) manifest rows" >&2
    exit 2
}
row=${lines[$((task_id + 1))]}
IFS=$'\t' read -r name scene command steps approach seed prompt <<< "$row"
where="Manifest data row $((task_id + 1))"
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "$where needs a name of [A-Za-z0-9._-]" >&2; exit 2; }
[[ -n $scene && $scene = /* && -f $scene ]] || {
    echo "$where must name an existing absolute scene file" >&2
    exit 2
}
[[ $command = straight || $command = left || $command = right ]] || {
    echo "$where needs straight, left, or right" >&2
    exit 2
}
[[ $steps =~ ^[0-9]+$ && $approach =~ ^[0-9]+$ ]] || {
    echo "$where needs integer steps and approach" >&2
    exit 2
}
[[ $seed = - || $seed =~ ^[0-9]+$ ]] || { echo "$where needs an integer seed or -" >&2; exit 2; }
[[ -n $prompt ]] || { echo "$where needs a prompt or -" >&2; exit 2; }

campaign="$output_root/$(basename "$manifest" .tsv)-$array_job"
out="$campaign/$(printf '%02d' "$task_id")-$name"
mkdir -p "$campaign"
mkdir "$out"
chmod 2770 "$campaign" "$out"
printf '%s\n%s\n' "$header" "$row" > "$out/manifest-row.tsv"
job_cache="$cache/batch/$array_job-$task_id"
echo "Job: $SLURM_JOB_ID; array: $array_job; task: $task_id; name: $name; output: $out/run"
echo "Scene: $scene; command: $command; steps: $steps; approach: $approach; seed: $seed; prompt: $prompt"

if [[ $prompt != - ]]; then
    prompt_script=$(cd "$repo_root" && pwd)/tools/apptainer/write_prompt_session.py
    case "$prompt_script" in
        "$project"/*) ;;
        *) echo "AlpaSim checkout must be inside project storage" >&2; exit 2 ;;
    esac
    case "$seed_session" in
        "$project"/*) ;;
        *) echo "Seed session must be inside project storage" >&2; exit 2 ;;
    esac
    python3 "$repo_root/tools/apptainer/apptainer_exec.py" \
        --project "$project" \
        --image "$image" \
        --profile core \
        --cache-dir "$job_cache/prompt-prep" \
        --output-dir "$out/prompt-prep" \
        --timeout 120 \
        -- -I "/workspace/${prompt_script#"$project"/}" \
        --source "/workspace/${seed_session#"$project"/}" \
        --output "/workspace/${out#"$project"/}/prompt-session.pb" \
        --positive "$prompt"
    seed_session="$out/prompt-session.pb"
fi

seed_args=()
[[ $seed = - ]] || seed_args=(--renderer-seed "$seed")
# About 3.3 s per step was measured; allow startup, 8 s per step, and GIF export.
export_timeout=$((600 + 3 * steps))
srun --ntasks=1 --gpus=1 python3 "$repo_root/tools/apptainer/closed_loop.py" \
    --project "$project" \
    --image "$image" \
    --cache-dir "$job_cache" \
    --output-dir "$out/run" \
    --timeout $((600 + 8 * steps + export_timeout)) \
    --export-timeout "$export_timeout" \
    --steps "$steps" \
    --approach-steps "$approach" \
    --command "$command" \
    "${seed_args[@]}" \
    --scene "$scene" \
    --checkpoint "$checkpoint" \
    --tokenizer "$tokenizer" \
    --seed-session "$seed_session" \
    --run
