#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${ALPASIM_PROJECT:?Set ALPASIM_PROJECT to your project storage directory}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

heading() {
    printf '\n[%s]\n' "$1"
}

show_version() {
    local command_name="$1"
    shift
    if command -v "$command_name" >/dev/null 2>&1; then
        printf '%-12s %s\n' "$command_name" "$($command_name "$@" 2>&1 | head -n 1)"
    else
        printf '%-12s NOT FOUND\n' "$command_name"
    fi
}

heading "location"
printf 'repo=%s\n' "$REPO_ROOT"
printf 'project=%s\n' "$PROJECT_ROOT"
printf 'project_exists=%s\n' "$([[ -d "$PROJECT_ROOT" ]] && printf yes || printf no)"
printf 'project_writable=%s\n' "$([[ -w "$PROJECT_ROOT" ]] && printf yes || printf no)"

heading "allocation"
printf 'host=%s\n' "$(hostname)"
printf 'arch=%s\n' "$(uname -m)"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-NOT_SET}"
printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-NOT_SET}"
printf 'cuda_visible_devices_set=%s\n' "$([[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && printf yes || printf no)"

heading "software"
show_version apptainer --version
show_version git --version
show_version python3 --version
show_version uv --version

heading "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    printf 'nvidia-smi NOT FOUND\n'
fi

heading "storage"
df -h "$PROJECT_ROOT" 2>&1 | tail -n 1
probe_file="$PROJECT_ROOT/.alpasim-write-probe-$$"
if (umask 077 && : > "$probe_file") 2>/dev/null; then
    printf 'project_write_probe=passed\n'
    rm -f "$probe_file"
else
    printf 'project_write_probe=failed\n'
fi

heading "result"
if [[ "$(uname -m)" != "aarch64" ]]; then
    printf 'FAIL: expected an ARM64 GPU node\n'
elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'FAIL: run this inside an interactive Slurm allocation\n'
elif ! command -v apptainer >/dev/null 2>&1; then
    printf 'FAIL: Apptainer is unavailable\n'
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'FAIL: NVIDIA GPU is unavailable\n'
else
    printf 'PASS: basic ARM64 GPU environment detected\n'
fi
