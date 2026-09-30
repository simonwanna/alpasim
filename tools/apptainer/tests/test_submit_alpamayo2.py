# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "submit_alpamayo2.sh"


@pytest.fixture
def scheduler(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Capture scheduler arguments without submitting jobs or starting models.
    stub = """#!/usr/bin/env python3
import json, os, pathlib, signal, sys, time
pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))
if pathlib.Path(sys.argv[0]).name == 'sbatch':
    print('12345')
if os.environ.get('WAIT_FOR_SIGNAL'):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pathlib.Path(os.environ['READY']).touch()
    time.sleep(30)
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
"""
    for name in ("sbatch", "srun"):
        path = bin_dir / name
        path.write_text(stub)
        path.chmod(0o755)
    return dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        CAPTURE=str(tmp_path / "argv.json"),
        RUN_TIMEZONE="UTC",
    )


def test_submission_groups_all_outputs_and_reserves_two_gpus(tmp_path, scheduler):
    root = tmp_path / "personal outputs"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(root),
            "--project",
            str(tmp_path),
            "--navigation-instruction",
            "Turn right; keep going",
        ],
        env=scheduler,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(scheduler["CAPTURE"]).read_text())
    assert "--gpus=2" in args and "--ntasks=1" in args
    assert "--gpus-per-task=2" in args
    assert not any(a.startswith("--nodes") for a in args)
    assert not any(a.startswith("--array") for a in args)
    run = next(root.glob("*/*"))
    assert len(run.parent.name) == 10 and run.name.endswith("+0000")
    assert f"--output={run}/slurm.log" in args
    assert f"--error={run}/slurm.log" in args
    assert args[-1] == "Turn right; keep going"
    assert (run / "job-id.txt").read_text().strip() == "12345"
    assert (run / "code-revision.txt").is_file()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_records_timing_and_preserves_exit_status(
    tmp_path, scheduler, exit_code
):
    env = dict(scheduler, SLURM_JOB_ID="12345", FAKE_EXIT=str(exit_code))
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--worker",
            str(SCRIPT.parents[2]),
            str(tmp_path),
            "--project",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    args = json.loads(Path(scheduler["CAPTURE"]).read_text())
    assert "--gpus=2" in args
    assert args[args.index("--policy") + 1] == "alpamayo2"
    assert args[args.index("--policy-gpu") + 1] == "0"
    assert args[args.index("--renderer-gpu") + 1] == "1"
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "run")
    summary = (tmp_path / "job-summary.txt").read_text()
    assert f"exit_status={exit_code}\n" in summary
    assert "elapsed_seconds=" in summary and "excludes queue time" in summary


def test_worker_forwards_cancellation_and_records_failure(tmp_path, scheduler):
    ready = tmp_path / "ready"
    env = dict(scheduler, SLURM_JOB_ID="12345", WAIT_FOR_SIGNAL="1", READY=str(ready))
    process = subprocess.Popen(
        ["bash", str(SCRIPT), "--worker", str(SCRIPT.parents[2]), str(tmp_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
        assert "exit_status=143" in (tmp_path / "job-summary.txt").read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "option", ["--output-dir=/elsewhere", "--policy=vavam", "--renderer-gpu=0"]
)
def test_rejects_fixed_job_overrides_before_submission(tmp_path, scheduler, option):
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(output), option],
        env=scheduler,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not output.exists() and not Path(scheduler["CAPTURE"]).exists()
