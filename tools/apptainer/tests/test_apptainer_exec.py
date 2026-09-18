# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import argparse
import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "apptainer_exec.py"
SPEC = importlib.util.spec_from_file_location("apptainer_exec", MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def args(tmp_path):
    project = tmp_path / "project with spaces"
    for directory in (
        "apps/alpasim/.venv",
        "apps/alpasim/.venv-vavam",
        "apps/flashdreams/flashdreams/.venv",
        "shared/flashdreams/huggingface",
        "shared/flashdreams/scenes",
    ):
        path = project / directory
        path.mkdir(parents=True)
        if ".venv" in directory:
            (path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    image = project / "image.sif"
    image.touch()
    return argparse.Namespace(
        project=project,
        image=image,
        output_dir=project / "shared/run",
        cache_dir=project / "personal/cache",
        profile="policy",
        gpu=False,
        python_args=["-c", 'print("literal $HOME and `text`")'],
    )


@pytest.mark.parametrize(
    "profile,python",
    [
        ("core", "/workspace/apps/alpasim/.venv/bin/python"),
        ("policy", "/workspace/apps/alpasim/.venv-vavam/bin/python"),
        ("renderer", "/workspace/flashdreams/.venv/bin/python"),
    ],
)
def test_separate_environments_and_literal_arguments(args, profile, python):
    args.profile = profile
    command, env = launcher.build_command(args, {})
    assert command[-5:] == [python, "-B", "-u", *args.python_args]
    assert shlex.split(shlex.join(command)) == command
    assert "--cleanenv" in command
    assert "--nv" not in command
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert not args.output_dir.exists() and not args.cache_dir.exists()
    if profile == "renderer":
        assert env["HF_HOME"] == "/models"
        assert (
            f"{args.project}/apps/flashdreams/flashdreams:/workspace/flashdreams:ro"
            in command
        )
        assert f"{args.project}/shared/flashdreams/huggingface:/models:ro" in command


def test_gpu_selection_is_preserved(args):
    args.gpu = True
    command, env = launcher.build_command(args, {"CUDA_VISIBLE_DEVICES": "2"})
    assert "--nv" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["TRITON_LIBCUDA_PATH"] == "/.singularity.d/libs"
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        launcher.build_command(args, {})


@pytest.mark.parametrize("field", ["cache_dir", "output_dir", "image"])
def test_paths_outside_project_are_rejected(args, tmp_path, field):
    setattr(args, field, tmp_path / "outside")
    with pytest.raises(ValueError, match="inside project"):
        launcher.build_command(args, {})


def test_symlink_outside_project_is_rejected(args, tmp_path):
    args.cache_dir.parent.mkdir(parents=True)
    args.cache_dir.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="inside project"):
        launcher.build_command(args, {})


def test_output_and_cache_must_be_separate(args):
    args.cache_dir = args.output_dir / "cache"
    with pytest.raises(ValueError, match="overlap"):
        launcher.build_command(args, {})


def test_existing_results_are_preserved(args):
    args.output_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be new"):
        launcher.build_command(args, {})


def test_print_command_does_not_launch_or_write(args):
    command = [
        sys.executable,
        str(MODULE_PATH),
        "--project",
        str(args.project),
        "--image",
        str(args.image),
        "--profile",
        "policy",
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--timeout",
        "10",
        "--print-command",
        "--",
        "-V",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert shlex.split(result.stdout)[0] == "apptainer"
    assert not args.output_dir.exists() and not args.cache_dir.exists()


def test_stopping_group_does_not_touch_unrelated_process():
    owned = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        launcher.stop_group(owned)
        assert owned.poll() is not None
        assert other.poll() is None
    finally:
        launcher.stop_group(owned)
        launcher.stop_group(other)


@pytest.mark.parametrize("timeout,expected", [(False, 7), (True, 124)])
def test_execution_logs_and_exit_status(args, monkeypatch, timeout, expected):
    # A local fake Apptainer process executes no container and contacts no cluster.
    binary = args.project / "bin"
    binary.mkdir()
    stub = binary / "apptainer"
    stub.write_text(
        f"#!{sys.executable}\nimport time\nprint('local process', flush=True)\n"
        + ("time.sleep(60)\n" if timeout else "raise SystemExit(7)\n")
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SLURM_JOB_ID", "local-test-only")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--project",
            str(args.project),
            "--image",
            str(args.image),
            "--profile",
            "policy",
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(args.output_dir),
            "--timeout",
            "1",
            "--",
            "-V",
        ],
    )
    assert launcher.main() == expected
    assert "local process" in (args.output_dir / "process.log").read_text()
    assert (args.output_dir / "command.json").is_file()


def test_sigterm_cleans_up_gpu_monitor_and_owned_process(args):
    binary = args.project / "bin"
    binary.mkdir()
    for name in ("apptainer", "nvidia-smi"):
        stub = binary / name
        stub.write_text(
            f"#!{sys.executable}\nimport os, time\n"
            "print(os.getpid(), flush=True)\ntime.sleep(60)\n"
        )
        stub.chmod(0o755)
    command = [
        sys.executable,
        "-c",
        "import platform, runpy, sys; platform.machine=lambda:'aarch64'; "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
        str(MODULE_PATH),
        "--project",
        str(args.project),
        "--image",
        str(args.image),
        "--profile",
        "policy",
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--timeout",
        "20",
        "--gpu",
        "--",
        "-V",
    ]
    env = dict(
        os.environ,
        PATH=str(binary) + os.pathsep + os.environ["PATH"],
        SLURM_JOB_ID="local-test-only",
        CUDA_VISIBLE_DEVICES="0",
    )
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    children = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            logs = [args.output_dir / "process.log", args.output_dir / "gpu.csv"]
            if all(p.exists() and p.read_text().strip().isdigit() for p in logs):
                children = [int(p.read_text().strip()) for p in logs]
                break
            time.sleep(0.02)
        assert len(children) == 2, "Local test children did not start"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
