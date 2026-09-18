# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run one Python process in an existing, separate AlpaSim Apptainer environment.

No SSH, allocation requests, installation, image builds or model downloads.
The caller supplies Python arguments after --. This is not a simulator orchestrator.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import signal
import subprocess
from pathlib import Path


def project_path(value: Path, project: Path) -> Path:
    path = value.resolve()
    if not path.is_relative_to(project) or path == project:
        raise ValueError(f"Path must be inside project storage: {value}")
    if any(character in str(path) for character in ",:\n"):
        raise ValueError(
            "Apptainer bind paths cannot contain commas, colons or newlines"
        )
    return path


def build_command(args: argparse.Namespace, environ: dict) -> tuple[list[str], dict]:
    project = args.project.resolve()
    if not project.is_dir() or any(c in str(project) for c in ",:\n"):
        raise ValueError(
            "--project must be an existing directory with a bind-compatible path"
        )
    image = project_path(args.image, project)
    output = project_path(args.output_dir, project)
    cache = project_path(args.cache_dir, project)
    if not image.is_file():
        raise ValueError("--image must name an existing local SIF")
    if output.exists():
        raise ValueError(
            "--output-dir must be new; previous results will not be overwritten"
        )
    if cache == output or cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Output and cache directories must not overlap")
    mount = Path("/project" if args.profile == "renderer" else "/workspace")

    def inside(path: Path) -> str:
        return str(mount / path.relative_to(project))

    binds = [f"{project}:{mount}"]
    if args.profile == "renderer":
        renderer = project / "apps/flashdreams/flashdreams"
        models = project / "shared/flashdreams/huggingface"
        scenes = project / "shared/flashdreams/scenes"
        for path in (renderer, models, scenes):
            if not project_path(path, project).is_dir():
                raise ValueError(f"Missing renderer directory: {path}")
        if not (renderer / ".venv/pyvenv.cfg").is_file():
            raise ValueError("Missing renderer environment")
        binds += [
            f"{renderer}:/workspace/flashdreams:ro",
            f"{models}:/models:ro",
            f"{scenes}:/scenes:ro",
        ]
        python = "/workspace/flashdreams/.venv/bin/python"
        hf_home = "/models"
    else:
        environment = ".venv-vavam" if args.profile == "policy" else ".venv"
        if not (project / "apps/alpasim" / environment / "pyvenv.cfg").is_file():
            raise ValueError(f"Missing AlpaSim {args.profile} environment")
        python = f"/workspace/apps/alpasim/{environment}/bin/python"
        hf_home = inside(cache / "huggingface")
    container_cache = inside(cache)
    environment = {
        "TMPDIR": f"{container_cache}/tmp",
        "XDG_CACHE_HOME": f"{container_cache}/xdg",
        "TORCH_HOME": f"{container_cache}/torch",
        "TORCH_EXTENSIONS_DIR": f"{container_cache}/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": f"{container_cache}/inductor",
        "TRITON_CACHE_DIR": f"{container_cache}/triton",
        "CUDA_CACHE_PATH": f"{container_cache}/cuda",
        "WARP_CACHE_PATH": f"{container_cache}/warp",
        "MPLCONFIGDIR": f"{container_cache}/matplotlib",
        "FLASHDREAMS_CACHE_DIR": f"{container_cache}/flashdreams",
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": f"{hf_home}/hub",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ALPASIM_RUN_DIR": inside(output),
    }
    command = ["apptainer", "exec", "--cleanenv"]
    if args.gpu:
        if not environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("CUDA_VISIBLE_DEVICES must identify the allocated GPU(s)")
        command.append("--nv")
        environment["CUDA_VISIBLE_DEVICES"] = environ["CUDA_VISIBLE_DEVICES"]
        environment["TRITON_LIBCUDA_PATH"] = "/.singularity.d/libs"
    for bind in binds:
        command += ["--bind", bind]
    # /usr/bin/env uses argv directly, avoiding Apptainer --env shell expansion.
    command += ["--pwd", inside(output), str(image), "/usr/bin/env"]
    command += [f"{key}={value}" for key, value in environment.items()]
    command += [python, "-B", "-u", *args.python_args]
    return command, environment


def stop_group(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # The group may contain children even if the direct process already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("core", "policy", "renderer"), required=True
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeout", type=int, required=True, help="Maximum process runtime in seconds"
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print only; no files or processes created",
    )
    parser.add_argument("python_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.python_args[:1] == ["--"]:
        args.python_args = args.python_args[1:]
    if args.timeout <= 0 or not args.python_args:
        parser.error("Provide a positive timeout and Python arguments after --")
    try:
        command, environment = build_command(args, os.environ)
    except ValueError as error:
        parser.error(str(error))
    if args.print_command:
        print(shlex.join(command))
        return 0
    if platform.machine() != "aarch64" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside your existing ARM64 Slurm allocation")

    os.umask(0o002)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.chmod(0o2770)
    for directory in (
        "tmp",
        "xdg",
        "torch",
        "torch-extensions",
        "inductor",
        "triton",
        "cuda",
        "warp",
        "matplotlib",
        "flashdreams",
        "huggingface",
    ):
        (args.cache_dir / directory).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Log: {args.output_dir / 'process.log'}", flush=True)
    process = monitor = None

    def interrupt(signum, frame):
        raise InterruptedError(signum)

    previous_handlers = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    with (args.output_dir / "process.log").open("w") as log, (
        args.output_dir / "gpu.csv"
    ).open("w") as gpu_log:
        try:
            if args.gpu:
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
                        "--format=csv",
                        "--loop=2",
                    ],
                    stdout=gpu_log,
                    stderr=log,
                    start_new_session=True,
                )
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            status = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("Runtime limit reached; stopping this run's processes", flush=True)
            status = 124
        except KeyboardInterrupt:
            status = 130
        except InterruptedError as error:
            status = 128 + error.args[0]
        finally:
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            stop_group(process)
            stop_group(monitor)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    print(f"Process exit status: {status}", flush=True)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
