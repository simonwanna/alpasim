# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check environments or launch a bounded native AlpaSim loop in Apptainer.

Uses existing software and local assets. No allocations, installs or downloads.
Default action checks imports only; --run starts services and a short rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from apptainer_exec import allocated_gpu, project_path

SERVICES = ("controller", "physics", "driver", "renderer")


def validate_alpamayo_checkpoint(path: Path, project: Path):
    """Check local checkpoint completeness without loading weights or downloading."""
    path = project_path(path, project)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(
            "Alpamayo checkpoint must be a local directory containing config.json"
        )
    json.loads((path / "config.json").read_text())
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            raise ValueError("Alpamayo checkpoint weight index is empty")
    else:
        shards = {"model.safetensors"}
    for name in shards:
        shard = project_path(path / name, project)
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Missing or empty Alpamayo checkpoint shard: {name}")


def reserve_ports():
    reservations = {}
    try:
        for name in SERVICES:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            reservations[name] = sock
    except BaseException:
        for sock in reservations.values():
            sock.close()
        raise
    return reservations


def stop_launchers(processes):
    """Let each launcher stop its own independently grouped container children."""
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 35
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class Run:
    def __init__(self, args):
        self.args = args
        self.root = Path(__file__).resolve().parents[2]
        self.project = args.project.resolve()
        self.work = project_path(args.output_dir, self.project)
        self.cache = project_path(args.cache_dir, self.project)
        self.image = project_path(args.image, self.project)
        self.policy_venv = project_path(
            args.policy_venv
            or self.project
            / "apps/alpasim"
            / (".venv-vavam" if args.policy == "vavam" else ".venv-alpamayo1_5"),
            self.project,
        )
        if args.policy_hf_home is not None:
            project_path(args.policy_hf_home, self.project)
        project_path(self.root, self.project)
        if (
            self.cache == self.work
            or self.cache.is_relative_to(self.work)
            or self.work.is_relative_to(self.cache)
        ):
            raise ValueError("Cache and output directories must not overlap")
        if not self.image.is_file():
            raise ValueError("Missing local container image")
        if self.work.exists():
            raise ValueError("Output directory must be new")
        self.deadline = time.monotonic() + args.timeout
        self.processes = []
        self.logs = []
        self.runtime_started = False
        self.export_completed = False

    def inside(self, path):
        return str(Path("/workspace") / Path(path).resolve().relative_to(self.project))

    def remaining(self):
        remaining = int(self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Overall run limit reached")
        return remaining

    def launch(self, name, profile, python_args, gpu=False, limit=None):
        timeout = self.remaining()
        if limit is not None:
            timeout = min(timeout, limit)
        command = [
            sys.executable,
            str(self.root / "tools/apptainer/apptainer_exec.py"),
            "--project",
            str(self.project),
            "--image",
            str(self.image),
            "--profile",
            profile,
            "--cache-dir",
            str(
                self.cache
                / (
                    self.args.policy
                    if profile == "policy" and self.args.policy != "vavam"
                    else profile
                )
            ),
            "--output-dir",
            str(self.work / name),
            "--timeout",
            str(timeout),
        ]
        if profile == "policy":
            command += ["--venv", str(self.policy_venv)]
            if self.args.policy_hf_home is not None:
                command += ["--hf-home", str(self.args.policy_hf_home)]
        if gpu:
            index = self.args.policy_gpu if name == "driver" else self.args.renderer_gpu
            command += ["--gpu", "--gpu-index", str(index)]
        command += ["--", *python_args]
        log = (self.work / f"{name}.launcher.log").open("w")
        self.logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(process)
        print(f"Started {name}; log: {self.work / name / 'process.log'}", flush=True)
        return process

    def entry(self, module, *arguments):
        return [
            "-I",
            self.inside(self.root / "tools/apptainer/service_entry.py"),
            "--module",
            module,
            *arguments,
        ]

    def finish(self, process, name):
        while process.poll() is None:
            self.remaining()
            time.sleep(0.2)
        if process.returncode != 0:
            for path in [
                self.work / f"{name}.launcher.log",
                self.work / name / "process.log",
            ]:
                if path.exists():
                    print(path.read_text(errors="replace")[-16000:], flush=True)
            raise RuntimeError(f"{name} failed with exit {process.returncode}")

    def check(self):
        failed = []
        for profile in ("core", "policy"):
            name = f"check-{profile}"
            process = self.launch(
                name,
                profile,
                [
                    "-I",
                    self.inside(self.root / "tools/apptainer/service_entry.py"),
                    "--check",
                    profile,
                    "--policy",
                    self.args.policy,
                ],
                limit=120,
            )
            try:
                self.finish(process, name)
            except RuntimeError as error:
                failed.append(str(error))
            else:
                print(
                    (self.work / name / "process.log").read_text(errors="replace"),
                    flush=True,
                )
        if failed:
            raise RuntimeError(
                "Environment checks failed; no GPU services were started"
            )
        print(
            "PASS: both service environments imported; models not started", flush=True
        )

    def wait_ready(self, services, ports):
        waiting = set(services)
        deadline = min(self.deadline, time.monotonic() + 600)
        next_report = 0
        while waiting:
            if time.monotonic() >= deadline:
                raise TimeoutError("Service startup deadline reached")
            for name, process in services.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} stopped during startup; see its process.log"
                    )
                if name in waiting:
                    try:
                        with socket.create_connection(
                            ("127.0.0.1", ports[name]), timeout=0.2
                        ):
                            waiting.remove(name)
                    except OSError:
                        pass
            if waiting and time.monotonic() >= next_report:
                print("Waiting for:", ", ".join(sorted(waiting)), flush=True)
                next_report = time.monotonic() + 15
            if waiting:
                time.sleep(0.5)

    def simulate(self):
        (self.work / "controller-output").mkdir()
        reservations = reserve_ports()
        try:
            ports = {name: sock.getsockname()[1] for name, sock in reservations.items()}
            spec = {
                name: self.inside(getattr(self.args, name))
                for name in ("scene", "checkpoint", "seed_session")
            }
            if self.args.tokenizer is not None:
                spec["tokenizer"] = self.inside(self.args.tokenizer)
            spec.update(
                work=self.inside(self.work),
                ports=ports,
                steps=self.args.steps,
                command=self.args.command,
                approach_steps=self.args.approach_steps,
                policy=self.args.policy,
                policy_gpu=self.args.policy_gpu,
                renderer_gpu=self.args.renderer_gpu,
            )
            spec_path = self.work / "launch.json"
            spec_path.write_text(json.dumps(spec, indent=2) + "\n")
            self.finish(
                self.launch(
                    "prepare",
                    "core",
                    self.entry("prepare_loop", "--spec", self.inside(spec_path)),
                    limit=90,
                ),
                "prepare",
            )
            configs = Path(self.inside(self.work / "configs"))
            definitions = {
                "controller": (
                    "core",
                    self.entry(
                        "alpasim_controller.server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(ports["controller"]),
                        "--log_dir",
                        self.inside(self.work / "controller-output"),
                    ),
                    False,
                ),
                "physics": (
                    "core",
                    self.entry(
                        "alpasim_physics.server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(ports["physics"]),
                        "--artifact-glob",
                        spec["scene"],
                        "--use-ground-mesh",
                        "True",
                        "--cache-size",
                        "1",
                    ),
                    True,
                ),
                "driver": (
                    "policy",
                    self.entry(
                        "alpasim_driver",
                        "--config-path",
                        str(configs),
                        "--config-name",
                        "driver",
                        "hydra.run.dir=.",
                    ),
                    True,
                ),
                "renderer": (
                    "renderer",
                    [
                        "-I",
                        "-m",
                        "omnidreams.impl.grpc.server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(ports["renderer"]),
                        "--max_workers",
                        "1",
                        "--pipeline_config_name",
                        "omnidreams",
                        "--resolution",
                        "704p",
                    ],
                    True,
                ),
            }
            services = {}
            for name, (profile, arguments, gpu) in definitions.items():
                reservations[name].close()
                services[name] = self.launch(name, profile, arguments, gpu=gpu)
            self.wait_ready(services, ports)
            print(
                "Services listening; native runtime will validate their RPC versions",
                flush=True,
            )
            runtime = self.launch(
                "runtime",
                "core",
                self.entry(
                    "alpasim_runtime.simulate",
                    "--user-config",
                    str(configs / "user.yaml"),
                    "--network-config",
                    str(configs / "network.yaml"),
                    "--eval-config",
                    str(configs / "eval.yaml"),
                    "--log-dir",
                    self.inside(self.work / "simulation"),
                ),
            )
            self.runtime_started = True
            while runtime.poll() is None:
                self.remaining()
                for name, process in services.items():
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{name} stopped during rollout; see its process.log"
                        )
                time.sleep(0.5)
            self.finish(runtime, "runtime")
            stop_launchers(list(services.values()))
            logs = list((self.work / "simulation" / "rollouts").rglob("rollout.asl"))
            if len(logs) != 1:
                raise RuntimeError(f"Expected one rollout log; found {len(logs)}")
            self.finish(
                self.launch(
                    "export",
                    "policy",
                    self.entry(
                        "export_loop",
                        "--log",
                        self.inside(logs[0]),
                        "--output-dir",
                        self.inside(self.work / "video"),
                        "--overlay",
                        "--overlay-reference",
                        "both",
                        "--overlay-min-forward-m",
                        str(self.args.overlay_min_forward_m),
                        "--require-closed-loop",
                    ),
                    limit=self.args.export_timeout,
                ),
                "export",
            )
            self.export_completed = True
            print("PASS: native closed-loop activity validated", flush=True)
            print("Video:", self.work / "video" / "preview-slow.gif", flush=True)
            print(
                "Plan overlay:",
                self.work / "video" / "plan-overlay-slow.gif",
                flush=True,
            )
            print(
                "Ego display:",
                self.work / "video" / "ego-plan-overlay-slow.gif",
                flush=True,
            )
        finally:
            for sock in reservations.values():
                sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "image", "cache-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--export-timeout",
        type=int,
        default=600,
        help="GIF export limit in seconds, within the overall timeout",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--policy", choices=("vavam", "alpamayo1_5"), default="vavam")
    parser.add_argument(
        "--policy-venv",
        type=Path,
        help="Existing policy environment; default apps/alpasim/.venv-<policy>",
    )
    parser.add_argument(
        "--policy-hf-home",
        type=Path,
        help="Offline HF cache containing the policy's processor assets",
    )
    parser.add_argument(
        "--policy-gpu",
        type=int,
        default=0,
        help="Policy GPU ordinal within CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--renderer-gpu",
        type=int,
        default=0,
        help="Renderer/physics GPU ordinal within CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--overlay-min-forward-m",
        type=float,
        default=0.0,
        help="Hide overlay samples closer than this rig-forward distance; display only",
    )
    parser.add_argument(
        "--approach-steps",
        type=int,
        default=None,
        help="Extra recorded control intervals before policy handover (default: VaVAM 0, Alpamayo 6)",
    )
    parser.add_argument(
        "--command",
        choices=("straight", "left", "right"),
        help="VaVAM instruction (default straight); Alpamayo uses the recorded route",
    )
    for name in ("scene", "checkpoint", "tokenizer", "seed-session"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.policy == "alpamayo1_5":
        if args.command is not None or args.tokenizer is not None:
            parser.error(
                "Alpamayo uses a checkpoint directory and recorded-route navigation, not --command/--tokenizer"
            )
        if args.approach_steps is None:
            args.approach_steps = 6
        if args.approach_steps < 6:
            parser.error("Alpamayo requires at least 6 approach steps for ego history")
    else:
        args.command = args.command or "straight"
        if args.approach_steps is None:
            args.approach_steps = 0
    if not 4 <= args.steps <= 180 or args.timeout <= 0:
        parser.error("Use 4..180 simulation steps and a positive timeout")
    if args.export_timeout <= 0:
        parser.error("Export timeout must be positive")
    if not math.isfinite(args.overlay_min_forward_m) or args.overlay_min_forward_m < 0:
        parser.error("Overlay minimum forward distance must be finite and nonnegative")
    if not 0 <= args.approach_steps <= args.steps - 3:
        parser.error("Approach must leave at least two policy-controlled intervals")
    if args.policy_gpu < 0 or args.renderer_gpu < 0:
        parser.error("GPU ordinals must be nonnegative")
    if args.run:
        names = ["scene", "checkpoint", "seed_session"]
        if args.policy == "vavam":
            names.append("tokenizer")
        for name in names:
            path = getattr(args, name)
            if name == "checkpoint" and args.policy == "alpamayo1_5":
                if path is None:
                    parser.error(
                        "Alpamayo --run requires a local --checkpoint directory"
                    )
                try:
                    validate_alpamayo_checkpoint(path, args.project.resolve())
                except (ValueError, KeyError, TypeError, OSError) as error:
                    parser.error(str(error))
                continue
            if path is None or not path.is_file():
                parser.error(f"--run requires existing --{name.replace('_', '-')} file")
            project_path(path, args.project.resolve())
        try:
            allocated_gpu(os.environ, args.policy_gpu)
            allocated_gpu(os.environ, args.renderer_gpu)
        except ValueError as error:
            parser.error(str(error))
    run = Run(args)
    os.umask(0o002)
    run.work.mkdir(parents=True)
    run.work.chmod(0o2770)
    status = 1

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    previous = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run.check()
        if args.run:
            run.simulate()
        status = 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"STOPPED: {type(error).__name__}: {error}", flush=True)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        stop_launchers(run.processes)
        for log in run.logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        (run.work / "status.json").write_text(
            json.dumps(
                {
                    "exit_status": status,
                    "requested_simulation": args.run,
                    "runtime_started": run.runtime_started,
                    "export_completed": run.export_completed,
                    "steps": args.steps,
                    "command": args.command,
                    "approach_steps": args.approach_steps,
                    "policy": args.policy,
                    "navigation": "recorded_route"
                    if args.policy == "alpamayo1_5"
                    else "fixed_command",
                    "policy_gpu": args.policy_gpu,
                    "renderer_gpu": args.renderer_gpu,
                }
            )
            + "\n"
        )
    print("Exit status:", status, "Outputs:", run.work, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
