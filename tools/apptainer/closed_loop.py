# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check environments or launch a bounded native AlpaSim loop in Apptainer.

Uses existing software and local assets. No allocations, installs or downloads.
Default action checks imports only; --run starts services and a short rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from apptainer_exec import project_path

SERVICES = ("controller", "physics", "driver", "renderer")


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
            str(self.cache / profile),
            "--output-dir",
            str(self.work / name),
            "--timeout",
            str(timeout),
        ]
        if gpu:
            command.append("--gpu")
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
                for name in ("scene", "checkpoint", "tokenizer", "seed_session")
            }
            spec.update(
                work=self.inside(self.work),
                ports=ports,
                steps=self.args.steps,
                command=self.args.command,
                approach_steps=self.args.approach_steps,
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
                        "--require-closed-loop",
                    ),
                    limit=120,
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
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--approach-steps",
        type=int,
        default=0,
        help="Extra recorded control intervals before policy handover",
    )
    parser.add_argument(
        "--command", choices=("straight", "left", "right"), default="straight"
    )
    for name in ("scene", "checkpoint", "tokenizer", "seed-session"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if not 4 <= args.steps <= 180 or args.timeout <= 0:
        parser.error("Use 4..180 simulation steps and a positive timeout")
    if not 0 <= args.approach_steps <= args.steps - 3:
        parser.error("Approach must leave at least two policy-controlled intervals")
    if args.run:
        for name in ("scene", "checkpoint", "tokenizer", "seed_session"):
            path = getattr(args, name)
            if path is None or not path.is_file():
                parser.error(f"--run requires existing --{name.replace('_', '-')} file")
            project_path(path, args.project.resolve())
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("--run requires the allocation's CUDA_VISIBLE_DEVICES")
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
                }
            )
            + "\n"
        )
    print("Exit status:", status, "Outputs:", run.work, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
