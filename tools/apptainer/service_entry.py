# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check or run existing services from this checkout in a selected environment."""

import argparse
import importlib
import importlib.metadata
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (
    "src/driver/src",
    "src/runtime",
    "src/controller",
    "src/physics",
    "src/utils",
    "src/plugins",
    "src/eval/src",
    "tools/apptainer",
)


def check_environment(profile, policy="vavam"):
    """Import service entry points without constructing services or models."""
    modules = {
        "core": [
            "alpasim_runtime.simulate.__main__",
            "alpasim_controller.server",
            "alpasim_physics.server",
            "alpasim_utils.logs",
            "PIL.Image",
        ],
        "policy": [
            "alpasim_driver.main",
            "alpasim_driver.models.vam_model"
            if policy == "vavam"
            else "alpasim_driver.models.alpamayo1_5_model",
            "alpasim_utils.logs",
        ],
    }[profile]
    distributions = {
        "core": [
            "numpy",
            "grpcio",
            "protobuf",
            "utils-rs",
            "aiofiles",
            "mergedeep",
            "polars",
            "prometheus-client",
            "rich",
            "requests",
            "pyarrow",
            "alpasim-controller",
        ],
        "policy": [
            "numpy",
            "grpcio",
            "protobuf",
            "utils-rs",
            "alpasim-driver",
            "alpasim-plugins",
            "torch",
            "opencv-python-headless",
        ],
    }[profile]
    if profile == "policy" and policy == "alpamayo1_5":
        distributions += ["alpamayo1_5", "transformers", "torchvision", "accelerate"]
    inventory = {}
    for name in distributions:
        try:
            inventory[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            inventory[name] = None
        print(f"{name}: {inventory[name] or 'NOT INSTALLED'}", flush=True)
    failures = {}
    for name in modules:
        print("Checking:", name, flush=True)
        try:
            importlib.import_module(name)
            print("PASS:", name, flush=True)
        except Exception:
            failures[name] = traceback.format_exc(limit=8)
            print(failures[name], flush=True)
    if profile == "policy":
        try:
            importlib.metadata.version("alpasim-driver")
            from alpasim_plugins.plugins import models

            models.get("vam" if policy == "vavam" else policy)
            print(
                f"PASS: installed {policy} entry point and driver version", flush=True
            )
        except Exception:
            failures["driver_registration"] = traceback.format_exc(limit=8)
            print(failures["driver_registration"], flush=True)
        if policy == "alpamayo1_5":
            try:
                from alpamayo1_5 import helper
                from transformers import AutoProcessor

                AutoProcessor.from_pretrained(
                    helper.BASE_PROCESSOR_NAME, local_files_only=True
                )
                print("PASS: cached Alpamayo image processor", flush=True)
            except Exception:
                failures["processor_cache"] = traceback.format_exc(limit=8)
                print(failures["processor_cache"], flush=True)
    else:
        try:
            from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory

            assert DynamicTrajectory and Pose and Trajectory
            from alpasim_plugins.plugins import mpc_controllers

            mpc_controllers.get("linear")
            print("PASS: native geometry and linear controller entry point", flush=True)
        except Exception:
            failures["geometry_controller"] = traceback.format_exc(limit=8)
            print(failures["geometry_controller"], flush=True)
    report = {
        "profile": profile,
        "policy": policy if profile == "policy" else None,
        "inventory": inventory,
        "failures": failures,
    }
    Path(os.environ["ALPASIM_RUN_DIR"], "environment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "PASS" if not failures else "BLOCKED",
        "service environment:",
        profile,
        flush=True,
    )
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", choices=("core", "policy"))
    group.add_argument("--module")
    parser.add_argument("--policy", choices=("vavam", "alpamayo1_5"), default="vavam")
    args, module_args = parser.parse_known_args()
    sys.path[:0] = [str(ROOT / name) for name in SOURCE_DIRS]
    if args.check:
        if module_args:
            parser.error("Unexpected arguments for --check")
        return check_environment(args.check, args.policy)
    sys.argv = [args.module, *module_args]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
