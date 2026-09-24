# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Write configurations for a short native-runtime loop from existing assets."""

import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile


def validate_route_file(route, scene_id=None):
    """Require explicit units/frame and scene binding for a supplied route."""
    if not isinstance(route, dict) or route.get("coordinate_frame") != "local":
        raise ValueError(
            "Route coordinate_frame must be local (native scene coordinates)"
        )
    if route.get("units") != "metres":
        raise ValueError("Route units must be metres")
    if not isinstance(route.get("scene_id"), str) or not route["scene_id"]:
        raise ValueError("Route requires a scene_id")
    if scene_id is not None and route["scene_id"] != scene_id:
        raise ValueError("Route scene_id does not match the selected scene")
    points = route.get("waypoints")
    if not isinstance(points, list) or not 2 <= len(points) <= 10000:
        raise ValueError("Route requires 2..10000 XYZ waypoints")
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 3
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in point
            )
        ):
            raise ValueError("Route waypoints must be finite XYZ numbers")
    if not any(math.dist(points[0], p) >= 0.2 for p in points[1:]):
        raise ValueError("Route needs distinct waypoints")


def build_configs(spec, scene_id, rectification, prompt):
    work = Path(spec["work"])
    ports = spec["ports"]
    services = ("driver", "controller", "physics", "renderer")
    approach_steps = spec["approach_steps"]
    if not 0 <= approach_steps <= spec["steps"] - 3:
        raise ValueError(
            "Recorded approach must leave at least two closed-loop intervals"
        )
    policy = spec["policy"]
    if policy not in ("vavam", "alpamayo1_5"):
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "alpamayo1_5" and approach_steps < 6:
        raise ValueError("Alpamayo needs at least 6 approach steps for ego history")
    user = {
        "nr_workers": 1,
        "max_rollout_retries": 0,
        "enable_autoresume": False,
        "smooth_trajectories": True,
        "extra_cameras": [],
        "prometheus": {"url": None, "worker_ports": [0]},
        "scene_provider": {
            "kind": "usdz",
            "usdz": {"data_dir": spec["scene"], "artifact_cache_size": 1},
        },
        "scenes": [{"scene_id": scene_id}],
        "endpoints": {
            **{name: {"skip": False, "n_concurrent_rollouts": 1} for name in services},
            "trafficsim": {"skip": True, "n_concurrent_rollouts": 1},
            "startup_timeout_s": 600,
            "do_shutdown": False,
        },
        "renderer": {
            "kind": "video_model",
            "video_model_config": {
                "fps": 30,
                "first_chunk_frames": 5,
                "chunk_frames": 8,
                "text_prompt_positive": prompt["positive"],
                "text_prompt_negative": prompt["negative"],
            },
        },
        "simulation_config": {
            "n_sim_steps": spec["steps"],
            "n_rollouts": 1,
            "control_timestep_us": 8 * 33333,
            "force_gt_duration_us": (13 + 8 * approach_steps) * 33333,
            "skip_driver_during_force_gt": approach_steps > 0,
            "planner_delay_us": 0,
            "assert_zero_decision_delay": True,
            "physics_update_mode": "EGO_ONLY",
            "route_generator_type": "RECORDED",
            "ego_mask_rig_config_id": None,
            "render_bundling": "NONE",
            "cameras": [
                {
                    "logical_id": "camera_front_wide_120fov",
                    "height": 704,
                    "width": 1280,
                    "frame_interval_us": 33333,
                    "shutter_duration_us": 30000,
                }
            ],
        },
    }
    network = {
        name: {"endpoints": [{"address": f"127.0.0.1:{ports[name]}", "managed": False}]}
        for name in services
    }
    network["trafficsim"] = {"endpoints": []}
    driver = {
        "host": "127.0.0.1",
        "port": ports["driver"],
        "output_dir": str(work / "driver-output"),
        "model": {
            "model_type": "vam" if policy == "vavam" else policy,
            "checkpoint_path": spec["checkpoint"],
            "device": "cuda:0",
            "image_decode_device": "cpu",
        },
        "inference": {
            "use_cameras": ["camera_front_wide_120fov"],
            "max_batch_size": 1,
            "context_length": 1 if policy == "vavam" else 4,
            "subsample_factor": 1 if policy == "vavam" else 3,
        },
        "route": {
            "use_waypoint_commands": False,
        },
        "trajectory_optimizer": {"enabled": False},
    }
    if policy == "vavam":
        driver["model"]["tokenizer_path"] = spec["tokenizer"]
        driver["route"]["default_command"] = {"right": 0, "left": 1, "straight": 2}[
            spec["command"]
        ]
        driver["rectification"] = rectification
    else:
        # The upstream single-camera preset uses four temporal frames. Navigation
        # comes from route geometry via Alpamayo's route_to_nav_text adapter.
        driver["model"].update(num_trajectory_samples=1, cfg_guidance_weight=None)
        user["simulation_config"]["skip_driver_during_force_gt"] = True
    if spec.get("route") is not None:
        if policy != "alpamayo1_5":
            raise ValueError("Explicit route requires the Alpamayo policy")
        validate_route_file(spec["route"], scene_id)
        user["simulation_config"].update(
            route_generator_type="CUSTOM",
            route_waypoints_in_local=spec["route"]["waypoints"],
        )
    return user, network, driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    import yaml
    from alpasim_driver.schema import DriverConfig
    from alpasim_grpc.v0 import video_model_pb2
    from alpasim_runtime.config import NetworkSimulatorConfig, UserSimulatorConfig
    from alpasim_utils.scenario import Rig
    from alpasim_utils.yaml_utils import typed_parse_config
    from eval.schema import EvalConfig
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    with ZipFile(spec["scene"]) as archive:
        metadata = yaml.safe_load(archive.read("metadata.yaml"))
        (rig,) = Rig.load_from_json(archive.read("rig_trajectories.json").decode())
    scene_id = metadata["scene_id"]
    request = video_model_pb2.SessionRequest.FromString(
        Path(spec["seed_session"]).read_bytes()
    )
    prompt = {
        "positive": request.text_prompt.positive,
        "negative": request.text_prompt.negative,
    }
    rectification = None
    if spec["policy"] == "vavam":
        rectification = yaml.safe_load(
            (root / "src/wizard/configs/driver/vavam_video_model.yaml").read_text()
        )["driver"]["rectification"]
    user, network, driver = build_configs(spec, scene_id, rectification, prompt)
    if spec.get("route") is not None:
        from alpasim_runtime.config import RouteGeneratorType
        from alpasim_runtime.route_generator import RouteGenerator

        # Validate route geometry before the launcher starts any GPU services.
        RouteGenerator.create(
            rig.trajectory.positions,
            vector_map=None,
            route_generator_type=RouteGeneratorType.CUSTOM,
            custom_waypoints_in_local=spec["route"]["waypoints"],
        )
    anchor_us = rig.first_camera_frame_end_us(["camera_front_wide_120fov"])
    end_us = anchor_us + (5 + 8 * spec["steps"]) * 33333
    if end_us >= rig.trajectory.time_range_us.stop:
        raise ValueError(
            "Requested approach and maneuver exceed the recorded scene duration"
        )
    work = Path(spec["work"])
    config_dir = work / "configs"
    config_dir.mkdir()
    evaluation = yaml.safe_load(
        (root / "src/wizard/configs/base_config.yaml").read_text()
    )["eval"]
    evaluation.update(enabled=False, num_processes=1)
    evaluation["video"]["render_video"] = False
    evaluation["database"] = {
        "upload_metadata": False,
        "upload_leaderboard": False,
        "upload_full_metrics": False,
    }
    for name, content, schema in [
        ("user", user, UserSimulatorConfig),
        ("network", network, NetworkSimulatorConfig),
        ("driver", driver, DriverConfig),
        ("eval", evaluation, EvalConfig),
    ]:
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(content, sort_keys=False))
        OmegaConf.to_container(
            typed_parse_config(path, schema), resolve=True, throw_on_missing=True
        )
    simulation = work / "simulation"
    simulation.mkdir()
    (simulation / "run_metadata.yaml").write_text("run_name: native-closed-loop\n")
    print(
        "PASS: native runtime and service configs parsed; scene:", scene_id, flush=True
    )
    print(
        "Policy handover offset (s):",
        user["simulation_config"]["force_gt_duration_us"] / 1e6,
        flush=True,
    )
    print(
        "Policy:",
        spec["policy"],
        "| Navigation:",
        "recorded route" if spec["policy"] == "alpamayo1_5" else spec["command"],
        flush=True,
    )
    print("Configured rollout span (s):", (end_us - anchor_us) / 1e6, flush=True)


if __name__ == "__main__":
    main()
