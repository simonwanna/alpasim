# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CPU-only synthetic check of custom route configuration, geometry and guidance.

Run through service_entry in the core environment. No scene, model weights,
simulator services or GPU are needed. This does not test policy compliance.
"""

import runpy
from pathlib import Path

import numpy as np
from alpasim_runtime.config import RouteGeneratorType, SimulationConfig
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_utils.geometry import Pose
from omegaconf import OmegaConf


def main():
    # Load this pure NumPy adapter without importing the model package (torch).
    nav_path = (
        Path(__file__).resolve().parents[2]
        / "src/driver/src/alpasim_driver/models/nav_text.py"
    )
    route_to_nav_text = runpy.run_path(str(nav_path))["route_to_nav_text"]
    recorded = np.array([[0, 0, 0], [120, 0, 0]], dtype=float)
    pose = Pose(np.zeros(3), np.array([0, 0, 0, 1], dtype=float))
    theta = np.linspace(0, np.pi / 2, 25)
    left = np.vstack(
        (
            [[0, 0, 0]],
            np.column_stack(
                (10 + 12 * np.sin(theta), 12 * (1 - np.cos(theta)), theta * 0)
            ),
            [[22, 100, 0]],
        )
    )
    right = left * [1, -1, 1]
    for name, points, expected in (
        ("straight", recorded, "Continue straight"),
        ("left", left, "Turn left in "),
        ("right", right, "Turn right in "),
    ):
        cfg = OmegaConf.merge(
            OmegaConf.structured(SimulationConfig),
            dict(
                n_sim_steps=1,
                n_rollouts=1,
                cameras=[],
                route_generator_type="CUSTOM",
                route_waypoints_in_local=points.tolist(),
            ),
        )
        cfg = OmegaConf.to_object(cfg)
        assert cfg.route_generator_type == RouteGeneratorType.CUSTOM
        route = RouteGenerator.create(
            recorded,
            None,
            cfg.route_generator_type,
            custom_waypoints_in_local=cfg.route_waypoints_in_local,
        )
        np.testing.assert_allclose(route.route_polyline_in_local.points[-1], points[-1])
        projected = route.generate_route(0, pose)
        prepared = RouteGenerator.prepare_for_policy(projected)
        assert np.isfinite(prepared.waypoints).all(), "Route was unexpectedly truncated"
        guidance = route_to_nav_text(prepared.waypoints)
        assert guidance and guidance.startswith(expected), (name, guidance)
        print(f"PASS: custom {name} route -> {guidance}", flush=True)
    print("PASS: native route configuration, resampling and navigation guidance")
    print("Synthetic check only; no policy inference or driving was tested.")


if __name__ == "__main__":
    main()
