# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Inspect a synthetic USDZ using real Parquet files and the CPU plotting CLI."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "inspect_scene.py"


def point(x, y):
    return {"x": x, "y": y, "z": 0.0}


def parquet_bytes(kind, elements):
    table = pa.Table.from_pylist([{kind: element} for element in elements])
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    return buffer.getvalue().to_pybytes()


@pytest.fixture
def scene(tmp_path):
    lane = {
        "left_rail": [point(0.0, 0.0), point(0.0, 20.0)],
        "right_rail": [point(4.0, 0.0), point(4.0, 20.0)],
    }
    wait_line = {"location": [point(0.0, 10.0), point(4.0, 10.0)]}
    crosswalk = {"location": [point(0.0, 11.0), point(4.0, 11.0), point(4.0, 12.0)]}
    map_ego = [{"location": point(2.0, 1.0)}, {"location": point(2.0, 15.0)}]
    matrices = []
    for (x, y), heading in [((4.0, -3.0), 30.0), ((4.0, 8.0), 75.0)]:
        angle = np.deg2rad(heading)
        transform = np.eye(4)
        transform[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        transform[:2, 3] = [x, y]
        matrices.append(transform.tolist())
    rig = {
        "rig_trajectories": [
            {
                "T_rig_world_timestamps_us": [1_000_000, 6_000_000],
                "T_rig_worlds": matrices,
            }
        ]
    }
    seed = io.BytesIO()
    Image.new("RGB", (8, 8), (80, 120, 160)).save(seed, format="JPEG")
    path = tmp_path / "synthetic.usdz"
    with ZipFile(path, "w") as archive:
        for kind, rows in [
            ("lane", [lane]),
            ("wait_line", [wait_line]),
            ("crosswalk", [crosswalk]),
            ("egomotion_estimate", map_ego),
            ("intersection_area", [{"id": "synthetic-junction"}]),
        ]:
            archive.writestr(f"clipgt/{kind}.parquet", parquet_bytes(kind, rows))
        archive.writestr("rig_trajectories.json", json.dumps(rig))
        archive.writestr("frames/front/1000000.jpeg", seed.getvalue())
    return path


def inspect(scene, output, tmp_path):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--scene",
            str(scene),
            "--output-dir",
            str(output),
        ],
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl-cache")},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_parquet_scene_exports_map_heading_and_seed_evidence(scene, tmp_path):
    output = tmp_path / "inspection"
    result = inspect(scene, output, tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene-inspection.json").read_text())
    assert (
        report["lane"]["rows"]
        == report["wait_line"]["rows"]
        == report["crosswalk"]["rows"]
        == 1
    )
    assert report["egomotion_estimate"]["rows"] == 2
    assert "left_rail" in report["lane"]["schema"]
    assert report["intersection_area"]["sample"] == [
        {"intersection_area": {"id": "synthetic-junction"}}
    ]
    assert report["seed_images"] == ["frames/front/1000000.jpeg"]
    assert report["rig"] == {
        "first_timestamp_us": 1_000_000,
        "last_timestamp_us": 6_000_000,
        "first_xy": [4.0, -3.0],
        "last_xy": [4.0, 8.0],
    }
    samples = report["rig_samples"]
    assert [sample["timestamp_us"] for sample in samples] == [1_000_000, 6_000_000]
    assert [sample["elapsed_s"] for sample in samples] == [0.0, 5.0]
    assert [sample["heading_delta_deg"] for sample in samples] == pytest.approx(
        [0.0, 45.0]
    )
    for filename in ("map-overview.png", "recorded-route.png"):
        with Image.open(output / filename) as image:
            assert image.format == "PNG"
            assert image.width > 200 and image.height > 200
            assert np.ptp(np.array(image.convert("RGB"))) > 100
    assert "intersection approach still requires verification" in result.stdout


def test_inspector_refuses_to_overwrite_existing_output(scene, tmp_path):
    output = tmp_path / "inspection"
    first = inspect(scene, output, tmp_path)
    assert first.returncode == 0, first.stderr
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    second = inspect(scene, output, tmp_path)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert saved == {path.name: path.read_bytes() for path in output.iterdir()}
