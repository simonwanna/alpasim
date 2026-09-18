# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from alpasim_driver.rectification import build_ftheta_rectifier_for_resolution
from alpasim_driver.schema import RectificationTargetConfig
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, video_model_pb2
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "overlay_plan.py"
SPEC = importlib.util.spec_from_file_location("overlay_plan", SCRIPT)
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


def test_mount_translation_axis_conversion_and_perspective():
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=1, y=0, z=2), quat=common_pb2.Quat(w=1)
    )
    points = np.array([[11, 0, 0], [11, 1, 0], [11, -1, 0], [-1, 0, 0]])
    optical = overlay.rig_to_optical(points, pose)
    np.testing.assert_allclose(optical[0], [0, 2, 10])
    target = RectificationTargetConfig(
        focal_length=(100, 100), principal_point=(100, 50), resolution_hw=(100, 200)
    )
    pixels, valid = overlay.project_rectified(optical, target)
    np.testing.assert_allclose(pixels[:3], [[100, 70], [90, 70], [110, 70]])
    np.testing.assert_array_equal(valid, [True, True, True, False])


def test_inverse_rotated_camera_mount():
    pose = common_pb2.Pose(
        vec=common_pb2.Vec3(x=2, y=3, z=1),
        quat=common_pb2.Quat(z=np.sqrt(0.5), w=np.sqrt(0.5)),
    )
    # Camera points left in the rig. The point is straight ahead of that camera.
    points = np.array([[2, 13, 0]], dtype=float)
    np.testing.assert_allclose(
        overlay.rig_to_optical(points, pose), [[0, 1, 10]], atol=1e-6
    )


def test_missing_rotation_rejected():
    with pytest.raises(ValueError, match="unit quaternion"):
        overlay.rig_to_optical(np.array([[10, 0, 0]]), common_pb2.Pose())


def test_radial_and_tangential_distortion_matches_analytic_projection():
    target = RectificationTargetConfig(
        focal_length=(100, 120),
        principal_point=(200, 100),
        resolution_hw=(300, 500),
        radial=(0.2,),
        tangential=(0.01, -0.02),
    )
    x, y = 0.2, 0.1
    r2 = x * x + y * y
    xd = x * (1 + 0.2 * r2) + 2 * 0.01 * x * y - 0.02 * (r2 + 2 * x * x)
    yd = y * (1 + 0.2 * r2) + 0.01 * (r2 + 2 * y * y) - 2 * 0.02 * x * y
    pixels, valid = overlay.project_rectified(np.array([[x, y, 1]]), target)
    np.testing.assert_allclose(pixels, [[200 + 100 * xd, 100 + 120 * yd]])
    assert valid.all()


def test_invalid_interval_is_not_bridged():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels = np.array([[10, 50], [np.nan, np.nan], [90, 50], [90, 70]])
    drawn = overlay.draw_plan(image, pixels, np.array([True, False, True, True]))
    assert not drawn[50, 50].any()
    assert drawn[60, 90].any()
    assert not image.any()


def test_dense_path_keeps_continuous_foreground():
    pixels = np.column_stack((np.arange(10, 91), np.full(81, 50)))
    drawn = overlay.draw_plan(
        np.zeros((100, 100, 3), dtype=np.uint8), pixels, np.ones(81, dtype=bool)
    )
    assert np.all(drawn[50, 10:91, 1] > 240)


def test_projection_matches_dot_rectified_from_resized_ftheta_image():
    """An independently placed source dot lands at the projected target ray."""
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera()
    camera.intrinsics.resolution_h = 600
    camera.intrinsics.resolution_w = 1000
    param = camera.intrinsics.ftheta_param
    param.reference_poly = sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE
    param.pixeldist_to_angle_poly.extend([0, 0.002])
    param.principal_point_x, param.principal_point_y = 500, 300
    target = RectificationTargetConfig(
        focal_length=(220, 210),
        principal_point=(160, 100),
        resolution_hw=(200, 320),
        radial=(-0.2, 0.05),
        tangential=(0.01, -0.02),
    )
    ray = np.array([[0.35, 0.2, 1]])
    xy = ray[0, :2]
    radius = np.arctan(np.linalg.norm(xy)) / 0.002
    # Unequal axis scales must apply once to the native analytic projection.
    source_pixel = (np.array([500, 300]) + radius * xy / np.linalg.norm(xy)) * [
        0.5,
        1 / 3,
    ]
    source = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.circle(source, tuple(np.rint(source_pixel).astype(int)), 2, (255, 0, 0), -1)
    rectifier = build_ftheta_rectifier_for_resolution(camera, target, source.shape[:2])
    rectified = rectifier.rectify(source)
    projected, valid = overlay.project_rectified(ray, target)
    assert valid.all()
    x, y = np.rint(projected[0]).astype(int)
    assert rectified[y - 2 : y + 3, x - 2 : x + 3, 0].max() > 200


@pytest.fixture
def saved_run(tmp_path):
    request = video_model_pb2.SessionRequest()
    spec = request.camera_specs.add(
        logical_id="camera_front_wide_120fov", resolution_h=300, resolution_w=500
    )
    spec.ftheta_param.reference_poly = (
        sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE
    )
    spec.ftheta_param.principal_point_x = 250
    spec.ftheta_param.principal_point_y = 150
    spec.ftheta_param.pixeldist_to_angle_poly.extend([0, 0.005])
    pose = request.rig_to_camera.add()
    pose.quat.w = 1
    pose.vec.x, pose.vec.z = 1, 1
    (tmp_path / "session.pb").write_bytes(request.SerializeToString())
    target_config = dict(
        focal_length=[200, 200],
        principal_point=[160, 100],
        resolution_hw=[200, 320],
        radial=[-0.1, 0.02],
    )
    source = np.full((300, 500, 3), 80, dtype=np.uint8)
    Image.fromarray(source).save(tmp_path / "source.png")
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id=spec.logical_id
    )
    camera.intrinsics.CopyFrom(spec)
    rectifier = build_ftheta_rectifier_for_resolution(
        camera, RectificationTargetConfig(**target_config), source.shape[:2]
    )
    Image.fromarray(rectifier.rectify(source)).save(tmp_path / "rectified.png")
    metadata = dict(
        target_config=target_config,
        source_shape=list(source.shape),
        frame_sha256=hashlib.sha256((tmp_path / "source.png").read_bytes()).hexdigest(),
        session_request_sha256=hashlib.sha256(
            (tmp_path / "session.pb").read_bytes()
        ).hexdigest(),
    )
    (tmp_path / "rectification.json").write_text(json.dumps(metadata))
    policy = dict(
        image_sha256=hashlib.sha256(
            (tmp_path / "rectified.png").read_bytes()
        ).hexdigest(),
        positions_m=[[4, 0, 0], [8, 0.5, 0], [15, -0.5, 0]],
    )
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    return tmp_path


def run_cli(work):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--policy-result",
            str(work / "policy.json"),
            "--rectification-result",
            str(work / "rectification.json"),
            "--rectified-image",
            str(work / "rectified.png"),
            "--source-frame",
            str(work / "source.png"),
            "--session-request",
            str(work / "session.pb"),
            "--output-dir",
            str(work / "output"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_renders_images_and_preserves_existing_output(saved_run):
    result = run_cli(saved_run)
    assert result.returncode == 0, result.stderr
    with Image.open(saved_run / "output/rendered-overlay.jpg") as image:
        assert image.size == (500, 300)
        assert np.array(image)[:, :, 1].max() > 200
    with Image.open(saved_run / "output/rectified-overlay.png") as image:
        assert image.size == (320, 200)
    report = json.loads((saved_run / "output/projection.json").read_text())
    assert report["source_visible_samples"] > 0
    assert report["rectified_visible_samples"] > 0
    assert run_cli(saved_run).returncode != 0


@pytest.mark.parametrize("filename", ["rectified.png", "source.png", "session.pb"])
def test_cli_rejects_mismatched_input_before_creating_output(saved_run, filename):
    with (saved_run / filename).open("ab") as file:
        file.write(b"changed")
    result = run_cli(saved_run)
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not (saved_run / "output").exists()
