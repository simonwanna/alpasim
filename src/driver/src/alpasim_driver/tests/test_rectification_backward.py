# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_driver.rectification import (
    _FthetaCamera,
    _scale_ftheta_intrinsics_to_resolution,
    build_ftheta_rectifier_for_resolution,
)
from alpasim_driver.schema import RectificationTargetConfig
from alpasim_grpc.v0 import sensorsim_pb2


def _backward_camera() -> sensorsim_pb2.FthetaCameraParam:
    intrinsics = sensorsim_pb2.FthetaCameraParam(
        reference_poly=sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE,
        principal_point_x=500.0,
        principal_point_y=300.0,
        pixeldist_to_angle_poly=[0.0, 0.002, 0.0, 5e-10],
    )
    intrinsics.linear_cde.linear_c = 1.1
    intrinsics.linear_cde.linear_d = 0.04
    intrinsics.linear_cde.linear_e = -0.02
    return intrinsics


@pytest.mark.parametrize("resolution", [(600, 1000), (300, 500), (240, 800)])
def test_backward_projection_round_trip_and_resize(resolution) -> None:
    """Invert a nonlinear calibration with a nontrivial C/D/E transform."""
    intrinsics = _backward_camera()
    x, y = np.meshgrid(np.linspace(10, 990, 79), np.linspace(10, 590, 53))
    pixels = np.stack((x, y), axis=-1)
    offsets = np.linalg.solve(
        np.array([[1.1, 0.04], [-0.02, 1.0]]),
        (pixels.reshape(-1, 2) - [500.0, 300.0]).T,
    ).T
    radii = np.linalg.norm(offsets, axis=1)
    theta = 0.002 * radii + 5e-10 * radii**3
    direction = np.divide(
        offsets, radii[:, None], out=np.zeros_like(offsets), where=radii[:, None] > 0
    )
    rays = np.column_stack((direction * np.sin(theta)[:, None], np.cos(theta)))
    rays = rays.reshape(*pixels.shape[:2], 3)

    scaled = _scale_ftheta_intrinsics_to_resolution(intrinsics, (600, 1000), resolution)
    projected, valid = _FthetaCamera(scaled, resolution).ray_to_pixel(rays)

    assert valid.shape == pixels.shape[:2]
    assert valid.all()
    expected = pixels * [resolution[1] / 1000, resolution[0] / 600]
    np.testing.assert_allclose(projected, expected, atol=0.001, rtol=0)


def test_explicit_backward_reference_overrides_forward_coefficients() -> None:
    intrinsics = _backward_camera()
    rays = np.array([[0.2, 0.1, 1.0]])
    expected, _ = _FthetaCamera(intrinsics, (600, 1000)).ray_to_pixel(rays)
    intrinsics.angle_to_pixeldist_poly.extend([0.0, 1.0])
    actual, valid = _FthetaCamera(intrinsics, (600, 1000)).ray_to_pixel(rays)
    np.testing.assert_array_equal(actual, expected)
    assert valid.all()


@pytest.mark.parametrize(
    "coefficients",
    [[], [0.0], [0.0, np.nan], [0.0, np.inf], [0.0, -0.002]],
)
def test_invalid_backward_calibration_is_rejected(coefficients) -> None:
    intrinsics = _backward_camera()
    intrinsics.pixeldist_to_angle_poly[:] = coefficients
    with pytest.raises(ValueError, match="backward calibration|Backward calibration"):
        _FthetaCamera(intrinsics, (600, 1000))


def test_backward_polynomial_with_interior_turn_is_rejected() -> None:
    """Positive endpoint slopes must not hide a negative interior derivative."""
    intrinsics = _backward_camera()
    # Derivative (r-100)(r-200)*1e-7 is negative between the two roots.
    intrinsics.pixeldist_to_angle_poly[:] = [0, 0.002, -1.5e-5, 1e-7 / 3]
    with pytest.raises(ValueError, match="not strictly increasing"):
        _FthetaCamera(intrinsics, (600, 1000))


def test_out_of_domain_rays_are_not_clamped_into_the_image() -> None:
    intrinsics = sensorsim_pb2.FthetaCameraParam(
        reference_poly=sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE,
        principal_point_x=50,
        principal_point_y=50,
        pixeldist_to_angle_poly=[0.01, 0.001],
    )
    rays = np.array([[0, 0, 1], [np.sin(0.5), 0, np.cos(0.5)], [0, 0, -1], [0, 0, 0]])
    _, valid = _FthetaCamera(intrinsics, (100, 100)).ray_to_pixel(rays)
    assert not valid.any()


def test_backward_projection_respects_max_angle() -> None:
    intrinsics = _backward_camera()
    intrinsics.max_angle = 0.1
    _, valid = _FthetaCamera(intrinsics, (600, 1000)).ray_to_pixel(
        np.array([[0, 0, 1], [np.sin(0.2), 0, np.cos(0.2)]])
    )
    np.testing.assert_array_equal(valid, [True, False])


def test_missing_forward_calibration_has_descriptive_error() -> None:
    with pytest.raises(ValueError, match="Missing forward calibration"):
        _FthetaCamera(sensorsim_pb2.FthetaCameraParam(), (100, 100))


def test_full_rectifier_remaps_backward_only_camera() -> None:
    """Exercise OpenCV remapping and the public builder without model assets."""
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(logical_id="front")
    camera.intrinsics.resolution_h = 600
    camera.intrinsics.resolution_w = 1000
    camera.intrinsics.ftheta_param.CopyFrom(_backward_camera())
    target = RectificationTargetConfig(
        focal_length=(100.0, 100.0),
        principal_point=(3.0, 2.0),
        resolution_hw=(5, 7),
    )
    rectifier = build_ftheta_rectifier_for_resolution(camera, target, (600, 1000))
    # Channel zero codes source x; channel one codes source y. The optical axis
    # must sample the original principal point, independently of the lookup table.
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    image[..., 0] = np.arange(1000, dtype=np.uint16)[None, :] % 251
    image[..., 1] = np.arange(600, dtype=np.uint16)[:, None] % 251
    image[..., 2] = 173
    result = rectifier.rectify(image)
    assert result.shape == (5, 7, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[2, 3], image[300, 500])
    assert (result[..., 2] == 173).all()
