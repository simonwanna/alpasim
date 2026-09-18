# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Overlay a saved rig-frame plan on its rendered and rectified camera images.

For video-model sessions built from USDZ sensor-to-rig FLU calibration.
CPU only. Shows the predicted plan, without terrain adjustment or occlusion.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def rig_to_optical(points, pose):
    """Invert the recorded sensor-to-rig FLU pose, then convert FLU to RDF."""
    translation = np.array([pose.vec.x, pose.vec.y, pose.vec.z], dtype=np.float64)
    q = np.array([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w], dtype=np.float64)
    if not np.isfinite(translation).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite camera pose")
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-4):
        raise ValueError("Camera pose must contain a unit quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    sensor = (points - translation) @ rotation
    return np.column_stack((-sensor[:, 1], -sensor[:, 2], sensor[:, 0]))


def project_rectified(points, target):
    """Use the same OpenCV lens model that the rectifier puts in the image."""
    from alpasim_driver.rectification import _dist_coeff_vector

    fx, fy = target.focal_length
    cx, cy = target.principal_point
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-3)
    pixels = np.full((len(points), 2), np.nan)
    if valid.any():
        projected, _ = cv2.projectPoints(
            points[valid], np.zeros(3), np.zeros(3), matrix, _dist_coeff_vector(target)
        )
        pixels[valid] = projected.reshape(-1, 2)
    height, width = target.resolution_hw
    valid &= np.isfinite(pixels).all(axis=1)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, valid


def draw_plan(image, pixels, valid):
    """Draw connected visible samples without bridging invalid intervals."""
    output = image.copy()
    # Finish the outline before the foreground so adjacent segments stay green.
    for color, thickness in [((20, 20, 20), 9), ((60, 255, 110), 5)]:
        for index in range(1, len(pixels)):
            if valid[index - 1] and valid[index]:
                a, b = np.rint(pixels[index - 1 : index + 1]).astype(int)
                cv2.line(output, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
    return output


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--rectification-result", type=Path, required=True)
    parser.add_argument("--rectified-image", type=Path, required=True)
    parser.add_argument("--source-frame", type=Path, required=True)
    parser.add_argument("--session-request", type=Path, required=True)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument("--grpc-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.grpc_wheel:
        if not args.grpc_wheel.is_file():
            parser.error("--grpc-wheel must name an existing wheel")
        sys.path.insert(0, str(args.grpc_wheel.resolve()))
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src/driver/src"))

    from alpasim_driver.rectification import (
        _FthetaCamera,
        _scale_ftheta_intrinsics_to_resolution,
    )
    from alpasim_driver.schema import RectificationTargetConfig
    from alpasim_grpc.v0 import video_model_pb2

    cv2.setNumThreads(1)
    policy = json.loads(args.policy_result.read_text())
    metadata = json.loads(args.rectification_result.read_text())
    image_hash = hashlib.sha256(args.rectified_image.read_bytes()).hexdigest()
    if policy["image_sha256"] != image_hash:
        raise ValueError("Rectified image does not match the policy input hash")
    source_hash = hashlib.sha256(args.source_frame.read_bytes()).hexdigest()
    session_hash = hashlib.sha256(args.session_request.read_bytes()).hexdigest()
    if "frame_sha256" in metadata and metadata["frame_sha256"] != source_hash:
        raise ValueError("Source frame does not match rectification metadata")
    if "source_frame" in metadata:
        if Path(metadata["source_frame"]).name != args.source_frame.name:
            raise ValueError("Source frame name does not match rectification metadata")
    if "session_request_sha256" in metadata:
        if metadata["session_request_sha256"] != session_hash:
            raise ValueError("Session request does not match rectification metadata")
    source = load_rgb(args.source_frame)
    rectified = load_rgb(args.rectified_image)
    if list(source.shape) != metadata["source_shape"]:
        raise ValueError("Source frame shape does not match rectification metadata")
    target = RectificationTargetConfig(**metadata["target_config"])
    if rectified.shape != (*target.resolution_hw, 3):
        raise ValueError("Rectified image shape does not match camera configuration")
    request = video_model_pb2.SessionRequest.FromString(
        args.session_request.read_bytes()
    )
    indices = [
        i
        for i, spec in enumerate(request.camera_specs)
        if spec.logical_id == args.camera
    ]
    if len(indices) != 1 or len(request.rig_to_camera) != len(request.camera_specs):
        raise ValueError("Expected one matching camera and an extrinsic for every view")
    index = indices[0]
    spec = request.camera_specs[index]
    if spec.WhichOneof("camera_param") != "ftheta_param":
        raise ValueError("This tool expects recorded f-theta camera calibration")
    positions = np.asarray(policy["positions_m"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("Expected at least two rig-frame 3D waypoints")
    if not np.isfinite(positions).all():
        raise ValueError("Plan contains non-finite waypoints")
    # Sample each straight 3D segment before applying nonlinear lens projection.
    samples = np.concatenate(
        [
            np.linspace(a, b, 32, endpoint=False)
            for a, b in zip(positions, positions[1:])
        ]
        + [positions[-1:]]
    )
    optical = rig_to_optical(samples, request.rig_to_camera[index])
    scaled = _scale_ftheta_intrinsics_to_resolution(
        spec.ftheta_param, (spec.resolution_h, spec.resolution_w), source.shape[:2]
    )
    source_pixels, source_valid = _FthetaCamera(scaled, source.shape[:2]).ray_to_pixel(
        optical
    )
    target_pixels, target_valid = project_rectified(optical, target)
    if not (source_valid.any() and target_valid.any()):
        raise ValueError("Plan has no visible samples; check calibration and pose axes")
    source_overlay = draw_plan(source, source_pixels, source_valid)
    target_overlay = draw_plan(rectified, target_pixels, target_valid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(source_overlay).save(
        args.output_dir / "rendered-overlay.jpg", quality=95
    )
    Image.fromarray(target_overlay).save(args.output_dir / "rectified-overlay.png")
    canvas = Image.new("RGB", (1600, 510), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for column, (array, label) in enumerate(
        [
            (source_overlay, "Rendered camera"),
            (target_overlay, "Rectified policy input"),
        ]
    ):
        view = Image.fromarray(array)
        view.thumbnail((800, 450), Image.Resampling.LANCZOS)
        canvas.paste(view, (column * 800 + (800 - view.width) // 2, 30))
        draw.text((column * 800 + 12, 8), label, fill="white")
    draw.text(
        (12, 487),
        "Green: predicted trajectory",
        fill="white",
    )
    canvas.save(args.output_dir / "comparison.jpg", quality=95)
    (args.output_dir / "projection.json").write_text(
        json.dumps(
            {
                "camera": args.camera,
                "camera_pose_axes": "sensor-to-rig FLU; inverted then converted to RDF",
                "policy_result_sha256": hashlib.sha256(
                    args.policy_result.read_bytes()
                ).hexdigest(),
                "image_sha256": image_hash,
                "source_frame_sha256": source_hash,
                "session_request_sha256": session_hash,
                "positions_m": positions.tolist(),
                "sample_count": len(samples),
                "source_visible_samples": int(source_valid.sum()),
                "rectified_visible_samples": int(target_valid.sum()),
                "scope": (
                    "Saved prediction, not executed motion; "
                    "no terrain or occlusion correction"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("Preview:", args.output_dir / "comparison.jpg", flush=True)
    print("PASS: saved plan projected; visual inspection pending", flush=True)


if __name__ == "__main__":
    main()
