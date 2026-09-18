# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Rectify one saved video-model frame using this checkout's driver code.

CPU only; no model loading, service startup, installation or downloads.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-request", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "src/wizard/configs/driver/vavam_video_model.yaml",
    )
    parser.add_argument(
        "--grpc-wheel", type=Path, help="Use an already built alpasim-grpc wheel"
    )
    args = parser.parse_args()
    if args.grpc_wheel:
        if not args.grpc_wheel.is_file():
            parser.error("--grpc-wheel must name an existing wheel")
        sys.path.insert(0, str(args.grpc_wheel.resolve()))
    sys.path.insert(0, str(root / "src/driver/src"))

    import cv2
    import numpy as np
    import yaml
    from alpasim_driver.rectification import build_ftheta_rectifier_for_resolution
    from alpasim_driver.schema import RectificationTargetConfig
    from alpasim_grpc.v0 import sensorsim_pb2, video_model_pb2
    from PIL import Image, ImageDraw

    cv2.setNumThreads(1)
    request = video_model_pb2.SessionRequest.FromString(
        args.session_request.read_bytes()
    )
    spec = next(c for c in request.camera_specs if c.logical_id == args.camera)
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id=args.camera
    )
    camera.intrinsics.CopyFrom(spec)
    config = yaml.safe_load(args.config.read_text())["driver"]["rectification"][
        args.camera
    ]
    target = RectificationTargetConfig(**config)
    with Image.open(args.frame) as image_file:
        image = np.array(image_file.convert("RGB"))
    print("Input shape:", image.shape, flush=True)
    rectifier = build_ftheta_rectifier_for_resolution(camera, target, image.shape[:2])
    rectified = rectifier.rectify(image)
    expected_shape = (*target.resolution_hw, 3)
    if rectified.shape != expected_shape or rectified.dtype != np.uint8:
        raise ValueError(f"Unexpected output: {rectified.shape}, {rectified.dtype}")
    support = rectifier.rectify(np.full_like(image, 255))
    fraction = float(np.mean(np.all(support == 255, axis=-1)))
    if fraction == 0:
        raise ValueError("No fully supported output pixels")

    # Refuse to overwrite previous results. The caller selects project storage.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(args.output_dir / "rendered-input.png")
    Image.fromarray(rectified).save(args.output_dir / "rectified.png")
    Image.fromarray(support).save(args.output_dir / "source-support.png")
    canvas = Image.new("RGB", (1280, 400), (35, 35, 35))
    draw = ImageDraw.Draw(canvas)
    views = ((image, "Rendered input"), (rectified, "Rectified VaVAM camera view"))
    for column, (array, label) in enumerate(views):
        view = Image.fromarray(array)
        view.thumbnail((640, 360), Image.Resampling.LANCZOS)
        canvas.paste(
            view,
            (column * 640 + (640 - view.width) // 2, 32 + (360 - view.height) // 2),
        )
        draw.text((column * 640 + 12, 10), label, fill="white")
    canvas.save(args.output_dir / "comparison.jpg", quality=95)
    source = root / "src/driver/src/alpasim_driver/rectification.py"
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "camera": args.camera,
                "frame_sha256": hashlib.sha256(args.frame.read_bytes()).hexdigest(),
                "session_request_sha256": hashlib.sha256(
                    args.session_request.read_bytes()
                ).hexdigest(),
                "rectifier_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_shape": list(image.shape),
                "target_config": config,
                "fully_supported_fraction": fraction,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Fully supported output pixels: {fraction:.2%}", flush=True)
    print("Preview:", args.output_dir / "comparison.jpg", flush=True)
    print(
        "PASS: frame rectified; inspect the preview before policy inference", flush=True
    )


if __name__ == "__main__":
    main()
