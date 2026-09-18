# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Predict one trajectory from a saved rectified image with AlpaSim's VAM adapter.

Uses existing local weights. This is policy inference, not a simulator rollout.
Run through apptainer_exec.py --profile policy --gpu for timeout and telemetry.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True, help="Already rectified VaVAM camera image"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("straight", "left", "right"), default="straight"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in (args.image, args.checkpoint, args.tokenizer):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    output = args.output_dir
    if output is None:
        if "ALPASIM_RUN_DIR" not in os.environ:
            parser.error("Provide --output-dir or run through the Apptainer launcher")
        output = Path(os.environ["ALPASIM_RUN_DIR"]) / "policy"
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/driver/src"))

    print("Importing the branch's VaVAM adapter...", flush=True)
    import numpy as np
    import torch
    from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput
    from alpasim_driver.models.vam_model import VAMModel
    from PIL import Image

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This smoke test expects exactly one visible allocated GPU")
    device = torch.device("cuda:0")
    camera_id = "camera_front_wide_120fov"
    with Image.open(args.image) as image_file:
        image = np.array(image_file.convert("RGB"))
    if image.shape != (1080, 1920, 3):
        raise ValueError(
            f"Expected the rectified VaVAM preset image, got {image.shape}"
        )
    print("Loading cached checkpoint:", args.checkpoint.name, flush=True)
    started = time.monotonic()
    # Memory-stat APIs require the CUDA allocator to be initialized first.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    model = VAMModel(
        checkpoint_path=str(args.checkpoint),
        tokenizer_path=str(args.tokenizer),
        device=device,
        camera_ids=[camera_id],
        context_length=1,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.monotonic() - started
    print("PASS: actual AlpaSim VAMModel loaded; dtype:", model.DTYPE, flush=True)
    command = {
        "straight": DriveCommand.STRAIGHT,
        "left": DriveCommand.LEFT,
        "right": DriveCommand.RIGHT,
    }[args.command]
    prediction_input = PredictionInput(
        camera_images={camera_id: [CameraFrame(timestamp_us=0, image=image)]},
        command=command,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=[],
        inference_seed=args.seed,
        previous_plan=None,
        route=None,
    )
    # The one-frame adapter uses image + route command, not speed/history above.
    torch.manual_seed(args.seed)
    started = time.monotonic()
    prediction = model.predict(prediction_input)
    torch.cuda.synchronize(device)
    inference_seconds = time.monotonic() - started
    positions = prediction.selected_positions
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError(f"Unexpected trajectory shape: {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("Policy returned non-finite positions")
    np.savetxt(
        output / "trajectory.csv",
        positions,
        delimiter=",",
        header="x_m,y_m,z_m",
        comments="",
    )
    (output / "result.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint.name,
                "tokenizer": args.tokenizer.name,
                "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                "command": args.command,
                "seed": args.seed,
                "context_length": 1,
                "dtype": str(model.DTYPE),
                "torch": torch.__version__,
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "positions_m": positions.tolist(),
                "scope": "One policy prediction; no controller, physics or renderer feedback",
            },
            indent=2,
        )
        + "\n"
    )
    print("Waypoints:", len(positions), flush=True)
    print(
        "First/last position:",
        positions[0].tolist(),
        positions[-1].tolist(),
        flush=True,
    )
    print("Trajectory:", output / "trajectory.csv", flush=True)
    print(
        "PASS: AlpaSim VaVAM policy inference; closed-loop simulation still pending",
        flush=True,
    )


if __name__ == "__main__":
    main()
