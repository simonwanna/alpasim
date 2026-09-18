# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Export one native rollout ASL to timestamped frames and slow preview GIFs.

Overlays show logged predictions, without terrain or occlusion correction.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Frame:
    timestamp_us: int
    rig_pose: object
    image: object


@dataclass
class Plan:
    decision_timestamp_us: int
    timestamps_us: list[int]
    positions_local_m: list[list[float]]
    origin_pose: object = None


def positions(trajectory):
    return [[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in trajectory.poses]


def validate_trajectory(trajectory):
    times = [p.timestamp_us for p in trajectory.poses]
    if not times or len(times) != len(trajectory.poses):
        raise ValueError("Trajectory pose/timestamp counts must match and be nonempty")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    if not np.isfinite(positions(trajectory)).all():
        raise ValueError("Trajectory contains non-finite positions")
    return times


def pose_inverse_points(points, pose):
    """Transform local points into the rig frame using its logged local pose."""
    from overlay_plan import rig_to_optical

    optical = rig_to_optical(points, pose)
    # rig_to_optical additionally applies FLU -> RDF; undo that basis change.
    return np.column_stack((optical[:, 2], -optical[:, 0], -optical[:, 1]))


@dataclass
class RolloutExport:
    camera: str
    max_frames: int = 2000
    counts: Counter = field(default_factory=Counter)
    frames: list[Frame] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    session: object = None
    metadata: object = None
    pending_chunk: object = None
    pending_driver: object = None
    nonidentity_egomotion_error: bool = False
    image_bytes: int = 0
    session_id: str | None = None

    def add(self, entry):
        kind = entry.WhichOneof("log_entry")
        self.counts[kind] += 1
        if sum(self.counts.values()) > 100_000:
            raise ValueError(
                "Export is limited to a short rollout (100000 log entries)"
            )
        if kind == "rollout_metadata":
            if self.metadata is not None:
                raise ValueError("Expected exactly one rollout metadata entry")
            self.metadata = entry.rollout_metadata
        elif kind == "video_model_session_request":
            if self.session is not None:
                raise ValueError("Expected exactly one video-model session")
            self.session = entry.video_model_session_request
        elif kind == "video_model_chunk_request":
            if self.pending_chunk is not None:
                raise ValueError("Ambiguous overlapping video chunk requests")
            request = entry.video_model_chunk_request
            session_id = request.session_id.session_id
            if self.session_id is not None and session_id != self.session_id:
                raise ValueError("Video chunk requests belong to different sessions")
            self.session_id = session_id
            validate_trajectory(request.rig_trajectory)
            self.pending_chunk = request
        elif kind == "video_model_chunk_return":
            if self.pending_chunk is None:
                raise ValueError("Video chunk return has no matching request")
            outputs = [
                c
                for c in entry.video_model_chunk_return.camera_outputs
                if c.camera_logical_id == self.camera
            ]
            if len(outputs) != 1:
                raise ValueError(
                    "Expected exactly one selected camera output per chunk"
                )
            trajectory = self.pending_chunk.rig_trajectory
            images = outputs[0].rgb_frames
            if len(images) != len(trajectory.poses):
                raise ValueError("Video frame/request timestamp counts do not match")
            if len(self.frames) + len(images) > self.max_frames:
                raise ValueError("Export frame limit exceeded")
            for timed_pose, image in zip(trajectory.poses, images):
                timestamp, pose = timed_pose.timestamp_us, timed_pose.pose
                if self.frames and timestamp <= self.frames[-1].timestamp_us:
                    raise ValueError(
                        "Frame timestamps must be strictly increasing across chunks"
                    )
                self.image_bytes += len(image.data)
                if self.image_bytes > 256 * 1024 * 1024:
                    raise ValueError("Compressed image export limit exceeded (256 MiB)")
                self.frames.append(Frame(timestamp, pose, image))
            self.pending_chunk = None
        elif kind == "driver_request":
            if self.pending_driver is not None:
                raise ValueError("Ambiguous overlapping driver requests")
            request = entry.driver_request
            if (
                self.plans
                and request.time_now_us <= self.plans[-1].decision_timestamp_us
            ):
                raise ValueError(
                    "Driver decision timestamps must be strictly increasing"
                )
            self.pending_driver = request
        elif kind == "driver_return":
            if self.pending_driver is None:
                raise ValueError("Driver return has no matching request")
            trajectory = entry.driver_return.trajectory
            times = validate_trajectory(trajectory)
            self.plans.append(
                Plan(
                    self.pending_driver.time_now_us,
                    times,
                    positions(trajectory),
                    (
                        trajectory.poses[0].pose
                        if trajectory.poses[0].HasField("pose")
                        else None
                    ),
                )
            )
            self.pending_driver = None
        elif kind == "egomotion_estimate_error":
            pose = entry.egomotion_estimate_error.pose
            q = np.array([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w])
            self.nonidentity_egomotion_error |= not (
                np.allclose([pose.vec.x, pose.vec.y, pose.vec.z], 0, atol=1e-7)
                and np.allclose(q[:3], 0, atol=1e-7)
                and np.isclose(abs(q[3]), 1, atol=1e-7)
            )

    def finish(self):
        if self.pending_chunk is not None or self.pending_driver is not None:
            raise ValueError("ASL ends with an unmatched service request")
        if self.metadata is None or self.session is None or not self.frames:
            raise ValueError(
                "ASL must contain metadata, a video session, and generated frames"
            )
        if len(self.session.rig_to_camera) != len(self.session.camera_specs):
            raise ValueError("Camera specification/extrinsic counts do not match")
        indices = [
            i
            for i, s in enumerate(self.session.camera_specs)
            if s.logical_id == self.camera
        ]
        if len(indices) != 1:
            raise ValueError("Expected one selected camera specification")
        return indices[0]

    @property
    def handover_us(self):
        return (
            self.metadata.session_metadata.render_start_timestamp_us
            + self.metadata.force_gt_duration
        )

    def plan_at(self, timestamp_us):
        # Decision time is when the prediction was requested in simulated time.
        # time_query_us is its control target, not a wall-clock completion time.
        eligible = [p for p in self.plans if p.decision_timestamp_us <= timestamp_us]
        return eligible[-1] if eligible else None


def overlay_frame(rgb, frame, plan, spec, extrinsic, reference="world"):
    from alpasim_driver.rectification import (
        _FthetaCamera,
        _scale_ftheta_intrinsics_to_resolution,
    )
    from overlay_plan import draw_plan, rig_to_optical

    if reference not in ("world", "ego"):
        raise ValueError("Overlay reference must be world or ego")
    if plan is None:
        return rgb
    if reference == "ego" and plan.origin_pose is None:
        raise ValueError(
            "Ego-anchored overlay requires the logged prediction origin pose"
        )
    # Match DriverResponses.render_on_camera: retain the full world-space plan
    # between decisions. Removing waypoints as their timestamps pass makes the
    # near end jump; camera depth/FOV clipping determines what remains visible.
    points = np.array(plan.positions_local_m)
    if len(points) < 2:
        return rgb
    samples = np.concatenate(
        [np.linspace(a, b, 32, endpoint=False) for a, b in zip(points, points[1:])]
        + [points[-1:]]
    )
    # Ego display holds the prediction in its own rig frame between decisions.
    # The world overlay instead uses the current frame's rig pose.
    origin = frame.rig_pose if reference == "world" else plan.origin_pose
    rig = pose_inverse_points(samples, origin)
    optical = rig_to_optical(rig, extrinsic)
    scaled = _scale_ftheta_intrinsics_to_resolution(
        spec.ftheta_param, (spec.resolution_h, spec.resolution_w), rgb.shape[:2]
    )
    pixels, valid = _FthetaCamera(scaled, rgb.shape[:2]).ray_to_pixel(optical)
    return draw_plan(rgb, pixels, valid)


def caption_overlay(rgb, frame, plan, reference):
    """Label display semantics outside the camera image's pixel coordinates."""
    image = Image.fromarray(rgb)
    font_size = max(12, round(image.width / 60))
    font = ImageFont.load_default(size=font_size)
    canvas = Image.new(
        "RGB", (image.width, image.height + 2 * font_size + 16), (20, 20, 20)
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title = (
        "Ego-anchored plan display"
        if reference == "ego"
        else "World-referenced plan projection"
    )
    draw.text((8, image.height + 4), title, fill="white", font=font)
    detail = "No prediction available at this frame"
    if plan is not None:
        age_ms = (frame.timestamp_us - plan.decision_timestamp_us) / 1000
        detail = f"Decision: {plan.decision_timestamp_us} us | age: {age_ms:.1f} ms"
    draw.text((8, image.height + font_size + 9), detail, fill="white", font=font)
    return canvas


def pose_json(pose):
    if pose is None:
        return None
    return {
        "position_local_m": [pose.vec.x, pose.vec.y, pose.vec.z],
        "quaternion_xyzw": [pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w],
    }


def ego_motion_summary(frames):
    first, last = frames[0], frames[-1]
    first_position = np.array(pose_json(first.rig_pose)["position_local_m"])
    last_position = np.array(pose_json(last.rig_pose)["position_local_m"])
    delta = pose_inverse_points(last_position[None, :], first.rig_pose)[0]
    # Inverse-transform translated world basis points to recover each unit
    # quaternion's rotation matrix, using the same convention as the overlays.
    first_rotation = pose_inverse_points(np.eye(3) + first_position, first.rig_pose)
    last_rotation = pose_inverse_points(np.eye(3) + last_position, last.rig_pose)
    relative = first_rotation.T @ last_rotation
    return {
        "source": "Logged video-model request rig poses",
        "first_timestamp_us": first.timestamp_us,
        "last_timestamp_us": last.timestamp_us,
        "first_position_local_m": first_position.tolist(),
        "last_position_local_m": last_position.tolist(),
        "delta_in_initial_rig_m": delta.tolist(),
        "forward_delta_m": float(delta[0]),
        "lateral_delta_m": float(delta[1]),
        "yaw_change_deg": float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))),
    }


def save_gif(paths, destination, fps):
    def previews():
        for path in paths:
            with Image.open(path) as image:
                image.thumbnail((640, 360), Image.Resampling.LANCZOS)
                yield image.convert("P", palette=Image.Palette.ADAPTIVE)

    iterator = previews()
    first = next(iterator)
    first.save(
        destination,
        save_all=True,
        append_images=iterator,
        duration=round(1000 / fps),
        loop=0,
    )


def require_closed_loop_activity(run):
    if run.session.debug_options.skip_video_generation:
        raise ValueError("Video generation was disabled in the logged session")
    post_warmup = [p for p in run.plans if p.decision_timestamp_us >= run.handover_us]
    if not post_warmup:
        raise ValueError("No post-warmup policy decisions in rollout")
    if not any(
        f.timestamp_us > post_warmup[0].decision_timestamp_us for f in run.frames
    ):
        raise ValueError("No generated frames after a post-warmup policy decision")
    for service in ("driver", "controller", "physics"):
        requested = run.counts[f"{service}_request"]
        returned = run.counts[f"{service}_return"]
        if requested == 0 or returned != requested:
            raise ValueError(f"Expected completed {service} request/return activity")


def export_artifacts(
    run,
    output,
    overlay=False,
    fps=10,
    require_closed_loop=False,
    overlay_reference="world",
):
    from alpasim_grpc.v0 import video_model_pb2

    index = run.finish()
    if require_closed_loop:
        require_closed_loop_activity(run)
    spec = run.session.camera_specs[index]
    if overlay_reference not in ("world", "ego", "both"):
        raise ValueError("Overlay reference must be world, ego, or both")
    references = (
        ("world", "ego") if overlay_reference == "both" else (overlay_reference,)
    )
    if not overlay:
        references = ()
    if "ego" in references and any(plan.origin_pose is None for plan in run.plans):
        raise ValueError(
            "Ego-anchored overlay requires the logged prediction origin pose"
        )
    if overlay:
        if run.nonidentity_egomotion_error:
            raise ValueError(
                "Overlay requires zero egomotion error; raw plans use estimated local coordinates"
            )
        if spec.WhichOneof("camera_param") != "ftheta_param":
            raise ValueError("Overlay currently requires an f-theta camera")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    overlay_dirs = {"world": "overlay-frames", "ego": "ego-overlay-frames"}
    overlay_gifs = {
        "world": "plan-overlay-slow.gif",
        "ego": "ego-plan-overlay-slow.gif",
    }
    for reference in references:
        (output / overlay_dirs[reference]).mkdir()
    rows = []
    paths = []
    overlay_paths = {reference: [] for reference in references}
    expected_shape = None
    for i, frame in enumerate(run.frames):
        if frame.image.format not in (video_model_pb2.JPEG, video_model_pb2.PNG):
            raise ValueError("Export supports JPEG and PNG frame payloads")
        with Image.open(io.BytesIO(frame.image.data)) as image:
            if image.width * image.height > 8_000_000:
                raise ValueError("Decoded frame exceeds 8 million pixels")
            rgb = np.array(image.convert("RGB"))
        if expected_shape is not None and rgb.shape != expected_shape:
            raise ValueError("Frame resolution changed within the selected camera")
        expected_shape = rgb.shape
        path = output / "frames" / f"{i:05d}.jpg"
        Image.fromarray(rgb).save(path, quality=95)
        paths.append(path)
        plan = run.plan_at(frame.timestamp_us)
        overlay_pixels = {"world": 0, "ego": 0}
        for reference in references:
            rendered = overlay_frame(
                rgb, frame, plan, spec, run.session.rig_to_camera[index], reference
            )
            overlay_pixels[reference] = int(
                np.count_nonzero(np.any(rendered != rgb, axis=-1))
            )
            overlay_path = output / overlay_dirs[reference] / path.name
            caption_overlay(rendered, frame, plan, reference).save(
                overlay_path, quality=95
            )
            overlay_paths[reference].append(overlay_path)
        rows.append(
            {
                "file": str(path.relative_to(output)),
                "timestamp_us": frame.timestamp_us,
                "phase": (
                    "recorded_warmup"
                    if frame.timestamp_us < run.handover_us
                    else "closed_loop"
                ),
                "plan_decision_timestamp_us": (
                    None if plan is None else plan.decision_timestamp_us
                ),
                "plan_age_us": (
                    None
                    if plan is None
                    else frame.timestamp_us - plan.decision_timestamp_us
                ),
                "overlay_pixels": overlay_pixels["world"],
                "ego_overlay_pixels": overlay_pixels["ego"],
            }
        )
    save_gif(paths, output / "preview-slow.gif", fps)
    for reference in references:
        save_gif(overlay_paths[reference], output / overlay_gifs[reference], fps)
    (output / "predicted-plans.json").write_text(
        json.dumps(
            [
                {
                    "decision_timestamp_us": p.decision_timestamp_us,
                    "timestamps_us": p.timestamps_us,
                    "positions_local_m": p.positions_local_m,
                    "origin_pose": pose_json(p.origin_pose),
                    "origin_timestamp_us": (
                        p.timestamps_us[0] if p.origin_pose is not None else None
                    ),
                }
                for p in run.plans
            ],
            indent=2,
        )
        + "\n"
    )
    summary = {
        "camera": run.camera,
        "generated_frames": len(run.frames),
        "warmup_frames": sum(r["phase"] == "recorded_warmup" for r in rows),
        "closed_loop_frames": sum(r["phase"] == "closed_loop" for r in rows),
        "handover_timestamp_us": run.handover_us,
        "post_warmup_driver_requests": sum(
            p.decision_timestamp_us >= run.handover_us for p in run.plans
        ),
        "counts": {
            **{
                f"{service}_{direction}": 0
                for service in ("driver", "controller", "physics")
                for direction in ("request", "return")
            },
            **dict(run.counts),
        },
        "closed_loop_activity_required": require_closed_loop,
        "preview_fps": fps,
        "overlay_references": list(references),
        "ego_motion": ego_motion_summary(run.frames),
        "frames_with_visible_ego_overlay": sum(
            r["ego_overlay_pixels"] > 0 for r in rows
        ),
        "frames_with_visible_overlay": sum(r["overlay_pixels"] > 0 for r in rows),
        "overlay_scope": (
            "Latest prediction by simulated decision time; not executed motion; "
            "no terrain or occlusion correction"
        ),
        "frames": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2),
        flush=True,
    )
    print("Export:", output, flush=True)
    return summary


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument(
        "--overlay-reference", choices=("world", "ego", "both"), default="world"
    )
    parser.add_argument("--require-closed-loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 2000 or not 1 <= args.fps <= 30:
        parser.error("--max-frames must be 1..2000; --fps must be 1..30")
    from alpasim_utils.logs import async_read_pb_log

    run = RolloutExport(args.camera, max_frames=args.max_frames)
    async for entry in async_read_pb_log(str(args.log), raise_on_malformed=True):
        run.add(entry)
    export_artifacts(
        run,
        args.output_dir,
        overlay=args.overlay,
        fps=args.fps,
        require_closed_loop=args.require_closed_loop,
        overlay_reference=args.overlay_reference,
    )


if __name__ == "__main__":
    asyncio.run(main())
