# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Audit logged policy, controller handoff and motion without running models."""

import argparse
import asyncio
import json
import pickletools
from collections import Counter
from pathlib import Path

import numpy as np
from export_loop import (
    RolloutExport,
    ego_motion_summary,
    pose_inverse_points,
    pose_json,
    positions,
    validate_trajectory,
)


def command_name(payload):
    """Read only a literal command name; never execute or unpickle debug data."""
    if len(payload) > 1_000_000:
        return "unknown"
    try:
        ops = list(pickletools.genops(payload))
    except Exception:
        return "unknown"
    unsafe = {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "BUILD",
        "OBJ",
        "INST",
        "NEWOBJ",
        "NEWOBJ_EX",
        "PERSID",
        "BINPERSID",
        "EXT1",
        "EXT2",
        "EXT4",
    }
    if any(op.name in unsafe for op, _, _ in ops):
        return "unknown"
    literals = {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}
    ignored = {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}
    found = []
    for index, (op, value, _) in enumerate(ops):
        if op.name in literals and value == "command_name":
            following = next(
                (item for item in ops[index + 1 :] if item[0].name not in ignored), None
            )
            if following is None or following[0].name not in literals:
                return "unknown"
            found.append(following[1])
    return (
        found[0]
        if len(found) == 1 and found[0] in {"LEFT", "RIGHT", "STRAIGHT", "UNKNOWN"}
        else "unknown"
    )


def yaw(pose):
    # Reuse the exporter's finite-position and unit-quaternion validation.
    pose_inverse_points(np.zeros((1, 3)), pose)
    q = pose.quat
    return float(
        np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    )


def wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def interpolate(trajectory, timestamp):
    """Linear position and unwrapped-yaw diagnostic; never extrapolate."""
    times = validate_trajectory(trajectory)
    angles = np.unwrap([yaw(p.pose) for p in trajectory.poses])
    if timestamp < times[0] or timestamp > times[-1]:
        return None
    xyz = np.asarray(positions(trajectory))
    point = [float(np.interp(timestamp, times, xyz[:, axis])) for axis in range(3)]
    speed = None
    if len(times) > 1:
        i = min(
            max(int(np.searchsorted(times, timestamp, side="right")) - 1, 0),
            len(times) - 2,
        )
        speed = float(
            np.linalg.norm(xyz[i + 1] - xyz[i]) * 1e6 / (times[i + 1] - times[i])
        )
    return {
        "position_rig_m": point,
        "yaw_rad": float(np.interp(timestamp, times, angles)),
        "segment_speed_m_s": speed,
    }


class Audit:
    def __init__(self):
        self.export = RolloutExport("camera_front_wide_120fov")
        self.session = None
        self.pending = None
        self.drivers = []
        self.controllers = []
        self.motion = []
        self.driver_request = None

    def add(self, entry):
        kind = entry.WhichOneof("log_entry")
        if kind in ("driver_request", "controller_request"):
            session = getattr(entry, kind).session_uuid
            if not session or (self.session is not None and self.session != session):
                raise ValueError(
                    "Audit requires one nonempty driver/controller session"
                )
            self.session = session
        self.export.add(entry)
        if kind == "video_model_chunk_request":
            for timed_pose in entry.video_model_chunk_request.rig_trajectory.poses:
                yaw(timed_pose.pose)
        if kind == "driver_request":
            self.driver_request = entry.driver_request
        if kind == "driver_return":
            plan = self.export.plans[-1]
            for pose in entry.driver_return.trajectory.poses:
                yaw(pose.pose)
            self.drivers.append(
                {
                    "decision_timestamp_us": plan.decision_timestamp_us,
                    "query_timestamp_us": self.driver_request.time_query_us,
                    "command_name": command_name(
                        entry.driver_return.debug_info.unstructured_debug_info
                    ),
                    "timestamps_us": plan.timestamps_us,
                    "positions_local_m": plan.positions_local_m,
                    "endpoint_in_prediction_rig_m": pose_inverse_points(
                        np.asarray(plan.positions_local_m[-1:]), plan.origin_pose
                    )[0].tolist(),
                }
            )
        elif kind == "controller_request":
            if self.pending is not None:
                raise ValueError("Overlapping controller requests")
            request = entry.controller_request
            yaw(request.state.pose)
            validate_trajectory(request.planned_trajectory_in_rig)
            for pose in request.planned_trajectory_in_rig.poses:
                yaw(pose.pose)
            if request.future_time_us <= request.state.timestamp_us:
                raise ValueError("Controller target must follow state timestamp")
            if (
                self.controllers
                and request.state.timestamp_us
                < self.controllers[-1]["target_timestamp_us"]
            ):
                raise ValueError("Controller requests must be chronological")
            self.pending = (request, len(self.export.plans))
        elif kind == "controller_return":
            if self.pending is None:
                raise ValueError("Controller return has no matching request")
            request, available = self.pending
            self.controllers.append(
                self.compare(request, entry.controller_return, available)
            )
            self.pending = None

    def compare(self, request, response, available):
        times = validate_trajectory(request.planned_trajectory_in_rig)
        post = request.state.timestamp_us >= self.export.handover_us
        candidates = [
            p
            for p in self.export.plans[:available]
            if p.decision_timestamp_us <= request.state.timestamp_us
            and p.timestamps_us[0] <= times[0]
            and p.timestamps_us[-1] >= times[-1]
        ]
        plan = candidates[-1] if candidates else None
        error = None
        if plan is not None:
            xyz = np.asarray(plan.positions_local_m)
            aligned = np.column_stack(
                [
                    np.interp(times, plan.timestamps_us, xyz[:, axis])
                    for axis in range(3)
                ]
            )
            expected = pose_inverse_points(aligned, request.state.pose)
            error = float(
                np.max(
                    np.linalg.norm(
                        expected
                        - np.asarray(positions(request.planned_trajectory_in_rig)),
                        axis=1,
                    )
                )
            )
        states = list(response.states)
        if not states:
            raise ValueError("Controller response has no propagated states")
        previous = request.state.timestamp_us
        samples = []
        for state in states:
            if not previous < state.timestamp_us <= request.future_time_us:
                raise ValueError("Controller response timestamps do not match request")
            previous = state.timestamp_us
            world_yaw = yaw(state.pose_local_to_rig)
            v = state.dynamic_state.linear_velocity
            speed = float(np.linalg.norm([v.x, v.y, v.z]))
            if not np.isfinite(speed):
                raise ValueError("Nonfinite controller speed")
            point = pose_inverse_points(
                np.asarray(
                    [
                        [
                            state.pose_local_to_rig.vec.x,
                            state.pose_local_to_rig.vec.y,
                            state.pose_local_to_rig.vec.z,
                        ]
                    ]
                ),
                request.state.pose,
            )[0]
            desired = interpolate(request.planned_trajectory_in_rig, state.timestamp_us)
            sample = {
                "timestamp_us": state.timestamp_us,
                "pose": pose_json(state.pose_local_to_rig),
                "position_in_request_rig_m": point.tolist(),
                "speed_m_s": speed,
                "yaw_rad": world_yaw,
                "requested": desired,
                "tracking_position_error_m": None,
                "tracking_yaw_error_deg": None,
            }
            if desired is not None:
                sample["tracking_position_error_m"] = float(
                    np.linalg.norm(point - desired["position_rig_m"])
                )
                sample["tracking_yaw_error_deg"] = float(
                    np.degrees(
                        wrap(world_yaw - yaw(request.state.pose) - desired["yaw_rad"])
                    )
                )
            samples.append(sample)
            self.motion.append(sample)
        if previous != request.future_time_us:
            raise ValueError("Controller response does not reach requested target")
        return {
            "state_timestamp_us": request.state.timestamp_us,
            "request_pose": pose_json(request.state.pose),
            "requested_plan_timestamps_us": times,
            "requested_plan_positions_rig_m": positions(
                request.planned_trajectory_in_rig
            ),
            "target_timestamp_us": request.future_time_us,
            "post_warmup": post,
            "coerce_dynamic_state": request.coerce_dynamic_state,
            "matched_driver_timestamp_us": (
                None if plan is None else plan.decision_timestamp_us
            ),
            "handoff_max_position_error_m": error,
            "handoff_status": (
                "warmup"
                if not post
                else (
                    "unmatched"
                    if error is None
                    else "mismatch" if error > 1e-3 else "match"
                )
            ),
            "handoff_mismatch": post and error is not None and error > 1e-3,
            "target": samples[-1],
            "propagated_states": samples,
        }

    def finish(self):
        self.export.finish()
        if self.pending is not None:
            raise ValueError("ASL ends with an unmatched controller request")
        if not self.drivers or not self.controllers:
            raise ValueError("Audit requires completed driver and controller RPCs")
        if self.export.nonidentity_egomotion_error:
            raise ValueError(
                "Handoff comparison requires zero egomotion estimate error"
            )
        post = [c for c in self.controllers if c["post_warmup"]]
        errors = [
            c["handoff_max_position_error_m"]
            for c in post
            if c["handoff_max_position_error_m"] is not None
        ]
        tracking = [
            c["target"]["tracking_position_error_m"]
            for c in post
            if c["target"]["tracking_position_error_m"] is not None
        ]
        summary = {
            "session": self.session,
            "driver_calls": len(self.drivers),
            "command_counts": dict(Counter(d["command_name"] for d in self.drivers)),
            "controller_calls": len(self.controllers),
            "post_warmup_controller_calls": len(post),
            "handoff_mismatches": sum(c["handoff_mismatch"] for c in post),
            "handoff_unmatched": sum(c["handoff_status"] == "unmatched" for c in post),
            "max_handoff_position_error_m": max(errors, default=None),
            "max_target_tracking_position_error_m": max(tracking, default=None),
            "ego_motion": ego_motion_summary(self.export.frames),
            "controller_first_state": self.motion[0],
            "controller_last_state": self.motion[-1],
            "controller_motion": {
                "first_timestamp_us": self.motion[0]["timestamp_us"],
                "last_timestamp_us": self.motion[-1]["timestamp_us"],
                "first_speed_m_s": self.motion[0]["speed_m_s"],
                "last_speed_m_s": self.motion[-1]["speed_m_s"],
                "yaw_change_deg": float(
                    np.degrees(
                        wrap(self.motion[-1]["yaw_rad"] - self.motion[0]["yaw_rad"])
                    )
                ),
            },
            "scope": (
                "Logged handoff and linear-interpolation tracking diagnostics; "
                "no extrapolation, collision or lane metrics"
            ),
        }
        return {
            "summary": summary,
            "driver_samples": self.drivers,
            "controller_samples": self.controllers,
        }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    from alpasim_utils.logs import async_read_pb_log

    audit = Audit()
    async for entry in async_read_pb_log(str(args.log), raise_on_malformed=True):
        audit.add(entry)
    report = audit.finish()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "samples.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    summary = report["summary"]
    compact = {
        k: v
        for k, v in summary.items()
        if k not in {"controller_first_state", "controller_last_state", "ego_motion"}
    }
    compact["ego_motion"] = {
        key: summary["ego_motion"][key]
        for key in ("forward_delta_m", "lateral_delta_m", "yaw_change_deg")
    }
    decisions = [
        d
        for d in report["driver_samples"]
        if d["decision_timestamp_us"] >= audit.export.handover_us
    ]
    selected = np.linspace(0, len(decisions) - 1, min(5, len(decisions)), dtype=int)
    compact["selected_driver_samples"] = [
        {
            key: decisions[i][key]
            for key in (
                "decision_timestamp_us",
                "command_name",
                "endpoint_in_prediction_rig_m",
            )
        }
        for i in selected
    ]
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
    print("Samples:", args.output_dir / "samples.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
