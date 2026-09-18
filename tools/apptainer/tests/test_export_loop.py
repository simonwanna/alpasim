# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2 as common
from alpasim_grpc.v0 import egodriver_pb2 as driver
from alpasim_grpc.v0 import logging_pb2 as logging
from alpasim_grpc.v0 import sensorsim_pb2 as sensorsim
from alpasim_grpc.v0 import video_model_pb2 as video
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_loop", TOOLS / "export_loop.py")
export = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = export
SPEC.loader.exec_module(export)
CAMERA = "camera_front_wide_120fov"


@pytest.fixture(autouse=True)
def tool_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))


def pose(x=0, y=0, z=0, yaw=0):
    return common.Pose(
        vec=common.Vec3(x=x, y=y, z=z),
        quat=common.Quat(z=np.sin(yaw / 2), w=np.cos(yaw / 2)),
    )


def trajectory(times, points=None):
    if points is None:
        points = [(0, 0, 0)] * len(times)
    return common.Trajectory(
        poses=[
            common.PoseAtTime(timestamp_us=t, pose=pose(*p))
            for t, p in zip(times, points)
        ]
    )


def image():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (50, 90, 120)).save(buffer, format="PNG")
    return video.Image(format=video.PNG, data=buffer.getvalue())


def start():
    run = export.RolloutExport(CAMERA)
    run.add(
        logging.LogEntry(
            rollout_metadata=logging.RolloutMetadata(
                session_metadata=logging.RolloutMetadata.SessionMetadata(
                    render_start_timestamp_us=100
                ),
                force_gt_duration=100,
            )
        )
    )
    # Plain export only needs the session camera and mount, not a lens model.
    run.add(
        logging.LogEntry(
            video_model_session_request=video.SessionRequest(
                camera_specs=[sensorsim.CameraSpec(logical_id=CAMERA)],
                rig_to_camera=[pose()],
            )
        )
    )
    return run


def chunk_request(times):
    return logging.LogEntry(
        video_model_chunk_request=video.VideoChunkRequest(
            session_id=video.SessionId(session_id="test"),
            rig_trajectory=trajectory(times),
        )
    )


def chunk_return(count):
    return logging.LogEntry(
        video_model_chunk_return=video.VideoChunkReturn(
            camera_outputs=[
                video.CameraOutput(
                    camera_logical_id=CAMERA, rgb_frames=[image()] * count
                )
            ]
        )
    )


def add_plan(run, now, times=(200, 300)):
    run.add(
        logging.LogEntry(
            driver_request=driver.DriveRequest(time_now_us=now, time_query_us=now + 100)
        )
    )
    run.add(
        logging.LogEntry(
            driver_return=driver.DriveResponse(
                trajectory=trajectory(times, [(10, 0, 0), (20, 0, 0)])
            )
        )
    )


def test_pairing_and_future_prediction_selection():
    run = start()
    run.add(chunk_request([100, 200]))
    add_plan(run, 150)
    run.add(chunk_return(2))
    add_plan(run, 250, (300, 400))
    run.finish()
    assert run.plan_at(149) is None
    assert run.plan_at(150).decision_timestamp_us == 150
    assert run.plan_at(249).decision_timestamp_us == 150
    assert run.plan_at(250).decision_timestamp_us == 250
    assert [f.timestamp_us for f in run.frames] == [100, 200]


@pytest.mark.parametrize(
    "entries,match",
    [
        ([chunk_return(1)], "no matching request"),
        ([chunk_request([100]), chunk_request([200])], "overlapping"),
        ([chunk_request([100, 200]), chunk_return(1)], "counts"),
        ([chunk_request([100, 100])], "strictly increasing"),
        (
            [
                chunk_request([100]),
                chunk_return(1),
                chunk_request([100]),
                chunk_return(1),
            ],
            "across chunks",
        ),
    ],
)
def test_rejects_ambiguous_or_malformed_render_sequence(entries, match):
    run = start()
    with pytest.raises(ValueError, match=match):
        for entry in entries:
            run.add(entry)


def test_rejects_incomplete_driver_response_and_frame_limit():
    run = start()
    run.add(chunk_request([100]))
    run.add(chunk_return(1))
    run.add(logging.LogEntry(driver_request=driver.DriveRequest(time_now_us=100)))
    with pytest.raises(ValueError, match="unmatched"):
        run.finish()
    run = export.RolloutExport(CAMERA, max_frames=1)
    run.add(chunk_request([100, 200]))
    with pytest.raises(ValueError, match="frame limit"):
        run.add(chunk_return(2))


def test_local_plan_to_rotated_rig_then_optical_axes():
    from overlay_plan import rig_to_optical

    # Rig at local (10,20,0), heading +Y. Point is forward and left in rig.
    rig = export.pose_inverse_points(
        np.array([[8, 30, 0]]), pose(10, 20, yaw=np.pi / 2)
    )
    np.testing.assert_allclose(rig, [[10, 2, 0]], atol=1e-6)
    optical = rig_to_optical(rig, pose(x=1, z=2))
    np.testing.assert_allclose(optical, [[-2, 2, 9]], atol=1e-6)


def test_export_artifacts_preserves_warmup_and_plan_timestamps(tmp_path):
    import json

    run = start()
    run.add(chunk_request([100, 200, 300]))
    run.add(chunk_return(3))
    add_plan(run, 200, (300, 400))
    out = tmp_path / "export"
    summary = export.export_artifacts(run, out)
    assert summary["generated_frames"] == 3
    assert summary["warmup_frames"] == 1
    assert summary["closed_loop_frames"] == 2
    assert [r["plan_decision_timestamp_us"] for r in summary["frames"]] == [
        None,
        200,
        200,
    ]
    assert len(list((out / "frames").glob("*.jpg"))) == 3
    assert (out / "preview-slow.gif").is_file()
    plans = json.loads((out / "predicted-plans.json").read_text())
    assert plans[0]["timestamps_us"] == [300, 400]
    with pytest.raises(FileExistsError):
        export.export_artifacts(run, out)


def test_overlay_refuses_noisy_local_predictions(tmp_path):
    run = start()
    run.add(chunk_request([100]))
    run.add(chunk_return(1))
    run.add(
        logging.LogEntry(
            egomotion_estimate_error=common.PoseAtTime(timestamp_us=100, pose=pose(x=1))
        )
    )
    with pytest.raises(ValueError, match="zero egomotion error"):
        export.export_artifacts(run, tmp_path / "export", overlay=True)


def test_no_plan_leaves_frame_unchanged():
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    assert export.overlay_frame(rgb, None, None, None, None) is rgb


def test_plan_keeps_near_segment_between_decisions():
    spec = sensorsim.CameraSpec(logical_id=CAMERA, resolution_h=120, resolution_w=240)
    lens = spec.ftheta_param
    lens.principal_point_x = 120
    lens.principal_point_y = 60
    lens.angle_to_pixeldist_poly.extend([0, 80])
    lens.max_angle = 1.4
    rgb = np.zeros((120, 240, 3), dtype=np.uint8)
    plan = export.Plan(100, [100, 200, 300], [[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    early = export.overlay_frame(
        rgb, export.Frame(100, pose(), None), plan, spec, pose(z=2)
    )
    later = export.overlay_frame(
        rgb, export.Frame(250, pose(), None), plan, spec, pose(z=2)
    )
    assert np.any(early != rgb)
    np.testing.assert_array_equal(early, later)


def test_strict_gate_rejects_all_warmup_and_missing_physics(tmp_path):
    run = start()
    run.add(chunk_request([100, 200, 300]))
    run.add(chunk_return(3))
    add_plan(run, 150)
    with pytest.raises(ValueError, match="post-warmup"):
        export.export_artifacts(run, tmp_path / "warmup", require_closed_loop=True)
    add_plan(run, 200, (300, 400))
    with pytest.raises(ValueError, match="controller"):
        export.require_closed_loop_activity(run)
    run.add(logging.LogEntry(controller_request={}))
    run.add(logging.LogEntry(controller_return={}))
    with pytest.raises(ValueError, match="physics"):
        export.require_closed_loop_activity(run)
    run.add(logging.LogEntry(physics_request={}))
    run.add(logging.LogEntry(physics_return={}))
    export.require_closed_loop_activity(run)


def test_overlay_artifacts_do_not_apply_future_plan_to_earlier_frame(tmp_path):
    run = start()
    spec = run.session.camera_specs[0]
    spec.resolution_h, spec.resolution_w = 32, 64
    spec.ftheta_param.principal_point_x = 32
    spec.ftheta_param.principal_point_y = 16
    spec.ftheta_param.reference_poly = sensorsim.FthetaCameraParam.PIXELDIST_TO_ANGLE
    spec.ftheta_param.pixeldist_to_angle_poly.extend([0, 0.02])
    run.session.rig_to_camera[0].CopyFrom(pose(z=1.5))
    run.add(chunk_request([100, 200]))
    run.add(chunk_return(2))
    add_plan(run, 200, (300, 400))
    out = tmp_path / "overlay"
    export.export_artifacts(run, out, overlay=True)
    with Image.open(out / "overlay-frames/00000.jpg") as first:
        a = np.array(first)
    with Image.open(out / "overlay-frames/00001.jpg") as second:
        b = np.array(second)
    assert not (a[:, :, 1] > 200).any()
    assert (b[:, :, 1] > 200).any()
    assert (out / "plan-overlay-slow.gif").is_file()


def test_strict_gate_rejects_disabled_video_generation():
    run = start()
    run.session.debug_options.skip_video_generation = True
    with pytest.raises(ValueError, match="Video generation was disabled"):
        export.require_closed_loop_activity(run)
