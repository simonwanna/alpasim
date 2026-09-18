# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pickle
from pathlib import Path

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2 as common
from alpasim_grpc.v0 import controller_pb2 as controller
from alpasim_grpc.v0 import egodriver_pb2 as driver
from alpasim_grpc.v0 import logging_pb2 as logging
from alpasim_grpc.v0 import sensorsim_pb2 as sensorsim
from alpasim_grpc.v0 import video_model_pb2 as video

TOOLS = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit_module(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    import audit_loop

    return audit_loop


def pose(x=0, y=0, yaw=0):
    return common.Pose(
        vec=common.Vec3(x=x, y=y),
        quat=common.Quat(z=np.sin(yaw / 2), w=np.cos(yaw / 2)),
    )


def trajectory(points, times=(1000000, 2000000), yaw=0):
    return common.Trajectory(
        poses=[
            common.PoseAtTime(timestamp_us=t, pose=pose(x, y, yaw))
            for t, (x, y) in zip(times, points)
        ]
    )


def fixture(module, changed_plan=0, tracking_offset=0, controller_times=None):
    run = module.Audit()
    run.add(
        logging.LogEntry(
            rollout_metadata=logging.RolloutMetadata(
                session_metadata=logging.RolloutMetadata.SessionMetadata(
                    render_start_timestamp_us=1000000
                )
            )
        )
    )
    camera = "camera_front_wide_120fov"
    run.add(
        logging.LogEntry(
            video_model_session_request=video.SessionRequest(
                camera_specs=[sensorsim.CameraSpec(logical_id=camera)],
                rig_to_camera=[pose()],
            )
        )
    )
    run.add(
        logging.LogEntry(
            video_model_chunk_request=video.VideoChunkRequest(
                session_id=video.SessionId(session_id="render-session"),
                rig_trajectory=trajectory([(10, 20), (10, 30)], yaw=np.pi / 2),
            )
        )
    )
    run.add(
        logging.LogEntry(
            video_model_chunk_return=video.VideoChunkReturn(
                camera_outputs=[
                    video.CameraOutput(
                        camera_logical_id=camera,
                        rgb_frames=[video.Image(), video.Image()],
                    )
                ]
            )
        )
    )
    run.add(
        logging.LogEntry(
            driver_request=driver.DriveRequest(
                session_uuid="ego-session", time_now_us=1000000, time_query_us=1500000
            )
        )
    )
    run.add(
        logging.LogEntry(
            driver_return=driver.DriveResponse(
                trajectory=trajectory([(10, 20), (10, 30)], yaw=np.pi / 2),
                debug_info=driver.DriveResponse.DebugInfo(
                    unstructured_debug_info=pickle.dumps(
                        {"command": 0, "command_name": "LEFT"}
                    )
                ),
            )
        )
    )
    request = controller.RunControllerAndVehicleModelRequest(
        session_uuid="ego-session",
        state=common.StateAtTime(timestamp_us=1000000, pose=pose(10, 20, np.pi / 2)),
        future_time_us=1500000,
        planned_trajectory_in_rig=trajectory(
            [(0, 0), (10, changed_plan)], times=controller_times or (1000000, 2000000)
        ),
    )
    run.add(logging.LogEntry(controller_request=request))
    response = controller.RunControllerAndVehicleModelResponse(
        states=[
            controller.RunControllerAndVehicleModelResponse.PropagatedState(
                timestamp_us=1500000,
                pose_local_to_rig=pose(10 - tracking_offset, 25, np.pi / 2),
                dynamic_state=common.DynamicState(linear_velocity=common.Vec3(x=10)),
            )
        ]
    )
    return run, response


def complete(run, response):
    run.add(logging.LogEntry(controller_return=response))
    return run.finish()


def test_nonidentity_handoff_and_tracking(audit_module):
    report = complete(*fixture(audit_module))
    sample = report["controller_samples"][0]
    assert report["summary"]["command_counts"] == {"LEFT": 1}
    assert sample["handoff_max_position_error_m"] < 1e-5
    assert not sample["handoff_mismatch"]
    assert sample["target"]["tracking_position_error_m"] < 1e-5
    np.testing.assert_allclose(
        report["driver_samples"][0]["endpoint_in_prediction_rig_m"],
        [10, 0, 0],
        atol=1e-5,
    )


def test_altered_controller_plan_is_separate_from_tracking(audit_module):
    report = complete(*fixture(audit_module, changed_plan=2, tracking_offset=1))
    sample = report["controller_samples"][0]
    assert sample["handoff_max_position_error_m"] == pytest.approx(2, abs=1e-5)
    assert sample["handoff_mismatch"]
    assert sample["target"]["tracking_position_error_m"] < 1e-5


def test_tracking_error_does_not_claim_bad_handoff(audit_module):
    report = complete(*fixture(audit_module, tracking_offset=3))
    sample = report["controller_samples"][0]
    assert not sample["handoff_mismatch"]
    assert sample["target"]["tracking_position_error_m"] == pytest.approx(3, abs=1e-5)


def test_no_extrapolation_and_unmatched_are_separate_from_mismatch(audit_module):
    report = complete(*fixture(audit_module, controller_times=(800000, 1200000)))
    sample = report["controller_samples"][0]
    assert not sample["handoff_mismatch"]
    assert sample["handoff_status"] == "unmatched"
    assert sample["matched_driver_timestamp_us"] is None
    assert sample["target"]["requested"] is None


def test_retimed_controller_plan_matches_interpolated_driver(audit_module):
    run, response = fixture(audit_module, controller_times=(1000000, 1500000))
    request, available = run.pending
    request.planned_trajectory_in_rig.poses[-1].pose.vec.x = 5
    sample = complete(run, response)["controller_samples"][0]
    assert sample["handoff_status"] == "match"
    assert sample["handoff_max_position_error_m"] < 1e-5


def test_command_extraction_never_executes_pickle(audit_module, monkeypatch):
    monkeypatch.setattr(pickle, "loads", lambda *args: pytest.fail("must not unpickle"))
    for protocol in range(2, 6):
        payload = pickle.dumps(
            {"command": 0, "command_name": "LEFT"}, protocol=protocol
        )
        assert audit_module.command_name(payload) == "LEFT"
    assert audit_module.command_name(b"invalid pickle") == "unknown"
    assert (
        audit_module.command_name(pickle.dumps({"command_name": "MISSING"}))
        == "unknown"
    )
    assert audit_module.command_name(b"cos\nsystem\n(S'echo unsafe'\ntR.") == "unknown"


def test_unmatched_request_and_session_change_fail(audit_module):
    run, response = fixture(audit_module)
    with pytest.raises(ValueError, match="unmatched controller"):
        run.finish()
    with pytest.raises(ValueError, match="one nonempty"):
        run.add(
            logging.LogEntry(driver_request=driver.DriveRequest(session_uuid="other"))
        )
    empty = audit_module.Audit()
    with pytest.raises(ValueError, match="no matching"):
        empty.add(logging.LogEntry(controller_return=response))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_propagated_pose_rejected(audit_module, bad):
    run, response = fixture(audit_module)
    response.states[0].pose_local_to_rig.vec.x = bad
    with pytest.raises(ValueError, match="Non-finite"):
        complete(run, response)
