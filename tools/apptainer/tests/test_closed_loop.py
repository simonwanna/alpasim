# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1]


@pytest.fixture
def loop(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "closed_loop_test", TOOLS / "closed_loop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_preflight_checks_both_environments_without_gpu(loop, tmp_path):
    run = object.__new__(loop.Run)
    run.root = tmp_path
    run.work = tmp_path
    run.project = tmp_path
    calls = []

    def launch(name, profile, arguments, gpu=False, limit=None):
        calls.append((profile, gpu))
        return SimpleNamespace(returncode=1, poll=lambda: 1)

    run.launch = launch
    with pytest.raises(RuntimeError, match="no GPU services"):
        run.check()
    assert calls == [("core", False), ("policy", False)]


def test_startup_failure_is_detected_before_connecting(loop, monkeypatch):
    run = object.__new__(loop.Run)
    run.deadline = time.monotonic() + 10
    monkeypatch.setattr(
        loop.socket, "create_connection", lambda *a, **k: pytest.fail("dead service")
    )
    with pytest.raises(RuntimeError, match="physics stopped"):
        run.wait_ready({"physics": SimpleNamespace(poll=lambda: 1)}, {"physics": 1234})


def test_controller_can_write_session_log_at_launch(loop, tmp_path, monkeypatch):
    run = object.__new__(loop.Run)
    run.work = run.root = run.project = tmp_path
    run.args = SimpleNamespace(
        scene=tmp_path / "scene.usdz",
        checkpoint=tmp_path / "model.pt",
        tokenizer=tmp_path / "tokenizer.jit",
        seed_session=tmp_path / "session.pb",
        steps=12,
        command="straight",
    )
    monkeypatch.setattr(
        loop,
        "reserve_ports",
        lambda: {
            name: SimpleNamespace(
                getsockname=lambda: ("127.0.0.1", 1234), close=lambda: None
            )
            for name in loop.SERVICES
        },
    )
    run.finish = lambda *args: None

    def launch(name, profile, arguments, **kwargs):
        if name == "controller":
            container_dir = Path(arguments[arguments.index("--log_dir") + 1])
            host_dir = tmp_path / container_dir.relative_to("/workspace")
            (host_dir / "session.csv").write_text("timestamp,x,y\n")
            raise RuntimeError("controller launch checked")
        assert name == "prepare"

    run.launch = launch
    with pytest.raises(RuntimeError, match="controller launch checked"):
        run.simulate()
    assert (tmp_path / "controller-output/session.csv").is_file()
    launch = json.loads((tmp_path / "launch.json").read_text())
    assert launch["command"] == "straight"


def test_cleanup_only_stops_owned_process(loop):
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        for _ in range(2)
    ]
    try:
        loop.stop_launchers(processes[:1])
        assert processes[0].poll() is not None
        assert processes[1].poll() is None
    finally:
        loop.stop_launchers(processes)


@pytest.mark.parametrize("command", [None, "left", "right"])
def test_failed_check_preserves_evidence_and_does_not_report_runtime_started(
    loop, tmp_path, monkeypatch, command
):
    work = tmp_path / "results"
    state = SimpleNamespace(
        work=work, processes=[], logs=[], runtime_started=False, export_completed=False
    )

    def fail():
        raise RuntimeError("missing dependency")

    state.check = fail
    state.simulate = lambda: pytest.fail("must not simulate after failed check")
    monkeypatch.setattr(loop, "Run", lambda args: state)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "closed_loop",
            "--project",
            str(tmp_path),
            "--image",
            str(tmp_path / "image"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(work),
            *([] if command is None else ["--command", command]),
        ],
    )
    previous = signal.getsignal(signal.SIGTERM)
    assert loop.main() == 1
    assert signal.getsignal(signal.SIGTERM) == previous
    result = json.loads((work / "status.json").read_text())
    assert result["command"] == (command or "straight")
    assert result["exit_status"] == 1
    assert not result["runtime_started"] and not result["export_completed"]


@pytest.mark.parametrize("command,code", [("straight", 2), ("left", 1), ("right", 0)])
def test_generated_driver_config_parses_real_schema_and_preserves_camera_preset(
    monkeypatch,
    command,
    code,
):
    monkeypatch.syspath_prepend(str(TOOLS))
    import prepare_loop
    import yaml
    from alpasim_driver.schema import DriverConfig
    from omegaconf import OmegaConf

    root = TOOLS.parents[1]
    rectification = yaml.safe_load(
        (root / "src/wizard/configs/driver/vavam_video_model.yaml").read_text()
    )["driver"]["rectification"]
    spec = dict(
        work="/workspace/run",
        scene="/workspace/scene.usdz",
        checkpoint="/workspace/policy.pt",
        tokenizer="/workspace/tokenizer.jit",
        steps=12,
        command=command,
        ports=dict(driver=10001, physics=10002, controller=10003, renderer=10004),
    )
    user, network, driver = prepare_loop.build_configs(
        spec, "test-scene", rectification, dict(positive="road", negative="")
    )
    parsed = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DriverConfig), driver)
    )
    assert parsed.route.default_command == code
    assert not parsed.route.use_waypoint_commands
    assert parsed.model.checkpoint_path == spec["checkpoint"]
    assert (
        list(parsed.rectification["camera_front_wide_120fov"].resolution_hw)
        == rectification["camera_front_wide_120fov"]["resolution_hw"]
    )
    assert network["driver"]["endpoints"][0]["address"].endswith(str(parsed.port))
    assert (
        user["simulation_config"]["force_gt_duration_us"]
        < user["simulation_config"]["control_timestep_us"] * spec["steps"]
    )
    assert user["endpoints"]["trafficsim"]["skip"]


def test_environment_check_records_import_errors_and_keeps_checking(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(TOOLS))
    import service_entry

    calls = []

    def missing(name):
        calls.append(name)
        raise ImportError("fixture missing dependency")

    monkeypatch.setenv("ALPASIM_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(service_entry.importlib, "import_module", missing)
    assert service_entry.check_environment("policy") == 1
    report = json.loads((tmp_path / "environment.json").read_text())
    assert "alpasim_driver.main" in calls and "alpasim_utils.logs" in calls
    assert "fixture missing dependency" in report["failures"]["alpasim_driver.main"]
