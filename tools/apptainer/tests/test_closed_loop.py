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


def custom_route():
    return dict(
        scene_id="scene",
        coordinate_frame="local",
        units="metres",
        waypoints=[[0, 0, 0], [20, 0, 0], [25, 5, 0], [25, 40, 0]],
    )


def test_explicit_route_reaches_runtime_configuration(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    import prepare_loop

    spec = dict(
        policy="alpamayo1_5",
        work="/workspace/run",
        scene="/workspace/scene.usdz",
        steps=180,
        approach_steps=6,
        checkpoint="/workspace/checkpoint",
        ports={name: 10000 for name in ("driver", "controller", "physics", "renderer")},
        route=custom_route(),
    )
    user, _, driver = prepare_loop.build_configs(
        spec, "scene", None, dict(positive="", negative="")
    )
    assert user["simulation_config"]["route_generator_type"] == "CUSTOM"
    assert (
        user["simulation_config"]["route_waypoints_in_local"]
        == spec["route"]["waypoints"]
    )
    assert driver["model"]["model_type"] == "alpamayo1_5"
    with pytest.raises(ValueError, match="scene_id"):
        prepare_loop.build_configs(
            spec, "different-scene", None, dict(positive="", negative="")
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(coordinate_frame="nre"),
        dict(units="pixels"),
        dict(scene_id=""),
        dict(waypoints=[]),
        dict(waypoints=[[0, 0, 0], [0, 0, 0]]),
        dict(waypoints=[[0, 0, 0], [1, float("inf"), 0]]),
        dict(waypoints=[[0, 0, 0], [1, True, 0]]),
    ],
)
def test_explicit_route_rejects_ambiguous_coordinates(monkeypatch, change):
    monkeypatch.syspath_prepend(str(TOOLS))
    from prepare_loop import validate_route_file

    with pytest.raises(ValueError):
        validate_route_file(custom_route() | change)


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
    run.args = SimpleNamespace(policy="vavam")
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
        approach_steps=0,
        policy="vavam",
        policy_gpu=0,
        renderer_gpu=0,
        route_file=None,
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
@pytest.mark.parametrize("approach_steps", [0, 8])
def test_generated_driver_config_parses_real_schema_and_preserves_camera_preset(
    monkeypatch,
    command,
    code,
    approach_steps,
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
        policy="vavam",
        work="/workspace/run",
        scene="/workspace/scene.usdz",
        checkpoint="/workspace/policy.pt",
        tokenizer="/workspace/tokenizer.jit",
        steps=12,
        command=command,
        approach_steps=approach_steps,
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
    simulation = user["simulation_config"]
    assert simulation["skip_driver_during_force_gt"] == (approach_steps > 0)
    handover = simulation["force_gt_duration_us"]
    end = 5 * 33333 + simulation["n_sim_steps"] * simulation["control_timestep_us"]
    assert (end - handover) // simulation["control_timestep_us"] == 11 - approach_steps


@pytest.mark.parametrize("approach", [-1, 10, 12])
def test_reject_approach_without_feedback_intervals(monkeypatch, approach):
    monkeypatch.syspath_prepend(str(TOOLS))
    import prepare_loop

    with pytest.raises(ValueError, match="two closed-loop intervals"):
        prepare_loop.build_configs(
            dict(work="/workspace/run", ports={}, steps=12, approach_steps=approach),
            "scene",
            {},
            {},
        )


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


def test_alpamayo_config_has_history_and_route_navigation_without_vavam_inputs(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(TOOLS))
    import prepare_loop
    import yaml
    from alpasim_driver.schema import DriverConfig
    from omegaconf import OmegaConf

    spec = dict(
        policy="alpamayo1_5",
        work="/workspace/run",
        scene="/workspace/scene.usdz",
        checkpoint="/workspace/models/policy",
        steps=180,
        approach_steps=6,
        ports=dict(driver=10001, physics=10002, controller=10003, renderer=10004),
    )
    user, _, driver = prepare_loop.build_configs(
        spec, "scene", None, dict(positive="road", negative="")
    )
    parsed = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DriverConfig), driver)
    )
    preset_dir = TOOLS.parents[1] / "src/wizard/configs/driver"
    upstream = yaml.safe_load((preset_dir / "alpamayo1_5.yaml").read_text())
    onecam = yaml.safe_load((preset_dir / "alpamayo1_5_1cam.yaml").read_text())
    assert parsed.model.model_type == "alpamayo1_5"
    assert parsed.model.checkpoint_path == spec["checkpoint"]
    assert parsed.model.num_trajectory_samples == 1
    assert parsed.model.cfg_guidance_weight is None
    assert parsed.rectification is None
    assert (
        "tokenizer_path" not in driver["model"]
        and "default_command" not in driver["route"]
    )
    assert (
        parsed.inference.context_length == upstream["inference"]["context_length"] == 4
    )
    assert (
        parsed.inference.subsample_factor
        == onecam["inference"]["subsample_factor"]
        == 3
    )
    assert parsed.inference.use_cameras == onecam["inference"]["use_cameras"]
    simulation = user["simulation_config"]
    assert simulation["route_generator_type"] == "RECORDED"
    assert simulation["skip_driver_during_force_gt"]
    warmup = yaml.safe_load((preset_dir.parent / "chunking/8frame.yaml").read_text())
    assert (
        simulation["force_gt_duration_us"]
        == warmup["runtime"]["simulation_config"]["force_gt_duration_us"]
    )
    assert simulation["force_gt_duration_us"] >= 1_500_000
    assert simulation["n_sim_steps"] * simulation["control_timestep_us"] > 47_000_000
    with pytest.raises(ValueError, match="ego history"):
        prepare_loop.build_configs(dict(spec, approach_steps=0), "scene", None, {})


@pytest.mark.parametrize("policy", ["vavam", "alpamayo1_5"])
def test_selected_backend_preflight_does_not_require_other_policy(
    tmp_path, monkeypatch, policy
):
    import types

    monkeypatch.syspath_prepend(str(TOOLS))
    import service_entry

    calls = []
    monkeypatch.setenv("ALPASIM_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(service_entry.importlib, "import_module", calls.append)
    monkeypatch.setattr(
        service_entry.importlib.metadata, "version", lambda name: "test"
    )
    plugins = types.ModuleType("alpasim_plugins.plugins")
    selected = []
    plugins.models = SimpleNamespace(get=selected.append)
    monkeypatch.setitem(
        sys.modules, "alpasim_plugins", types.ModuleType("alpasim_plugins")
    )
    monkeypatch.setitem(sys.modules, "alpasim_plugins.plugins", plugins)
    backend = types.ModuleType("alpamayo1_5")
    backend.helper = SimpleNamespace(BASE_PROCESSOR_NAME="fixture/processor")
    processor_calls = []
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *a, **k: processor_calls.append((a, k))
    )
    monkeypatch.setitem(sys.modules, "alpamayo1_5", backend)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert service_entry.check_environment("policy", policy) == 0
    expected = "vam" if policy == "vavam" else "alpamayo1_5"
    assert selected == [expected]
    assert f"alpasim_driver.models.{expected}_model" in calls
    other = "alpamayo1_5" if policy == "vavam" else "vam"
    assert f"alpasim_driver.models.{other}_model" not in calls
    assert bool(processor_calls) == (policy == "alpamayo1_5")
    if processor_calls:
        assert processor_calls[0][1] == {"local_files_only": True}


def test_missing_alpamayo_processor_cache_fails_preflight(tmp_path, monkeypatch):
    import types

    monkeypatch.syspath_prepend(str(TOOLS))
    import service_entry

    monkeypatch.setenv("ALPASIM_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(service_entry.importlib, "import_module", lambda name: None)
    backend = types.ModuleType("alpamayo1_5")
    backend.helper = SimpleNamespace(BASE_PROCESSOR_NAME="fixture/processor")

    def missing(*args, **kwargs):
        assert kwargs["local_files_only"]
        raise OSError("processor not cached")

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=missing)
    monkeypatch.setitem(sys.modules, "alpamayo1_5", backend)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert service_entry.check_environment("policy", "alpamayo1_5") == 1
    report = json.loads((tmp_path / "environment.json").read_text())
    assert "processor not cached" in report["failures"]["processor_cache"]


@pytest.mark.parametrize("policy_gpu,renderer_gpu", [(0, 0), (1, 0)])
def test_services_use_selected_environment_and_allocated_gpu_ordinals(
    loop, tmp_path, monkeypatch, policy_gpu, renderer_gpu
):
    run = object.__new__(loop.Run)
    run.args = SimpleNamespace(
        policy="alpamayo1_5",
        policy_gpu=policy_gpu,
        renderer_gpu=renderer_gpu,
        policy_hf_home=tmp_path / "assets",
    )
    run.project = run.root = tmp_path
    run.work = tmp_path / "output"
    run.work.mkdir()
    run.cache = tmp_path / "cache"
    run.image = tmp_path / "image.sif"
    run.policy_venv = tmp_path / "apps/policy-env"
    run.deadline = time.monotonic() + 60
    run.processes, run.logs = [], []
    calls = []
    monkeypatch.setattr(loop.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    try:
        for name, profile, gpu in (
            ("driver", "policy", True),
            ("renderer", "renderer", True),
            ("physics", "core", True),
            ("export", "policy", False),
        ):
            run.launch(name, profile, ["-V"], gpu=gpu)
            cmd = calls[-1]
            if gpu:
                expected = policy_gpu if name == "driver" else renderer_gpu
                assert cmd[cmd.index("--gpu-index") + 1] == str(expected)
            else:
                assert "--gpu" not in cmd and "--gpu-index" not in cmd
            if profile == "policy":
                assert cmd[cmd.index("--venv") + 1] == str(run.policy_venv)
                assert cmd[cmd.index("--hf-home") + 1] == str(run.args.policy_hf_home)
            else:
                assert "--venv" not in cmd and "--hf-home" not in cmd
    finally:
        for log in run.logs:
            log.close()


def test_checkpoint_shards_are_checked_before_model_loading(loop, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}})
    )
    (checkpoint / "part1.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="part2"):
        loop.validate_alpamayo_checkpoint(checkpoint, tmp_path)
    (checkpoint / "part2.safetensors").touch()
    with pytest.raises(ValueError, match="empty"):
        loop.validate_alpamayo_checkpoint(checkpoint, tmp_path)
    (checkpoint / "part2.safetensors").write_bytes(b"fixture")
    loop.validate_alpamayo_checkpoint(checkpoint, tmp_path)
    index.unlink()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    loop.validate_alpamayo_checkpoint(checkpoint, tmp_path)


@pytest.mark.parametrize(
    "extra",
    [
        ["--command", "left"],
        ["--tokenizer", "encoder.jit"],
        ["--approach-steps", "0"],
    ],
)
def test_alpamayo_rejects_incompatible_arguments_before_launch(
    loop, tmp_path, monkeypatch, extra
):
    monkeypatch.setattr(
        loop, "Run", lambda args: pytest.fail("must reject before creating a run")
    )
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
            str(tmp_path / "out"),
            "--policy",
            "alpamayo1_5",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        loop.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_alpamayo_import_check_does_not_start_models_and_records_route_mode(
    loop, tmp_path, monkeypatch
):
    work = tmp_path / "out"
    captured = []
    state = SimpleNamespace(
        work=work,
        processes=[],
        logs=[],
        runtime_started=False,
        export_completed=False,
        check=lambda: None,
        simulate=lambda: pytest.fail("check-only must not simulate"),
    )

    def make_run(args):
        captured.append(args)
        return state

    monkeypatch.setattr(loop, "Run", make_run)
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
            "--policy",
            "alpamayo1_5",
        ],
    )
    assert loop.main() == 0
    assert captured[0].approach_steps == 6
    status = json.loads((work / "status.json").read_text())
    assert status["policy"] == "alpamayo1_5"
    assert status["command"] is None and status["navigation"] == "recorded_route"
    assert not status["runtime_started"]
