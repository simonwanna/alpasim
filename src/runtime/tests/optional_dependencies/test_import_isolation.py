# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit checks for optional features, with unrelated service dependencies mocked."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0 import runtime_pb2

RUNTIME = Path(__file__).resolve().parents[2] / "alpasim_runtime"
BLOCKED = (
    "trajdata",
    "eval.runtime_evaluator",
    "eval.scenario_evaluator",
    "eval.data",
    "eval.aggregation.main",
    "alpasim_runtime.telemetry.plot_metrics",
)


def load_runtime_module(monkeypatch, relative_path):
    """Execute the whole source module, isolating unrelated core service imports."""
    path = RUNTIME / relative_path
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.startswith(BLOCKED):
            continue
        if node.module.startswith(
            ("alpasim_runtime", "alpasim_utils")
        ) or node.module in (
            "eval.schema",
            "eval.aggregation.failed_rollouts",
        ):
            module = ModuleType(node.module)
            for alias in node.names:
                setattr(module, alias.name, MagicMock(name=alias.name))
            monkeypatch.setitem(sys.modules, node.module, module)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(BLOCKED):
            raise ModuleNotFoundError(f"Optional dependency unavailable: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    name = "isolated_" + relative_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, original_import


@pytest.mark.parametrize(
    "relative_path",
    [
        "event_loop.py",
        "route_generator.py",
        "unbound_rollout.py",
        "worker/ipc.py",
        "daemon/engine.py",
        "simulate/__main__.py",
    ],
)
def test_disabled_features_do_not_require_optional_imports(monkeypatch, relative_path):
    load_runtime_module(monkeypatch, relative_path)


def make_rollout(module, enabled):
    gt = MagicMock()
    gt.timestamps_us = np.array([0, 100], dtype=np.uint64)
    gt.clip.return_value.timestamps_us = gt.timestamps_us
    gt.velocities.return_value = np.zeros((2, 3))
    gt.yaw_rates.return_value = np.zeros(2)
    unbound = SimpleNamespace(
        egomotion_context_start_us=0,
        first_policy_timestamp_us=100,
        gt_ego_trajectory=gt,
        traffic_objs=MagicMock(),
        planner_delay_us=0,
        vector_map=None,
        route_generator_type="RECORDED",
        route_start_offset_m=0,
        rollout_uuid="rollout",
        scene_id="scene",
        save_path_root="outputs",
    )
    # Keep the broadcaster container real so handler registration is observable.
    module.MessageBroadcaster = lambda handlers: SimpleNamespace(handlers=handlers)
    return module.EventBasedRollout(
        unbound=unbound,
        data_source=MagicMock(),
        driver=MagicMock(),
        renderer_service=MagicMock(),
        physics=MagicMock(),
        trafficsim=MagicMock(),
        controller=MagicMock(),
        camera_catalog=MagicMock(),
        eval_config=SimpleNamespace(enabled=enabled),
        eval_executor=MagicMock(),
    )


def test_disabled_evaluation_preserves_log_writer_without_evaluator(monkeypatch):
    module, _ = load_runtime_module(monkeypatch, "event_loop.py")
    rollout = make_rollout(module, enabled=False)
    assert rollout._runtime_evaluator is None
    assert rollout.broadcaster.handlers == [module.LogWriter.return_value]


def test_enabled_evaluation_still_requires_evaluator(monkeypatch):
    module, _ = load_runtime_module(monkeypatch, "event_loop.py")
    with pytest.raises(ModuleNotFoundError, match="eval.runtime_evaluator"):
        make_rollout(module, enabled=True)


def test_enabled_evaluation_registers_real_evaluator_interface(monkeypatch):
    module, original_import = load_runtime_module(monkeypatch, "event_loop.py")
    evaluator_module = ModuleType("eval.runtime_evaluator")
    evaluator_module.RuntimeEvaluator = MagicMock()
    monkeypatch.setitem(sys.modules, evaluator_module.__name__, evaluator_module)
    monkeypatch.setattr(builtins, "__import__", original_import)
    rollout = make_rollout(module, enabled=True)
    evaluator_module.RuntimeEvaluator.assert_called_once_with(
        eval_config=rollout.eval_config,
        rollout_uuid="rollout",
        scene_id="scene",
        save_path_root="outputs",
        vector_map=None,
    )
    assert rollout.broadcaster.handlers == [
        module.LogWriter.return_value,
        evaluator_module.RuntimeEvaluator.return_value,
    ]


def test_empty_metrics_do_not_import_eval_data(monkeypatch):
    module, _ = load_runtime_module(monkeypatch, "daemon/engine.py")
    result = SimpleNamespace(eval_result=None)
    assert module._build_timestep_metrics(result) == []
    assert module._build_aggregated_metrics(result) == {}


def test_enabled_metrics_keep_all_aggregation_mappings(monkeypatch):
    module, original_import = load_runtime_module(monkeypatch, "daemon/engine.py")

    class AggregationType(Enum):
        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        LAST = "last"

    data_module = ModuleType("eval.data")
    data_module.AggregationType = AggregationType
    monkeypatch.setitem(sys.modules, data_module.__name__, data_module)
    monkeypatch.setattr(builtins, "__import__", original_import)
    metrics = [
        SimpleNamespace(
            name=aggregation.value,
            timestamps_us=[100],
            values=[2.0],
            valid=[True],
            time_aggregation=aggregation,
        )
        for aggregation in AggregationType
    ]
    result = SimpleNamespace(eval_result=SimpleNamespace(timestep_metrics=metrics))
    converted = module._build_timestep_metrics(result)
    assert [metric.time_aggregation for metric in converted] == [
        runtime_pb2.TIME_AGGREGATION_MEAN,
        runtime_pb2.TIME_AGGREGATION_MEDIAN,
        runtime_pb2.TIME_AGGREGATION_MAX,
        runtime_pb2.TIME_AGGREGATION_MIN,
        runtime_pb2.TIME_AGGREGATION_LAST,
    ]
    assert [metric.name for metric in converted] == [metric.name for metric in metrics]
