# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Session command behavior without requiring inference/GPU dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]

# Execute the actual Session class with real config, cache, enum, navigation,
# and protobuf types. Only module-level inference imports are excluded.
PROGRAM = """
import ast
import dataclasses
import logging
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parent))
sys.modules['torch'] = types.ModuleType('torch')
from alpasim_driver.frame_cache import FrameCache
from alpasim_driver.models.base import DriveCommand
from alpasim_driver.navigation import determine_command_from_route
from alpasim_driver.schema import DriverConfig, InferenceConfig
from alpasim_grpc.v0 import common_pb2, egodriver_pb2

source = Path(sys.argv[1]) / 'main.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Session')
future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
unit = ast.Module(body=[future, node], type_ignores=[])
ast.fix_missing_locations(unit)
module = types.ModuleType('session_command_test')
sys.modules[module.__name__] = module
module.__dict__.update(
    dataclass=dataclasses.dataclass, field=dataclasses.field, threading=threading,
    FrameCache=FrameCache, DriveCommand=DriveCommand,
    determine_command_from_route=determine_command_from_route,
    logger=logging.getLogger('session_command_test'),
)
exec(compile(unit, str(source), 'exec'), module.__dict__)
Session = module.Session
request = egodriver_pb2.DriveSessionRequest(session_uuid='test')
request.rollout_spec.vehicle.available_cameras.add(logical_id='front')
cfg = DriverConfig(inference=InferenceConfig())
cfg.inference.use_cameras = ['front']
cfg.rectification = None

def create():
    return Session.create(request, cfg, context_length=1)

if sys.argv[2] == 'mapping':
    assert create().current_command == DriveCommand.STRAIGHT
    pairs = [(0, DriveCommand.RIGHT), (1, DriveCommand.LEFT), (2, DriveCommand.STRAIGHT)]
    for code, expected in pairs:
        cfg.route.default_command = code
        session = create()
        assert session.current_command == expected, (code, session.current_command)
        assert isinstance(session.frame_caches['front'], FrameCache)
elif sys.argv[2] == 'invalid':
    for invalid in [-1, 3, 99, 'left', True, 1.5, None]:
        cfg.route.default_command = invalid
        try:
            create()
        except ValueError as error:
            assert 'route.default_command' in str(error)
        else:
            raise AssertionError(f'Accepted invalid command {invalid!r}')
elif sys.argv[2] == 'route':
    cfg.route.default_command = 0
    session = create()
    session.poses.append(common_pb2.PoseAtTime(timestamp_us=100))
    route = egodriver_pb2.Route(waypoints=[common_pb2.Vec3(x=10, y=4)])
    session.update_command_from_route(route, use_waypoint_commands=False)
    assert session.current_command == DriveCommand.RIGHT
    session.update_command_from_route(route, use_waypoint_commands=True,
                                     command_distance_threshold=2, min_lookahead_distance=5)
    assert session.current_command == DriveCommand.LEFT
else:
    raise AssertionError('Unknown test mode')
"""


@pytest.mark.parametrize("mode", ["mapping", "invalid", "route"])
def test_session_default_command_and_route_override(mode):
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROGRAM, str(SOURCE), mode],
        check=True,
    )
