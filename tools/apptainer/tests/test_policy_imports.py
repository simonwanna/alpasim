# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Import-isolation checks without downloading Torch or model packages locally."""

import subprocess
import sys
from pathlib import Path

DRIVER_SOURCE = Path(__file__).resolve().parents[3] / "src/driver/src"


def test_shared_types_do_not_import_model_backends_or_native_geometry():
    program = """
import importlib.abc
import sys
import types
sys.path.insert(0, sys.argv[1])
# Only import dependencies are being tested; no tensor operations are simulated.
sys.modules['torch'] = types.ModuleType('torch')
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('vam', 'alpamayo', 'pygame', 'utils_rs', 'alpasim_utils')):
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import alpasim_driver.models as models
assert models.DriveCommand.STRAIGHT.value == 1
assert models.PredictionInput is not None
assert not any(name.endswith('_model') for name in sys.modules if name.startswith('alpasim_driver.models.'))
try:
    models.nonexistent
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute accepted')
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(DRIVER_SOURCE)], check=True
    )


def test_public_model_export_remains_lazy_and_cached():
    program = """
import sys
import types
sys.path.insert(0, sys.argv[1])
sys.modules['torch'] = types.ModuleType('torch')
import alpasim_driver.models as models
backend = types.ModuleType('alpasim_driver.models.vam_model')
backend.VAMModel = type('VAMModel', (), {})
sys.modules[backend.__name__] = backend
assert 'VAMModel' not in vars(models)
assert models.VAMModel is backend.VAMModel
assert vars(models)['VAMModel'] is backend.VAMModel
assert not any('alpamayo' in name for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(DRIVER_SOURCE)], check=True
    )
