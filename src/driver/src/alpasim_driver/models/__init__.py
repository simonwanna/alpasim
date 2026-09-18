# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Model abstraction layer for trajectory prediction models."""

from importlib import import_module

from .base import (
    BaseTrajectoryModel,
    CameraFrame,
    CameraImages,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

_MODEL_MODULES = {
    "Alpamayo15Model": ".alpamayo1_5_model",
    "Alpamayo1Model": ".alpamayo1_model",
    "Alpamayo2Model": ".alpamayo2_model",
    "ManualModel": ".manual_model",
    "VAMModel": ".vam_model",
}


def __getattr__(name: str):
    """Import optional model dependencies only when that backend is requested."""
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(_MODEL_MODULES[name], __name__), name)
    globals()[name] = model
    return model


__all__ = [
    "Alpamayo15Model",
    "Alpamayo1Model",
    "Alpamayo2Model",
    "BaseTrajectoryModel",
    "CameraFrame",
    "CameraImages",
    "DriveCommand",
    "ManualModel",
    "ModelPrediction",
    "PredictionInput",
    "VAMModel",
]
