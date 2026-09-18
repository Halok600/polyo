"""Loads the served model artifacts (`models/artifacts/`, produced by
`python -m models.train_production`) once, at process start -- not per
request, and not from `GET /health`, which must stay instant and untouched
by model state (plan §10).

Only `numpy` (via `models.export_numpy`) is needed here -- no torch, no
scikit-learn, no lightgbm (see `api/attribution.py`'s module docstring for
why the attribution side avoids lightgbm/scipy too), consistent with the
served image's dependency budget (plan §13).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from models.export_numpy import NumpyGnnModel

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


class ModelsNotTrainedError(RuntimeError):
    """Raised when `models/artifacts/` is missing -- `python -m
    models.train_production` hasn't been run yet. Distinct from a request
    error: this is a deployment/startup problem, not a bad request."""


@dataclass(frozen=True, slots=True)
class Calibration:
    temperature: float
    classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    gnn: NumpyGnnModel
    calibration: dict[str, Calibration]
    # Per-dimension normalised GBDT feature-importance weights, and one
    # shared per-feature training-set scale (see `models/train_production.
    # py`'s `_feature_scales`) -- `api/attribution.py` divides a request's
    # raw feature value by its scale before weighting, so a large-
    # magnitude feature (node_count) doesn't dominate a small-magnitude
    # one (max_loop_nesting_depth) on raw numbers alone.
    feature_importance: dict[str, dict[str, float]]
    feature_scales: dict[str, float]


def load_registry(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelRegistry:
    gnn_path = artifacts_dir / "gnn.npz"
    calibration_path = artifacts_dir / "calibration.json"
    importance_path = artifacts_dir / "feature_importance.json"
    if not (gnn_path.is_file() and calibration_path.is_file() and importance_path.is_file()):
        raise ModelsNotTrainedError(
            f"missing model artifacts under {artifacts_dir} -- run "
            "`python -m models.train_production` first"
        )

    gnn = NumpyGnnModel.load(gnn_path)

    calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = {
        dimension: Calibration(temperature=entry["temperature"], classes=tuple(entry["classes"]))
        for dimension, entry in calibration_raw.items()
    }

    importance_raw = json.loads(importance_path.read_text(encoding="utf-8"))

    return ModelRegistry(
        gnn=gnn,
        calibration=calibration,
        feature_importance=importance_raw["weights"],
        feature_scales=importance_raw["scales"],
    )
