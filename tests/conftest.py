"""Shared pytest fixtures. `tiny_registry` builds a real (but tiny, fast)
`ModelRegistry` -- a GNN exported to numpy + a GBDT's normalised feature
importance -- for `api/` tests that need a working registry without
depending on `models/artifacts/` (CI never runs `models.train_production`;
those artifacts only exist after a real, GPU-backed local training run).
"""
from __future__ import annotations

import pytest

from api.models_registry import Calibration, ModelRegistry
from data.corpus import CorpusRecord
from models import gbdt, gnn
from models.calibrate import fit_temperature
from models.conformal import fit_conformal
from models.dataset import build_examples, space_labels, time_labels, with_label
from models.export_numpy import NumpyGnnModel, export_gnn

_CONSTANT = "def f(x):\n    return x + 1\n"
_LINEAR = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
_QUADRATIC = (
    "def f(xs):\n"
    "    total = 0\n"
    "    for x in xs:\n"
    "        for y in xs:\n"
    "            total += x * y\n"
    "    return total\n"
)


def _record(code: str, i: int, time_label: str) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=time_label,
        space_class="O(1)" if time_label != "O(n)" else "O(n)",
    )


@pytest.fixture
def tiny_registry(tmp_path) -> ModelRegistry:
    codes_and_labels = [(_CONSTANT, "O(1)"), (_LINEAR, "O(n)"), (_QUADRATIC, "O(n^2)")]
    records = [
        _record(code, i * 10 + j, label)
        for i, (code, label) in enumerate(codes_and_labels)
        for j in range(10)
    ]
    examples, stats = build_examples(records)
    assert stats.failed == 0
    time_y = time_labels(examples)
    space_y = space_labels(examples)

    time_model, space_model = gnn.fit_multitask(
        examples,
        time_y,
        space_y,
        hidden_dim=8,
        num_layers=2,
        batch_size=16,
        max_epochs=2,
        patience=1,
    )
    calibration = {}
    conformal = {}
    for dimension, model, labels in (("time", time_model, time_y), ("space", space_model, space_y)):
        # This fixture's records always set both labels, so `with_label`
        # here only exists to narrow the type from `list[str | None]` to
        # `list[str]` for `fit_temperature`, not to actually drop anything.
        labelled_examples, labelled_y = with_label(examples, labels)
        scores = model.decision_function(labelled_examples)
        calibrator = fit_temperature(scores, labelled_y, tuple(model.classes))
        calibration[dimension] = Calibration(
            temperature=calibrator.temperature, classes=tuple(model.classes)
        )
        proba = calibrator.calibrate(scores)
        conformal[dimension] = fit_conformal(proba, labelled_y, tuple(model.classes))

    npz_path = tmp_path / "gnn.npz"
    export_gnn(time_model.core, time_model.edge_kinds, npz_path)
    numpy_gnn = NumpyGnnModel.load(npz_path)

    feature_importance = {}
    for dimension, labels in (("time", time_y), ("space", space_y)):
        labelled_examples, labelled_y = with_label(examples, labels)
        gbdt_model = gbdt.fit(labelled_examples, labelled_y)
        importances = gbdt_model.feature_importance()
        total = sum(importances.values()) or 1.0
        feature_importance[dimension] = {
            name: value / total for name, value in importances.items()
        }
    feature_scales = {name: 1.0 for name in gbdt.FEATURE_NAMES}

    return ModelRegistry(
        gnn=numpy_gnn,
        calibration=calibration,
        conformal=conformal,
        feature_importance=feature_importance,
        feature_scales=feature_scales,
    )
