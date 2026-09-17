"""Rung 2: IR features + LightGBM (plan §8).

Where rung 1 reads the IR symbol *sequence*, this rung reads the aggregated
feature vector from `features.tabular` -- feature engineering and
interpretability (attribution work lands alongside this once a trained
model is wired into the API in Phase 6) rather than an n-gram bag.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from features.tabular import TabularFeatures
from models.dataset import ParsedExample

FEATURE_NAMES: tuple[str, ...] = tuple(TabularFeatures(0, 0).as_dict().keys())


def _feature_matrix(examples: list[ParsedExample], feature_names: tuple[str, ...]) -> np.ndarray:
    rows = [e.features.as_dict() for e in examples]
    return np.array([[row[name] for name in feature_names] for row in rows], dtype=float)


@dataclass(frozen=True, slots=True)
class GbdtModel:
    booster: lgb.LGBMClassifier
    # Defaults to the full feature set -- overridden by `fit`'s
    # `feature_names` param, which exists for `eval/ablations.py`'s
    # leave-one-family-out ablation (plan §9) to train on a subset without
    # duplicating this module's training/scoring logic.
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def classes(self) -> list[str]:
        return list(self.booster.classes_)

    def predict_proba(self, examples: list[ParsedExample]) -> list[dict[str, float]]:
        x = _feature_matrix(examples, self.feature_names)
        proba = self.booster.predict_proba(x)
        return [dict(zip(self.classes, row.tolist(), strict=True)) for row in proba]

    def predict(self, examples: list[ParsedExample]) -> list[str]:
        x = _feature_matrix(examples, self.feature_names)
        return list(self.booster.predict(x))

    def decision_function(self, examples: list[ParsedExample]) -> np.ndarray:
        """Raw per-class margins (pre-softmax) -- what `models/calibrate.py`
        fits temperature scaling against."""
        x = _feature_matrix(examples, self.feature_names)
        return np.asarray(self.booster.predict(x, raw_score=True))

    def feature_importance(self) -> dict[str, float]:
        importances = self.booster.feature_importances_
        return dict(zip(self.feature_names, importances.tolist(), strict=True))


def fit(
    examples: list[ParsedExample],
    labels: list[str],
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> GbdtModel:
    if len(examples) != len(labels):
        raise ValueError("examples and labels must be the same length")
    x = _feature_matrix(examples, feature_names)
    booster = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        class_weight="balanced",
        verbose=-1,
    )
    booster.fit(x, labels)
    return GbdtModel(booster=booster, feature_names=feature_names)
