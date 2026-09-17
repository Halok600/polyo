"""Rung 2 tests: IR features + LightGBM (plan §8)."""
from __future__ import annotations

import pytest

from data.corpus import CorpusRecord
from models.dataset import build_examples
from models.gbdt import FEATURE_NAMES, fit

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


def _record(code: str, i: int, label: str) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=label,
        space_class=None,
    )


def _synthetic_examples(n_per_class: int = 100):
    # LightGBM's default min_child_samples=20 (sensible for a real,
    # 100K+-row corpus, so not weakened here) means a tree literally cannot
    # split at all on a too-small synthetic set -- n_per_class must clear
    # that floor with margin for this to be a meaningful test.
    records = []
    i = 0
    for code, label in ((_CONSTANT, "O(1)"), (_LINEAR, "O(n)"), (_QUADRATIC, "O(n^2)")):
        for _ in range(n_per_class):
            records.append(_record(code, i, label))
            i += 1
    examples, stats = build_examples(records)
    assert stats.failed == 0
    labels = [r.time_class for r in records]
    return examples, labels


def test_fit_rejects_mismatched_lengths():
    examples, labels = _synthetic_examples(n_per_class=1)
    with pytest.raises(ValueError):
        fit(examples, labels[:-1])


def test_fit_separates_obviously_different_loop_depths():
    examples, labels = _synthetic_examples()
    model = fit(examples, labels)
    predictions = model.predict(examples)
    accuracy = sum(p == y for p, y in zip(predictions, labels, strict=True)) / len(labels)
    # max_loop_nesting_depth alone perfectly separates these three classes --
    # a failure here means the feature pipeline is broken, not that the
    # model is weak.
    assert accuracy == 1.0


def test_predict_proba_rows_sum_to_one():
    examples, labels = _synthetic_examples()
    model = fit(examples, labels)
    distributions = model.predict_proba(examples[:5])
    for dist in distributions:
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        assert set(dist.keys()) == set(model.classes)


def test_decision_function_shape_matches_classes():
    examples, labels = _synthetic_examples()
    model = fit(examples, labels)
    scores = model.decision_function(examples[:4])
    assert scores.shape == (4, len(model.classes))


def test_feature_importance_covers_every_declared_feature():
    examples, labels = _synthetic_examples()
    model = fit(examples, labels)
    importances = model.feature_importance()
    assert set(importances.keys()) == set(FEATURE_NAMES)
    # max_loop_nesting_depth is the only signal that distinguishes these
    # three classes -- it should dominate.
    assert importances["max_loop_nesting_depth"] == max(importances.values())
