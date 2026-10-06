"""Predictor registry (`eval/predictors.py`): one interface so the rule baseline, the shipped GNN,
and (from Phase 2) the symbolic engine are all scored by the same harness on the same cases."""
from __future__ import annotations

import pytest

from core.taxonomy import SpaceClass, TimeClass
from eval.predictors import Prediction, available_predictors, get_predictor

_LINEAR_PY = "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n"
_QUADRATIC_PY = (
    "def f(xs):\n    t = 0\n    for x in xs:\n        for y in xs:\n            t += x * y\n"
    "    return t\n"
)
_QUADRATIC_CPP = (
    "int f(vector<int>& a) {\n  int t = 0;\n  for (int i = 0; i < a.size(); i++)\n"
    "    for (int j = 0; j < a.size(); j++) t += a[i] * a[j];\n  return t;\n}\n"
)


def test_available_predictors_lists_the_baselines() -> None:
    names = available_predictors()
    assert "rule" in names
    assert "gnn" in names


def test_unknown_predictor_name_is_rejected() -> None:
    with pytest.raises(KeyError, match="nope"):
        get_predictor("nope")


def test_rule_predictor_reads_loop_nesting_depth() -> None:
    rule = get_predictor("rule")
    assert rule(_LINEAR_PY, "python").time_class == "O(n)"
    assert rule(_QUADRATIC_PY, "python").time_class == "O(n^2)"
    assert rule(_QUADRATIC_CPP, "cpp").time_class == "O(n^2)"


def test_prediction_classes_are_always_valid_taxonomy_strings() -> None:
    valid_time = {c.value for c in TimeClass}
    valid_space = {c.value for c in SpaceClass}
    for name in ("rule", "gnn"):
        prediction = get_predictor(name)(_QUADRATIC_PY, "python")
        assert prediction.error is None, prediction.error
        assert prediction.time_class in valid_time
        assert prediction.space_class in valid_space


def test_a_failing_prediction_is_captured_not_raised() -> None:
    # The harness scores thousands of snippets; one bad one must be a recorded miss, never a crash
    # that hides the other 4999.
    prediction = get_predictor("rule")("def f(:", "cobol")
    assert isinstance(prediction, Prediction)
    assert prediction.time_class is None
    assert prediction.error


def test_predictors_are_deterministic() -> None:
    for name in ("rule", "gnn"):
        predictor = get_predictor(name)
        assert predictor(_QUADRATIC_PY, "python") == predictor(_QUADRATIC_PY, "python")


def test_a_candidate_artifacts_directory_can_be_named_in_the_environment(
    monkeypatch, tmp_path
) -> None:
    """A retrained GNN is compared against the served one before it replaces it: the predictors
    load whatever directory `POLYO_ARTIFACTS_DIR` names (and the served one when it is unset)."""
    import api.models_registry as registry_module
    from eval import predictors

    seen: list[object] = []

    def fake_load(*directory: object) -> str:
        seen.append(directory)
        return "registry"

    monkeypatch.setattr(registry_module, "load_registry", fake_load)
    try:
        monkeypatch.setenv("POLYO_ARTIFACTS_DIR", str(tmp_path))
        predictors._registry.cache_clear()
        assert predictors._registry() == "registry"
        monkeypatch.delenv("POLYO_ARTIFACTS_DIR")
        predictors._registry.cache_clear()
        assert predictors._registry() == "registry"
    finally:
        predictors._registry.cache_clear()
    assert seen == [(tmp_path,), ()]
