"""The served predictor is a hybrid (PolyO v2 phase 5): the symbolic engine answers whenever it
can prove or bound the cost, the ML model answers only when the engine says "unknown", and either
way the legacy response fields keep their shape while new ones explain where the answer came from.

These run against the real production artifacts (tracked in git), loaded once.
"""

from __future__ import annotations

import pytest

import api.predict as predict_module
from analysis.calibration import load_calibration
from api.models_registry import load_registry
from api.predict import predict
from eval.golden_suite import load_cases
from eval.perturb import add_dead_code

_CASES = {case.id: case for case in load_cases()}
_LANGUAGES = ("python", "cpp", "java", "javascript", "go")


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _source(case_id: str, language: str) -> str:
    return _CASES[case_id].sources[language]


# ---------------------------------------------------------------- the original bug, end to end
@pytest.mark.parametrize("language", _LANGUAGES)
def test_the_originally_reported_case_is_quadratic_in_every_language(registry, language) -> None:
    """The user's report: O(n^2) code came back as O(n^3)."""
    result = predict(registry, _source("quad_two_sum_bruteforce", language), language)
    assert result["time"]["class"] == "O(n^2)"
    assert result["time"]["engine"] == "symbolic"
    assert result["time"]["certainty"] == "certain"
    assert result["time"]["expression"] == "O(n^2)"
    assert result["space"]["class"] == "O(1)"


@pytest.mark.parametrize("padding", [0, 5, 20, 40, 80])
def test_dead_code_cannot_change_the_answer(registry, padding) -> None:
    source = _source("quad_two_sum_bruteforce", "python")
    padded = add_dead_code(source, "python", k=padding) if padding else source
    assert padded is not None
    assert predict(registry, padded, "python")["time"]["class"] == "O(n^2)"


# ---------------------------------------------------------------- the contract
def test_the_legacy_fields_keep_their_shape_and_the_new_ones_are_additive(registry) -> None:
    result = predict(registry, _source("quad_two_sum_bruteforce", "python"), "python")
    assert {
        "language_detected",
        "time",
        "space",
        "attribution",
        "curve",
        "ir",
        "warnings",
    } <= set(result)
    assert {"engine", "entry", "assumptions", "derivation"} <= set(result)
    legacy = {"class", "rank", "confidence", "distribution", "conformal_set"}
    legacy |= {"conformal_coverage", "abstain"}
    added = {"engine", "certainty", "expression", "extended_class", "projection_lossy"}
    for dimension in ("time", "space"):
        assert legacy | added <= set(result[dimension])


@pytest.mark.parametrize("dimension", ["time", "space"])
def test_a_symbolic_confidence_is_the_measured_accuracy(registry, dimension) -> None:
    result = predict(registry, _source("quad_two_sum_bruteforce", "python"), "python")
    entry = result[dimension]
    expected = load_calibration().confidence(entry["certainty"], dimension)
    assert entry["confidence"] == pytest.approx(expected, abs=1e-4)
    assert 0.0 < entry["confidence"] < 1.0
    assert entry["conformal_set"] == [entry["class"]]
    assert entry["abstain"] is False
    assert abs(sum(entry["distribution"].values()) - 1.0) < 1e-3
    assert max(entry["distribution"], key=entry["distribution"].get) == entry["class"]


def test_the_extended_class_and_the_lossy_flag_travel_with_the_legacy_class(registry) -> None:
    source = (
        "def is_prime(n):\n    i = 2\n    while i * i <= n:\n"
        "        if n % i == 0:\n            return False\n        i += 1\n    return True\n"
    )
    result = predict(registry, source, "python")["time"]
    assert result["class"] == "O(n)"  # the legacy field rounds up
    assert result["extended_class"] == "O(sqrt n)"
    assert result["expression"] == "O(sqrt n)"
    assert result["projection_lossy"] is True


def test_a_multivariable_expression_survives_the_class_projection(registry) -> None:
    source = "def f(a, b):\n    for x in a:\n        for y in b:\n            pass\n"
    result = predict(registry, source, "python")["time"]
    assert result["expression"] == "O(n * m)"
    assert result["class"] == "O(n^2)"


def test_the_derivation_is_served_with_one_based_lines(registry) -> None:
    source = _source("quad_two_sum_bruteforce", "python")
    result = predict(registry, source, "python")
    loops = [s for s in result["derivation"] if s["kind"] == "loop"]
    assert len(loops) == 2 and all(isinstance(s["line"], int) and s["line"] >= 1 for s in loops)
    assert result["derivation"][-2]["kind"] == result["derivation"][-1]["kind"] == "total"
    assert result["engine"] == "symbolic"
    assert result["entry"]


def test_attribution_of_a_symbolic_answer_points_at_the_loops_that_cost(registry) -> None:
    """The code panel highlights `attribution[].spans` (0-based line, col, line, col) and shows the
    feature as the tooltip: for a symbolic answer that is the derivation step on that line."""
    source = _source("quad_two_sum_bruteforce", "python")
    result = predict(registry, source, "python")
    lines = source.splitlines()
    loop_lines = {s["line"] for s in result["derivation"] if s["kind"] == "loop"}
    covered = set()
    for item in result["attribution"]:
        assert "loop" in item["feature"]
        for start_line, start_col, end_line, end_col in item["spans"]:
            assert start_line == end_line and 0 <= start_col < end_col <= len(lines[start_line])
            covered.add(start_line + 1)
    assert covered == loop_lines


# ---------------------------------------------------------------- the fallback
_UNPROVABLE = """\
def permutations(nums):
    out = []

    def go(current, used):
        if len(current) == len(nums):
            out.append(current[:])
            return
        for i in range(len(nums)):
            if i in used:
                continue
            used.add(i)
            current.append(nums[i])
            go(current, used)
            current.pop()
            used.discard(i)

    go([], set())
    return out
"""


def test_an_unbounded_recursion_falls_back_to_the_model_with_a_warning(registry) -> None:
    result = predict(registry, _UNPROVABLE, "python")
    assert result["engine"] == "ml_fallback"
    for dimension in ("time", "space"):
        entry = result[dimension]
        assert entry["engine"] == "ml_fallback"
        assert entry["certainty"] is None and entry["expression"] is None
        assert entry["conformal_coverage"] == pytest.approx(0.9)
        assert entry["class"] in entry["conformal_set"]
    assert any("fallback" in warning for warning in result["warnings"])
    assert result["derivation"] == []


def test_an_engine_crash_never_breaks_the_endpoint(registry, monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("engine bug")

    monkeypatch.setattr(predict_module, "analyze", boom)
    result = predict(registry, _source("quad_two_sum_bruteforce", "python"), "python")
    assert result["engine"] == "ml_fallback"
    assert any("RuntimeError" in warning for warning in result["warnings"])


def test_ml_mode_is_the_model_alone(registry, monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the engine must not run in ml mode")

    monkeypatch.setattr(predict_module, "analyze", boom)
    result = predict(registry, _source("quad_two_sum_bruteforce", "python"), "python", mode="ml")
    assert result["engine"] == "ml"
    assert result["time"]["engine"] == "ml"
    assert result["derivation"] == []
    assert result["time"]["conformal_coverage"] == pytest.approx(0.9)


def test_an_unknown_mode_is_rejected(registry) -> None:
    with pytest.raises(ValueError):
        predict(registry, "def f():\n    pass\n", "python", mode="oracle")
