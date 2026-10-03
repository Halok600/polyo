"""The symbolic engine against the golden suite: the ratchet and the phase gates.

`eval/golden_baseline.json` lists every (case, language) the engine gets right. A change may add to
it (and must say so by updating the file) but may not lose anything, so a fix that trades one case
for another cannot hide inside an unchanged percentage. The gate tests restate the numbers the plan
commits to for each phase; they are measured, never relaxed.
"""

from __future__ import annotations

import pytest

from eval.golden_baseline import improvements, load_baseline, passing, regressions
from eval.golden_suite import ALL_LANGUAGES, load_cases
from eval.predictors import get_predictor
from eval.scorecard import invariance_report, score_predictor, summarize

_CASES = load_cases()
_NON_RECURSIVE = [c for c in _CASES if c.group != "recursion" and "recursion" not in c.tags]


@pytest.fixture(scope="module")
def current() -> dict[str, list[str]]:
    return passing("symbolic")


def test_nothing_the_baseline_got_right_is_wrong_now(current: dict[str, list[str]]) -> None:
    lost = regressions(current, load_baseline())
    assert not lost, f"golden regressions: {lost}"


def test_the_baseline_records_every_improvement(current: dict[str, list[str]]) -> None:
    gained = improvements(current, load_baseline())
    assert not gained, (
        "the engine got more golden cases right; record them with "
        f"`python -m eval.golden_baseline --update`: {gained}"
    )


# ---------------------------------------------------------------- phase 2: the non-recursive subset
@pytest.fixture(scope="module")
def non_recursive_summary():  # noqa: ANN201 -- eval.scorecard.Summary
    results = score_predictor(get_predictor("symbolic"), _NON_RECURSIVE)
    return summarize(results, _NON_RECURSIVE)


@pytest.mark.parametrize("language", ALL_LANGUAGES)
def test_phase2_gate_time_accuracy_per_language(non_recursive_summary, language: str) -> None:  # noqa: ANN001
    assert non_recursive_summary.per_language[language].time_accuracy >= 0.95


@pytest.mark.parametrize("language", ALL_LANGUAGES)
def test_phase2_gate_space_accuracy_per_language(non_recursive_summary, language: str) -> None:  # noqa: ANN001
    assert non_recursive_summary.per_language[language].space_accuracy >= 0.90


def test_phase2_gate_cross_language_consistency(non_recursive_summary) -> None:  # noqa: ANN001
    assert non_recursive_summary.cross_language_consistency == 1.0


def test_the_engine_never_crashes_on_a_golden_source(non_recursive_summary) -> None:  # noqa: ANN001
    assert non_recursive_summary.errors == 0


def test_phase2_gate_perturbation_invariance() -> None:
    report = invariance_report(get_predictor("symbolic"), _NON_RECURSIVE)
    assert report.overall_rate == 1.0, report.flips[:10]


# ------------------------------------------------------------------- phase 3: the recursive subset
_RECURSIVE = [c for c in _CASES if c.group == "recursion" or "recursion" in c.tags]


@pytest.fixture(scope="module")
def recursive_summary():  # noqa: ANN201 -- eval.scorecard.Summary
    results = score_predictor(get_predictor("symbolic"), _RECURSIVE)
    return summarize(results, _RECURSIVE)


@pytest.mark.parametrize("language", ALL_LANGUAGES)
def test_phase3_gate_recursive_time_accuracy_per_language(recursive_summary, language: str) -> None:  # noqa: ANN001
    assert recursive_summary.per_language[language].time_accuracy >= 0.90


@pytest.mark.parametrize("language", ALL_LANGUAGES)
def test_phase4_gate_space_accuracy_per_language(recursive_summary, language: str) -> None:  # noqa: ANN001
    assert recursive_summary.per_language[language].space_accuracy >= 0.90


def test_whole_suite_is_cross_language_consistent() -> None:
    results = score_predictor(get_predictor("symbolic"), _CASES)
    assert summarize(results, _CASES).cross_language_consistency == 1.0


def test_phase3_gate_recursive_perturbation_invariance() -> None:
    report = invariance_report(get_predictor("symbolic"), _RECURSIVE)
    assert report.overall_rate == 1.0, report.flips[:10]
