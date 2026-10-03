"""Permanent regressions for the 2026-10-01 diagnosis of the served model.

User report: pasting O(n^2) code returned O(n^3). Measured against the real served artifacts:
20 of 28 curated golden (case, language) pairs were wrong -- every non-Python nested loop was
over-predicted, space was often inverted, and 40 dead statements flipped a correct answer.

The 20 known-wrong pairs are `xfail(strict=True)`: they document today's behaviour, and the moment
the hybrid engine (PolyO v2 Phase 5) fixes one, XPASS fails the suite and forces the marker to be
removed. The 8 pairs that are right today are plain tests so they can never regress.
"""
from __future__ import annotations

import pytest

from eval.golden_suite import load_cases
from eval.perturb import add_dead_code
from eval.predictors import get_predictor
from features.tabular import extract_features
from parsing.normalize import normalize_source

_CASES = {case.id: case for case in load_cases()}

_STILL_WRONG = {
    ("quad_two_sum_bruteforce", "python"),
    ("quad_two_sum_bruteforce", "cpp"),
    ("quad_two_sum_bruteforce", "java"),
    ("quad_two_sum_bruteforce", "javascript"),
    ("program_two_sum_with_main_harness", "cpp"),
    ("program_two_sum_with_main_harness", "javascript"),
    ("quad_bubble_sort", "java"),
    ("quad_bubble_sort", "cpp"),
    ("quad_bubble_sort", "javascript"),
    ("linear_sum", "cpp"),
    ("linear_sum", "go"),
    ("cubic_triple_loop", "python"),
    ("cubic_triple_loop", "cpp"),
    ("log_binary_search_iter", "python"),
    ("log_binary_search_iter", "cpp"),
    ("nlogn_sorted_unique", "python"),
    ("space_square_table", "python"),
    ("space_square_table", "java"),
    ("hash_two_sum", "python"),
    ("hash_two_sum", "cpp"),
}
_ALREADY_RIGHT = {
    ("quad_two_sum_bruteforce", "go"),
    ("program_two_sum_with_main_harness", "python"),
    ("program_two_sum_with_main_harness", "java"),
    ("program_two_sum_with_main_harness", "go"),
    ("linear_sum", "java"),
    ("nlogn_sorted_unique", "cpp"),
    ("rec_fib_naive", "python"),
    ("rec_fib_naive", "cpp"),
}
_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="PolyO v2: the served GNN reads graph size, not loop nesting (Phase 5 fixes)",
)


def _params() -> list:
    out = []
    for pair in sorted(_STILL_WRONG | _ALREADY_RIGHT):
        marks = [_XFAIL] if pair in _STILL_WRONG else []
        out.append(pytest.param(*pair, id=f"{pair[0]}-{pair[1]}", marks=marks))
    return out


@pytest.fixture(scope="module")
def product():
    predictor = get_predictor("product")
    probe = predictor("def f(x):\n    return x\n", "python")
    if probe.error and "ModelsNotTrainedError" in probe.error:
        pytest.skip("model artifacts are not present")
    return predictor


@pytest.mark.parametrize(("case_id", "language"), _params())
def test_product_answer_matches_the_golden_label(product, case_id: str, language: str) -> None:
    case = _CASES[case_id]
    expected = case.expected(language)
    got = product(case.sources[language], language)
    assert got.error is None, got.error
    assert (got.time_class, got.space_class) == (expected.time_class, expected.space_class)


_TWO_SUM = _CASES["quad_two_sum_bruteforce"].sources["python"]


@pytest.mark.parametrize(
    "padding",
    [
        0,
        5,
        pytest.param(20, marks=_XFAIL),
        pytest.param(40, marks=_XFAIL),
        pytest.param(80, marks=_XFAIL),
    ],
)
def test_dead_code_does_not_change_the_product_answer(product, padding: int) -> None:
    """The original padding experiment: pure O(1) statements cannot change a complexity class."""
    padded = add_dead_code(_TWO_SUM, "python", k=padding) if padding else _TWO_SUM
    assert padded is not None
    assert product(padded, "python").time_class == "O(n^2)"


@pytest.mark.parametrize("language", ["python", "cpp", "java", "javascript", "go"])
def test_the_ir_sees_the_same_loop_nest_in_every_language(language: str) -> None:
    """The IR itself was never the problem: two-sum has loop depth 2 in every language. This pins
    that premise, which the symbolic engine relies on and the GNN squandered."""
    case = _CASES["quad_two_sum_bruteforce"]
    features = extract_features(normalize_source(case.sources[language], language))
    assert features.max_loop_nesting_depth == 2
    assert features.loop_count == 2


def test_the_rule_baseline_beats_the_product_on_the_originally_reported_cases() -> None:
    """The unflattering fact that started the rebuild, kept as an executable statement: on the
    reported cases a lookup on loop depth is right where the served model is wrong."""
    rule = get_predictor("rule")
    reported = [("quad_two_sum_bruteforce", lang) for lang in ("cpp", "java", "javascript", "go")]
    reported += [("quad_bubble_sort", "java"), ("linear_sum", "cpp"), ("linear_sum", "go")]
    for case_id, language in reported:
        case = _CASES[case_id]
        got = rule(case.sources[language], language).time_class
        assert got == case.expected(language).time_class, (case_id, language)
