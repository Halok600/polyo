"""Permanent regressions for the 2026-10-01 diagnosis of the served model.

User report: pasting O(n^2) code returned O(n^3). Measured against the real served artifacts:
20 of 28 curated golden (case, language) pairs were wrong -- every non-Python nested loop was
over-predicted, space was often inverted, and 40 dead statements flipped a correct answer.

Those 20 pairs were `xfail(strict=True)` while the GNN was the product. Since PolyO v2 phase 5 the
product is the hybrid (symbolic engine first, GNN only when the engine says "unknown"), the strict
markers did their job -- XPASS forced their removal -- and every pair is now an ordinary assertion
about what the API answers. What stays on record is the historical fact itself: the GNN ALONE
(`gnn` predictor, `mode="ml"`) still gets those same pairs wrong, which is why it is no longer in
front.
"""
from __future__ import annotations

import pytest

from eval.golden_suite import load_cases
from eval.perturb import add_dead_code
from eval.predictors import get_predictor
from features.tabular import extract_features
from parsing.normalize import normalize_source

_CASES = {case.id: case for case in load_cases()}

# The pairs the served GNN got wrong on 2026-10-01 (the diagnosis), then the ones it got right.
_WRONG_FOR_THE_GNN = {
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
_RIGHT_FOR_THE_GNN = {
    ("quad_two_sum_bruteforce", "go"),
    ("program_two_sum_with_main_harness", "python"),
    ("program_two_sum_with_main_harness", "java"),
    ("program_two_sum_with_main_harness", "go"),
    ("linear_sum", "java"),
    ("nlogn_sorted_unique", "cpp"),
    ("rec_fib_naive", "python"),
    ("rec_fib_naive", "cpp"),
}
_ALL_PAIRS = sorted(_WRONG_FOR_THE_GNN | _RIGHT_FOR_THE_GNN)


def _ids(pairs: list[tuple[str, str]]) -> list[str]:
    return [f"{case_id}-{language}" for case_id, language in pairs]


@pytest.fixture(scope="module")
def product():
    predictor = get_predictor("product")
    probe = predictor("def f(x):\n    return x\n", "python")
    if probe.error and "ModelsNotTrainedError" in probe.error:
        pytest.skip("model artifacts are not present")
    return predictor


@pytest.mark.parametrize(("case_id", "language"), _ALL_PAIRS, ids=_ids(_ALL_PAIRS))
def test_product_answer_matches_the_golden_label(product, case_id: str, language: str) -> None:
    case = _CASES[case_id]
    expected = case.expected(language)
    got = product(case.sources[language], language)
    assert got.error is None, got.error
    assert (got.time_class, got.space_class) == (expected.time_class, expected.space_class)
    assert got.engine == "symbolic"


_TWO_SUM = _CASES["quad_two_sum_bruteforce"].sources["python"]


@pytest.mark.parametrize("padding", [0, 5, 20, 40, 80])
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


def _score(name: str, pairs: list[tuple[str, str]]) -> int:
    predictor = get_predictor(name)
    hits = 0
    for case_id, language in pairs:
        case = _CASES[case_id]
        expected = case.expected(language)
        got = predictor(case.sources[language], language)
        hits += (got.time_class, got.space_class) == (expected.time_class, expected.space_class)
    return hits


def test_the_gnn_alone_still_fails_where_it_failed_and_the_hybrid_does_not(product) -> None:
    """The unflattering fact that started the rebuild, kept as an executable statement: on the
    pairs the served model got wrong, the GNN alone is still wrong on most, a ten-line rule on loop
    depth beats it on time, and the hybrid that replaced it gets all of them."""
    wrong = sorted(_WRONG_FOR_THE_GNN)
    gnn_hits = _score("gnn", wrong)
    hybrid_hits = _score("product", wrong)
    assert hybrid_hits == len(wrong)
    assert gnn_hits < len(wrong) // 2, "the GNN alone no longer fails: revisit the promotion rule"


def test_the_rule_baseline_beats_the_gnn_alone_on_the_originally_reported_cases() -> None:
    """On the reported nested-loop cases a lookup on loop depth is right where the GNN is wrong."""
    rule = get_predictor("rule")
    gnn = get_predictor("gnn")
    reported = [("quad_two_sum_bruteforce", lang) for lang in ("cpp", "java", "javascript")]
    reported += [("quad_bubble_sort", "java"), ("linear_sum", "cpp"), ("linear_sum", "go")]
    rule_hits = gnn_hits = 0
    for case_id, language in reported:
        case = _CASES[case_id]
        want = case.expected(language).time_class
        rule_hits += rule(case.sources[language], language).time_class == want
        gnn_hits += gnn(case.sources[language], language).time_class == want
    assert rule_hits == len(reported)
    assert gnn_hits < rule_hits
