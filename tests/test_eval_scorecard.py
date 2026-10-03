"""Scorecard arithmetic (`eval/scorecard.py`), checked with scripted predictors.

The old headline was one pooled macro-F1 over a 96.6%-Python test set, which could not see the
failure users hit (C++/Java/JS nested loops). The scorecard reports per-language and per-class
numbers, cross-language consistency, and perturbation invariance, and keeps known-hard cases out of
the gated totals. These tests pin that arithmetic."""
from __future__ import annotations

import pytest

from eval.golden_suite import GoldenCase
from eval.predictors import Prediction
from eval.scorecard import (
    invariance_report,
    normalize_expr,
    render_markdown,
    score_predictor,
    summarize,
)


def _case(case_id: str, time_class: str, space_class: str = "O(1)", tags=()) -> GoldenCase:
    source = "def f(x):\n    return x\n"
    return GoldenCase(
        id=case_id,
        title=case_id,
        group="t",
        tags=tuple(tags),
        time_class=time_class,
        space_class=space_class,
        time_expr=f"{time_class}",
        space_expr=f"{space_class}",
        rationale="r",
        sources=dict.fromkeys(("python", "cpp", "java", "javascript", "go"), source),
    )


def _script(table: dict[tuple[str, str], tuple[str, str]]):
    """A predictor answering from a (case marker in source, language) table."""

    def predictor(source: str, language: str) -> Prediction:
        marker = source.splitlines()[0]
        time_class, space_class = table[(marker, language)]
        return Prediction(time_class, space_class, engine="script")

    return predictor


def _marked(case: GoldenCase, marker: str) -> GoldenCase:
    sources = {lang: f"# {marker}\n{src}" for lang, src in case.sources.items()}
    return GoldenCase(
        id=case.id, title=case.title, group=case.group, tags=case.tags,
        time_class=case.time_class, space_class=case.space_class, time_expr=case.time_expr,
        space_expr=case.space_expr, rationale=case.rationale, sources=sources,
    )  # fmt: skip


def test_per_language_accuracy_exposes_a_language_specific_failure() -> None:
    # A predictor that is perfect on Python and wrong on C++ must NOT hide behind a pooled number.
    case = _marked(_case("a", "O(n^2)"), "A")
    table = {("# A", lang): ("O(n^2)", "O(1)") for lang in case.languages}
    table[("# A", "cpp")] = ("O(n^3)", "O(1)")
    results = score_predictor(_script(table), [case])
    summary = summarize(results, [case])
    assert summary.per_language["python"].time_accuracy == 1.0
    assert summary.per_language["cpp"].time_accuracy == 0.0
    assert summary.overall.time_accuracy == 0.8  # 4 of the 5 languages are right


def test_cross_language_consistency_counts_agreement_not_correctness() -> None:
    wrong_but_consistent = _marked(_case("a", "O(n^2)"), "A")
    split = _marked(_case("b", "O(n^2)"), "B")
    table = {("# A", lang): ("O(n)", "O(1)") for lang in wrong_but_consistent.languages}
    table.update({("# B", lang): ("O(n^2)", "O(1)") for lang in split.languages})
    table[("# B", "go")] = ("O(n^3)", "O(1)")
    cases = [wrong_but_consistent, split]
    summary = summarize(score_predictor(_script(table), cases), cases)
    assert summary.cross_language_consistency == 0.5  # case A agrees with itself, case B does not
    assert summary.cross_language_correct == 0.0  # neither is right in every language


def test_known_limitation_cases_are_reported_separately_from_the_gate() -> None:
    easy = _marked(_case("easy", "O(n)"), "E")
    hard = _marked(_case("hard", "O(n)", tags=("known-limitation",)), "H")
    table = {("# E", lang): ("O(n)", "O(1)") for lang in easy.languages}
    table.update({("# H", lang): ("O(n^2)", "O(1)") for lang in hard.languages})
    summary = summarize(score_predictor(_script(table), [easy, hard]), [easy, hard])
    assert summary.overall.time_accuracy == 1.0  # only the gated case counts
    assert summary.known_limitations.time_accuracy == 0.0
    assert summary.known_limitations.n == 5


def test_per_class_recall_shows_which_classes_are_never_predicted() -> None:
    linear = _marked(_case("l", "O(n)"), "L")
    quad = _marked(_case("q", "O(n^2)"), "Q")
    table = {("# L", lang): ("O(n)", "O(1)") for lang in linear.languages}
    table.update({("# Q", lang): ("O(n)", "O(1)") for lang in quad.languages})  # never says n^2
    summary = summarize(score_predictor(_script(table), [linear, quad]), [linear, quad])
    assert summary.per_time_class["O(n)"].time_accuracy == 1.0
    assert summary.per_time_class["O(n^2)"].time_accuracy == 0.0


def test_ordinal_distance_penalises_far_misses_more_than_near_ones() -> None:
    near = _marked(_case("near", "O(n^2)"), "N")
    far = _marked(_case("far", "O(n^2)"), "F")
    table = {("# N", lang): ("O(n^3)", "O(1)") for lang in near.languages}  # one rank off
    table.update({("# F", lang): ("O(1)", "O(1)") for lang in far.languages})  # four ranks off
    near_summary = summarize(score_predictor(_script(table), [near]), [near])
    far_summary = summarize(score_predictor(_script(table), [far]), [far])
    assert near_summary.overall.mean_time_rank_error == 1.0
    assert far_summary.overall.mean_time_rank_error == 4.0


def test_errors_are_counted_and_scored_as_misses() -> None:
    case = _marked(_case("a", "O(n)"), "A")

    def broken(source: str, language: str) -> Prediction:
        return Prediction(None, None, error="boom")

    summary = summarize(score_predictor(broken, [case]), [case])
    assert summary.errors == 5
    assert summary.overall.time_accuracy == 0.0


def test_invariance_report_flags_a_predictor_that_reads_text_length() -> None:
    case = _marked(_case("a", "O(n)"), "A")

    def length_sensitive(source: str, language: str) -> Prediction:
        # Answers O(n) for short sources and O(n^2) for long ones: exactly the GNN's disease.
        return Prediction("O(n)" if len(source) < 60 else "O(n^2)", "O(1)")

    def structural(source: str, language: str) -> Prediction:
        return Prediction("O(n)", "O(1)")

    bad = invariance_report(length_sensitive, [case])
    good = invariance_report(structural, [case])
    assert good.overall_rate == 1.0
    assert bad.overall_rate < 1.0
    assert bad.by_perturbation["dead_code"].rate < 1.0


def test_render_markdown_contains_the_headline_sections() -> None:
    case = _marked(_case("a", "O(n)"), "A")

    def constant(source: str, language: str) -> Prediction:
        return Prediction("O(n)", "O(1)", engine="const")

    summary = summarize(score_predictor(constant, [case]), [case])
    text = render_markdown({"const": summary}, {"const": invariance_report(constant, [case])})
    for needle in ("Per language", "Per time class", "Cross-language", "Perturbation invariance"):
        assert needle in text


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("O(m * n)", "O(n * m)"),
        ("O(n + m)", "O(m + n)"),
        ("O(n log n)", "O(log n * n)"),
        ("O((n + m) * log n)", "O(log n * (m + n))"),
        ("O(n*m)", "O(n * m)"),
        ("O(n^2 log n)", "O(log n * n^2)"),
    ],
)
def test_normalize_expr_ignores_ordering_and_spacing(a: str, b: str) -> None:
    assert normalize_expr(a) == normalize_expr(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("O(n)", "O(n^2)"),
        ("O(n log n)", "O(n)"),
        ("O(m * n)", "O(m + n)"),
        ("O(m log n)", "O(n log m)"),
        ("O(sqrt n)", "O(n)"),
        ("O(2^n)", "O(n!)"),
    ],
)
def test_normalize_expr_keeps_genuinely_different_expressions_apart(a: str, b: str) -> None:
    assert normalize_expr(a) != normalize_expr(b)


def test_expression_accuracy_is_reported_when_a_predictor_gives_expressions() -> None:
    case = _marked(_case("a", "O(n^2)"), "A")
    case = GoldenCase(
        id=case.id, title=case.title, group=case.group, tags=case.tags,
        time_class=case.time_class, space_class=case.space_class, time_expr="O(n * m)",
        space_expr="O(1)", rationale=case.rationale, sources=case.sources,
    )  # fmt: skip

    def with_expr(source: str, language: str) -> Prediction:
        return Prediction("O(n^2)", "O(1)", time_expr="O(m * n)", space_expr="O(1)")

    summary = summarize(score_predictor(with_expr, [case]), [case])
    assert summary.overall.time_expr_accuracy == 1.0
    assert summary.overall.space_expr_accuracy == 1.0
