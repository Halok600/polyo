"""The comparison's statistics and its promotion rule: pure functions, tested on made-up rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.compare import (
    INTRO,
    MIN_ROWS,
    READING,
    ModelAnswer,
    Row,
    Tally,
    calibration_summary,
    model_answers,
    padding_survival,
    promotion,
    render,
    tally,
    wilson,
)
from eval.predictors import Prediction

_ROOT = Path(__file__).resolve().parent.parent


def _row(
    case_id: str,
    label: str,
    engine: str | None,
    model: str | None,
    language: str = "python",
    certainty: str = "certain",
    label_space: str | None = None,
) -> Row:
    return Row(
        "unit",
        case_id,
        language,
        label,
        label_space,
        {"symbolic": (engine, None), "gnn": (model, None)},
        certainty=certainty,
    )


def test_wilson_is_the_whole_range_with_no_data() -> None:
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_stays_inside_zero_and_one_at_the_extremes() -> None:
    low, high = wilson(10, 10)
    assert 0.6 < low < 1.0 and high == pytest.approx(1.0)
    low, high = wilson(0, 10)
    assert low == pytest.approx(0.0) and 0.0 < high < 0.4


def test_wilson_narrows_with_more_data() -> None:
    narrow = wilson(500, 1000)
    wide = wilson(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert narrow[0] < 0.5 < narrow[1]


def test_a_missing_answer_counts_as_wrong_and_a_missing_label_is_skipped() -> None:
    rows = [
        _row("a", "O(n)", "O(n)", None),
        _row("b", "O(n)", None, "O(n)"),
        Row("unit", "c", "python", None, None, {"symbolic": ("O(n)", None)}),
    ]
    engine = tally(rows, "symbolic", "time")["all"]
    model = tally(rows, "gnn", "time")["all"]
    assert (engine.correct, engine.total) == (1, 2)
    assert (model.correct, model.total) == (1, 2)


def test_tally_groups_by_the_key() -> None:
    rows = [_row("a", "O(n)", "O(n)", "O(1)"), _row("b", "O(n)", "O(1)", "O(n)", language="java")]
    by_language = tally(rows, "symbolic", "time", lambda r: r.language)
    assert by_language["python"].accuracy == 1.0 and by_language["java"].accuracy == 0.0


def _rows(engine_right: int, model_right: int, total: int, **kwargs: str) -> list[Row]:
    return [
        _row(
            f"r{i}",
            "O(n)",
            "O(n)" if i < engine_right else "O(1)",
            "O(n)" if i < model_right else "O(1)",
            **kwargs,
        )
        for i in range(total)
    ]


def test_the_model_is_promoted_only_when_it_is_clearly_better() -> None:
    decision = promotion(_rows(engine_right=20, model_right=39, total=40))["python/certain"]["time"]
    assert decision["answers"] == "model"


def test_a_model_that_is_only_somewhat_better_is_not_promoted() -> None:
    decision = promotion(_rows(engine_right=30, model_right=36, total=40))["python/certain"]["time"]
    assert decision["answers"] == "engine"
    assert "noise" in str(decision["reason"])


def test_a_slice_with_too_few_rows_stays_with_the_engine() -> None:
    rows = _rows(engine_right=0, model_right=MIN_ROWS - 1, total=MIN_ROWS - 1)
    decision = promotion(rows)["python/certain"]["time"]
    assert decision["answers"] == "engine" and "fewer than" in str(decision["reason"])


def test_slices_are_judged_separately() -> None:
    rows = _rows(20, 39, 40, language="python") + _rows(39, 20, 40, language="java")
    decisions = promotion(rows)
    assert decisions["python/certain"]["time"]["answers"] == "model"  # type: ignore[index]
    assert decisions["java/certain"]["time"]["answers"] == "engine"  # type: ignore[index]


def test_time_and_space_are_judged_separately() -> None:
    rows = [
        Row(
            "unit",
            f"r{i}",
            "python",
            "O(n)",
            "O(n)",
            {"symbolic": ("O(n)", "O(1)"), "gnn": ("O(1)", "O(n)")},
            certainty="certain",
        )
        for i in range(40)
    ]
    decisions = promotion(rows)["python/certain"]
    assert decisions["time"]["answers"] == "engine"  # type: ignore[index]
    assert decisions["space"]["answers"] == "model"  # type: ignore[index]


def test_the_report_names_every_predictor_and_the_promotion_table() -> None:
    rows = _rows(30, 36, 40)
    text = render({"unit": rows}, promotion(rows))
    assert "symbolic" in text and "gnn" in text and "Promotion" in text
    assert "python/certain" in text


def test_an_empty_tally_has_no_accuracy() -> None:
    assert Tally().accuracy is None


def test_the_committed_promotion_record_promotes_no_slice() -> None:
    """The served hybrid answers with the engine and consults the model only where the engine has
    no answer. `eval/promotion.json` is the evidence for that: if a re-run ever promotes a slice,
    the serving path (`api/predict.py`) has to route that slice to the model before the record
    changes."""
    record = json.loads((_ROOT / "eval" / "promotion.json").read_text(encoding="utf-8"))
    assert record, "the promotion record is missing or empty"
    promoted = sorted(
        f"{slice_key}/{dimension}"
        for slice_key, dimensions in record.items()
        for dimension, decision in dimensions.items()
        if decision["answers"] != "engine"
    )
    assert promoted == []


# ------------------------------------------------------------------------- padding invariance
_SUM = "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n"


def _structural(source: str, language: str) -> Prediction:
    return Prediction("O(n)", "O(1)")


def _counts_lines(source: str, language: str) -> Prediction:
    """Reads size, not structure: a long program looks quadratic."""
    return Prediction("O(n^2)" if len(source.splitlines()) > 30 else "O(n)", "O(1)")


def test_padding_survival_tells_a_structural_predictor_from_one_that_counts_lines() -> None:
    survival = padding_survival(
        [(_SUM, "python")] * 3,
        {"structural": _structural, "counts_lines": _counts_lines},
        (10, 40),
    )
    assert survival["structural"] == {10: (3, 3), 40: (3, 3)}
    assert survival["counts_lines"] == {10: (3, 3), 40: (0, 3)}


def test_a_program_that_cannot_be_padded_is_not_counted() -> None:
    survival = padding_survival([(_SUM, "python"), ("x", "cobol")], {"s": _structural}, (10,))
    assert survival["s"] == {10: (1, 1)}


def test_a_predictor_with_no_answer_on_the_original_is_not_counted() -> None:
    def failing(source: str, language: str) -> Prediction:
        return Prediction(None, None, error="boom")

    assert padding_survival([(_SUM, "python")], {"f": failing}, (10,))["f"] == {10: (0, 0)}


def test_the_report_has_the_invariance_table_when_given_one() -> None:
    rows = _rows(30, 36, 40)
    invariance = {
        "symbolic": {10: (144, 144), 40: (144, 144)},
        "gnn": {10: (48, 144), 40: (32, 144)},
    }
    text = render({"unit": rows}, promotion(rows), invariance)
    assert "unchanged by dead code" in text
    assert "144/144 (100%)" in text and "32/144 (22%)" in text


def test_the_report_opens_with_a_whole_sentence() -> None:
    rows = _rows(30, 36, 40)
    intro = render({"unit": rows}, promotion(rows)).splitlines()[2]
    assert intro == INTRO and intro.endswith(".")


def test_the_report_says_which_rows_are_a_measurement_and_which_a_regression_net() -> None:
    rows = _rows(30, 36, 40)
    text = render({"unit": rows}, promotion(rows))
    assert READING in text
    assert "first-contact" in READING and "regression net" in READING and "product" in READING


# ------------------------------------------------------------------ the model's own confidence
def test_the_calibration_of_a_model_that_means_what_it_says_is_zero() -> None:
    answers = [ModelAnswer(i < 5, 0.5, True, 2, False) for i in range(10)]
    summary = calibration_summary(answers)
    assert summary["accuracy"] == 0.5 and summary["mean_confidence"] == 0.5
    assert summary["ece"] == pytest.approx(0.0)


def test_an_overconfident_model_has_a_large_ece_and_no_coverage() -> None:
    summary = calibration_summary([ModelAnswer(False, 0.9, False, 1, False)] * 4)
    assert summary["ece"] == pytest.approx(0.9)
    assert summary["coverage"] == 0.0


def test_the_calibration_summary_reports_set_size_and_abstentions() -> None:
    summary = calibration_summary(
        [ModelAnswer(True, 0.6, True, 2, False), ModelAnswer(False, 0.4, True, 4, True)]
    )
    assert summary["mean_set_size"] == 3.0
    assert summary["abstain_share"] == 0.5
    assert summary["coverage"] == 1.0
    assert summary["n"] == 2


def test_the_calibration_of_no_answers_is_empty_not_an_error() -> None:
    assert calibration_summary([])["n"] == 0


def test_model_answers_runs_the_gnn_alone_and_keeps_what_the_api_says(
    tiny_registry, monkeypatch
) -> None:
    from eval import predictors

    monkeypatch.setattr(predictors, "_registry", lambda: tiny_registry)
    answers = model_answers([(_SUM, "python", "O(n)", "O(1)"), (_SUM, "python", None, "O(1)")])
    assert len(answers["time"]) == 1 and len(answers["space"]) == 2  # no label, no row
    first = answers["time"][0]
    assert 0.0 <= first.confidence <= 1.0 and first.set_size >= 1


def test_the_report_has_the_model_calibration_table_when_given_one() -> None:
    rows = _rows(30, 36, 40)
    summary = {"n": 144, "accuracy": 0.39, "mean_confidence": 0.68, "ece": 0.328,
               "coverage": 0.951, "mean_set_size": 4.2, "abstain_share": 0.73}  # fmt: skip
    text = render({"unit": rows}, promotion(rows), calibration={"blind_v1": {"space": summary}})
    assert "confidence" in text and "0.328" in text and "95%" in text
