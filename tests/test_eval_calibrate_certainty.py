"""The calibration script turns first-contact answers on blind programs into confidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval.calibrate_certainty as calibrate
from analysis.calibration import CALIBRATION_PATH, LevelStats, load_calibration


def _case(case_id: str, time_class: str, space_class: str, language: str = "python") -> dict:
    return {
        "id": case_id,
        "language": language,
        "source": "def f(a):\n    for x in a:\n        pass\n",
        "time_class": time_class,
        "space_class": space_class,
    }


def _answer(time_class: str, space_class: str, certainty: str) -> dict[str, str]:
    return {"time_class": time_class, "space_class": space_class, "certainty": certainty}


def test_score_counts_correct_answers_per_level_and_dimension() -> None:
    cases = [
        _case("a", "O(n)", "O(1)"),
        _case("b", "O(n^2)", "O(n)"),
        _case("c", "O(n)", "O(1)"),
        _case("d", "O(n)", "O(1)"),
    ]
    record = {
        "a": _answer("O(n)", "O(1)", "certain"),  # right, right
        "b": _answer("O(n)", "O(n)", "certain"),  # time wrong, space right
        "c": _answer("O(n)", "O(n)", "assumed"),  # time right, space wrong
        "d": _answer("O(n^3)", "O(n)", "unknown"),  # goes to the model: not scored
    }
    levels = calibrate.score(cases, record)
    assert levels["certain"]["time"] == LevelStats(1, 2)
    assert levels["certain"]["space"] == LevelStats(2, 2)
    assert levels["assumed"]["time"] == LevelStats(1, 1)
    assert levels["assumed"]["space"] == LevelStats(0, 1)


def test_a_crash_or_a_missing_answer_is_not_a_served_level() -> None:
    cases = [_case("a", "O(n)", "O(1)"), _case("b", "O(n)", "O(1)")]
    record = {"a": {"error": "ValueError: x", "certainty": "error"}}
    levels = calibrate.score(cases, record)
    assert all(s.total == 0 for per in levels.values() for s in per.values())


def test_the_payload_is_a_valid_calibration_file(tmp_path: Path) -> None:
    cases = [_case("a", "O(n)", "O(1)"), _case("b", "O(n)", "O(1)")]
    record = {
        "a": _answer("O(n)", "O(1)", "certain"),
        "b": _answer("O(n)", "O(1)", "assumed"),
    }
    payload = calibrate.build_payload(cases, record)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(payload), "utf-8")
    loaded = load_calibration(path)
    assert loaded.stats("certain", "time") == LevelStats(1, 1)
    assert loaded.source == calibrate.SOURCE_NAME
    assert payload["certainty_counts"] == {"assumed": 1, "certain": 1}
    assert payload["joint_accuracy_by_language"] == {"python": {"correct": 2, "total": 2}}


def test_first_contact_records_the_engine_answer_and_its_certainty() -> None:
    record = calibrate.first_contact([_case("a", "O(n)", "O(1)")])
    assert record["a"] == {
        "time_class": "O(n)",
        "space_class": "O(1)",
        "time_expr": "O(n)",
        "space_expr": "O(1)",
        "certainty": "certain",
    }


def test_first_contact_survives_a_program_the_engine_cannot_lower() -> None:
    bad = _case("bad", "O(1)", "O(1)")
    bad["source"] = "def f(:\n"
    record = calibrate.first_contact([bad])
    assert record["bad"]["certainty"] == "error" and "error" in record["bad"]


def test_first_contact_happens_once(tmp_path: Path, monkeypatch, capsys) -> None:
    cases = tmp_path / "cases.json"
    record = tmp_path / "record.json"
    cases.write_text(json.dumps([_case("a", "O(n)", "O(1)")]), "utf-8")
    record.write_text("{}", "utf-8")
    monkeypatch.setattr(calibrate, "CASES_PATH", cases)
    monkeypatch.setattr(calibrate, "RECORD_PATH", record)
    assert calibrate.main(["record"]) == 1
    assert record.read_text("utf-8") == "{}"  # untouched
    assert calibrate.main(["record", "--force"]) == 0
    assert json.loads(record.read_text("utf-8"))["a"]["certainty"] == "certain"


@pytest.mark.skipif(
    not (calibrate.CASES_PATH.exists() and calibrate.RECORD_PATH.exists()),
    reason="the blind set has not been recorded yet",
)
def test_the_committed_calibration_is_reproducible_from_the_recorded_evidence() -> None:
    cases = json.loads(calibrate.CASES_PATH.read_text("utf-8"))
    record = json.loads(calibrate.RECORD_PATH.read_text("utf-8"))
    committed = json.loads(CALIBRATION_PATH.read_text("utf-8"))
    assert calibrate.build_payload(cases, record) == committed
    assert set(record) == {case["id"] for case in cases}
