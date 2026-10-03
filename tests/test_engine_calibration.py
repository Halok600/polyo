"""The confidence served with a symbolic answer is a MEASURED accuracy, never a hard-coded 1.0.

`analysis/certainty_calibration.json` records, per certainty level and per dimension, how many
blind (first-contact) programs the engine got right. `confidence` is the Laplace-smoothed accuracy
`(correct + 1) / (total + 2)`, so it is never 0 or 1 however lucky the sample was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.calibration import (
    CALIBRATION_PATH,
    CalibrationError,
    LevelStats,
    SymbolicCalibration,
    load_calibration,
)


def test_accuracy_is_laplace_smoothed_and_never_one() -> None:
    assert LevelStats(correct=9, total=10).accuracy == pytest.approx(10 / 12)
    assert LevelStats(correct=100, total=100).accuracy < 1.0
    assert LevelStats(correct=0, total=50).accuracy > 0.0


def test_no_data_is_a_coin_flip_not_a_claim() -> None:
    assert LevelStats(correct=0, total=0).accuracy == 0.5


def test_the_committed_calibration_covers_both_served_levels_and_dimensions() -> None:
    calibration = load_calibration()
    for certainty in ("certain", "assumed"):
        for dimension in ("time", "space"):
            stats = calibration.stats(certainty, dimension)
            assert stats.total > 0, f"no blind cases measured for {certainty}/{dimension}"
            assert 0 <= stats.correct <= stats.total
            assert 0.0 < calibration.confidence(certainty, dimension) < 1.0


def test_a_less_certain_answer_is_never_reported_as_more_confident() -> None:
    calibration = load_calibration()
    for dimension in ("time", "space"):
        assert (
            calibration.confidence("assumed", dimension)
            <= calibration.confidence("certain", dimension) + 1e-9
        )


def test_an_unserved_level_has_no_confidence() -> None:
    with pytest.raises(KeyError):
        load_calibration().confidence("unknown", "time")


def test_a_missing_file_is_a_loud_deployment_error(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError):
        load_calibration(tmp_path / "nope.json")


def test_a_malformed_file_is_a_loud_deployment_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"levels": {"certain": {"time": {"correct": 5}}}}), "utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(bad)


def test_counts_round_trip_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    levels = {
        "certain": {"time": {"correct": 9, "total": 10}, "space": {"correct": 8, "total": 10}},
        "assumed": {"time": {"correct": 1, "total": 2}, "space": {"correct": 2, "total": 2}},
    }
    path.write_text(json.dumps({"source": "unit test", "levels": levels}), "utf-8")
    calibration = load_calibration(path)
    assert isinstance(calibration, SymbolicCalibration)
    assert calibration.stats("certain", "time") == LevelStats(9, 10)
    assert calibration.source == "unit test"


def test_the_committed_file_names_its_evidence() -> None:
    payload = json.loads(CALIBRATION_PATH.read_text("utf-8"))
    assert payload["source"]
    assert "PROVISIONAL" not in payload["source"], "replace the placeholder with measured counts"
    assert payload["levels"]
