"""How much to trust a symbolic answer: measured accuracy per certainty level.

The engine is deterministic, so it has no probability of its own. What it does have is a record:
on programs it had never seen (the blind set, scored at first contact, before any fix), how often
was an answer right when it said "certain", and when it said "assumed"? `certainty_calibration.json`
stores those counts per dimension; `confidence` is their Laplace-smoothed ratio, so a lucky sample
can never produce a claim of 100% (nor 0%).

The counts are written by `python -m eval.calibrate_certainty` and committed. They are NOT
recomputed against the current engine at serve time: an engine tuned on the blind set would only
measure how well it memorised it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_PATH = Path(__file__).with_name("certainty_calibration.json")
SERVED_LEVELS = ("certain", "assumed")  # "unknown" is not served: it falls back to the model
DIMENSIONS = ("time", "space")


class CalibrationError(RuntimeError):
    """The calibration file is missing or malformed: a deployment problem, not a bad request."""


@dataclass(frozen=True)
class LevelStats:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        """`(correct + 1) / (total + 2)`: the rule of succession. With no data it is 0.5."""
        return (self.correct + 1) / (self.total + 2)


@dataclass(frozen=True)
class SymbolicCalibration:
    levels: dict[str, dict[str, LevelStats]]
    source: str = ""

    def stats(self, certainty: str, dimension: str) -> LevelStats:
        return self.levels[certainty][dimension]

    def confidence(self, certainty: str, dimension: str) -> float:
        return self.stats(certainty, dimension).accuracy


def _parse(payload: object) -> SymbolicCalibration:
    if not isinstance(payload, dict) or not isinstance(payload.get("levels"), dict):
        raise CalibrationError("certainty calibration has no `levels`")
    levels: dict[str, dict[str, LevelStats]] = {}
    for certainty in SERVED_LEVELS:
        entry = payload["levels"].get(certainty)
        if not isinstance(entry, dict):
            raise CalibrationError(f"certainty calibration is missing level {certainty!r}")
        levels[certainty] = {}
        for dimension in DIMENSIONS:
            counts = entry.get(dimension)
            if not (
                isinstance(counts, dict)
                and isinstance(counts.get("correct"), int)
                and isinstance(counts.get("total"), int)
                and 0 <= counts["correct"] <= counts["total"]
            ):
                raise CalibrationError(
                    f"certainty calibration has malformed counts for {certainty}/{dimension}"
                )
            levels[certainty][dimension] = LevelStats(counts["correct"], counts["total"])
    return SymbolicCalibration(levels, str(payload.get("source", "")))


def load_calibration(path: Path = CALIBRATION_PATH) -> SymbolicCalibration:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CalibrationError(f"cannot read certainty calibration {path}: {error}") from error
    return _parse(payload)
