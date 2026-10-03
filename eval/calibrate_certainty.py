"""Measure how often the symbolic engine is right at each certainty level, on blind programs.

    python -m eval.calibrate_certainty record   # run the engine ONCE on the blind set
    python -m eval.calibrate_certainty fit      # counts per certainty level -> served calibration

The blind set (`tests/data/blind_v1_cases.json`) was written by independent labellers who never saw
the engine. `record` stores what the engine answered the first time it met each program
(`tests/data/blind_v1_first_contact.json`) and refuses to overwrite that record: once the engine has
been fixed against these programs, re-running it would only measure how well it memorised them.
`fit` combines the record with the labels (adjudicated labels are the cases file's: each correction
carries its reason) and writes `analysis/certainty_calibration.json`, which `api/predict.py` serves
as the confidence of a symbolic answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analysis.calibration import CALIBRATION_PATH, DIMENSIONS, SERVED_LEVELS, LevelStats

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "tests" / "data" / "blind_v1_cases.json"
RECORD_PATH = ROOT / "tests" / "data" / "blind_v1_first_contact.json"
SOURCE_NAME = "tests/data/blind_v1_cases.json scored at first contact (blind_v1_first_contact.json)"


def _load(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def first_contact(cases: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    """What the engine answers for each case right now. A crash is recorded as `error`."""
    from analysis.engine import analyze

    record: dict[str, dict[str, str]] = {}
    for case in cases:
        key = str(case["id"])
        try:
            result = analyze(str(case["source"]), str(case["language"]))
        except Exception as error:  # noqa: BLE001 -- a crash is data, not a reason to stop
            record[key] = {"error": f"{type(error).__name__}: {error}", "certainty": "error"}
            continue
        record[key] = {
            "time_class": result.time.cls,
            "space_class": result.space.cls,
            "time_expr": result.time.text,
            "space_expr": result.space.text,
            "certainty": result.certainty,
        }
    return record


def score(
    cases: list[dict[str, object]], record: dict[str, dict[str, str]]
) -> dict[str, dict[str, LevelStats]]:
    """Correct / total per served certainty level and dimension. Cases the engine called
    "unknown" (they go to the model) or crashed on are not part of a served level."""
    counts: dict[str, dict[str, list[int]]] = {
        level: {dimension: [0, 0] for dimension in DIMENSIONS} for level in SERVED_LEVELS
    }
    for case in cases:
        answer = record.get(str(case["id"]))
        if answer is None or answer["certainty"] not in SERVED_LEVELS:
            continue
        for dimension in DIMENSIONS:
            cell = counts[answer["certainty"]][dimension]
            cell[1] += 1
            cell[0] += answer[f"{dimension}_class"] == case[f"{dimension}_class"]
    return {
        level: {dimension: LevelStats(*cell) for dimension, cell in per.items()}
        for level, per in counts.items()
    }


def build_payload(
    cases: list[dict[str, object]], record: dict[str, dict[str, str]]
) -> dict[str, object]:
    levels = score(cases, record)
    served = {
        level: {dimension: {"correct": s.correct, "total": s.total} for dimension, s in per.items()}
        for level, per in levels.items()
    }
    by_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for case in cases:
        answer = record.get(str(case["id"]), {})
        if answer.get("certainty") in SERVED_LEVELS:
            cell = by_language[str(case["language"])]
            cell[1] += 1
            cell[0] += all(answer[f"{d}_class"] == case[f"{d}_class"] for d in DIMENSIONS)
    certainties = Counter(answer.get("certainty", "missing") for answer in record.values())
    return {
        "source": SOURCE_NAME,
        "cases": len(cases),
        "certainty_counts": dict(sorted(certainties.items())),
        "levels": served,
        "joint_accuracy_by_language": {
            lang: {"correct": c, "total": t} for lang, (c, t) in sorted(by_language.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["record", "fit"])
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing first-contact record"
    )
    args = parser.parse_args(argv)
    cases = _load(CASES_PATH)
    if args.command == "record":
        if RECORD_PATH.exists() and not args.force:
            print(f"{RECORD_PATH} exists: first contact happens once (use --force to overwrite)")
            return 1
        RECORD_PATH.write_text(
            json.dumps(first_contact(cases), indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"recorded first contact for {len(cases)} programs -> {RECORD_PATH}")
        return 0
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    payload = build_payload(cases, record)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["levels"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
