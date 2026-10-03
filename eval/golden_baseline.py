"""The golden regression ratchet: which (case, language) pairs the symbolic engine gets right.

`eval/golden_baseline.json` records the pairs whose time class, space class, time expression and
space expression the engine currently gets right. Improvements are allowed and change the file
explicitly (`python -m eval.golden_baseline --update`, committed with the change that earned it);
anything that was right and is now wrong fails `tests/test_engine_golden.py`. Known-limitation
cases are not gated and never appear here.

The numbers a phase gate quotes (accuracy per language on a subset) live in the tests; this file is
the fine-grained net underneath them, so a fix that trades one case for another cannot hide inside
an unchanged percentage.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.golden_suite import load_cases
from eval.predictors import get_predictor
from eval.scorecard import score_predictor

BASELINE = Path(__file__).with_name("golden_baseline.json")
METRICS = ("time", "space", "time_expr", "space_expr")


def _key(case_id: str, language: str) -> str:
    return f"{case_id}:{language}"


def passing(predictor_name: str = "symbolic") -> dict[str, list[str]]:
    """For each metric, the sorted `case:language` pairs the predictor gets right today."""
    results = score_predictor(get_predictor(predictor_name), load_cases())
    out: dict[str, set[str]] = {metric: set() for metric in METRICS}
    for r in results:
        if r.known_limitation:
            continue
        key = _key(r.case_id, r.language)
        if r.time_ok:
            out["time"].add(key)
        if r.space_ok:
            out["space"].add(key)
        if r.time_expr_ok:
            out["time_expr"].add(key)
        if r.space_expr_ok:
            out["space_expr"].add(key)
    return {metric: sorted(keys) for metric, keys in out.items()}


def load_baseline() -> dict[str, list[str]]:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {metric: list(data.get(metric, [])) for metric in METRICS}


def write_baseline(current: dict[str, list[str]]) -> None:
    BASELINE.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")


def regressions(
    current: dict[str, list[str]], baseline: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Pairs that were right in the baseline and are wrong now, per metric."""
    return {
        metric: sorted(set(baseline[metric]) - set(current[metric]))
        for metric in METRICS
        if set(baseline[metric]) - set(current[metric])
    }


def improvements(
    current: dict[str, list[str]], baseline: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Pairs that are right now and were not in the baseline, per metric."""
    return {
        metric: sorted(set(current[metric]) - set(baseline[metric]))
        for metric in METRICS
        if set(current[metric]) - set(baseline[metric])
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    current = passing()
    if "--update" in args:
        write_baseline(current)
        print(f"wrote {BASELINE} ({', '.join(f'{m}={len(current[m])}' for m in METRICS)})")
        return 0
    baseline = load_baseline()
    lost, gained = regressions(current, baseline), improvements(current, baseline)
    for metric, keys in lost.items():
        print(f"REGRESSED {metric}: {len(keys)}", *keys[:20], sep="\n  ")
    for metric, keys in gained.items():
        print(f"improved  {metric}: {len(keys)} (run with --update to record)")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
