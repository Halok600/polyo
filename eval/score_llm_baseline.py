#!/usr/bin/env python3
"""Scores the LLM zero-shot baseline's predictions against the real labels
in eval/llm_baseline_sample.json -- the classifying agents only ever saw
eval/llm_baseline_unlabeled.json (see eval/sample_llm_baseline.py), so this
is the first place ground truth and predictions are actually compared.

Uses the exact same eval/report.py:compute_metrics every other rung is
scored with (accuracy, macro-F1, mean ordinal distance), so this baseline
is a real row in the same table, not a differently-computed number that
merely looks comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.report import SPACE_CLASSES, TIME_CLASSES, compute_metrics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=Path("eval/llm_baseline_sample.json"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("eval/llm_baseline_predictions.json")
    )
    parser.add_argument("--out", type=Path, default=Path("eval/llm_baseline_metrics.json"))
    args = parser.parse_args(argv)

    labeled = {r["id"]: r for r in json.loads(args.labeled.read_text(encoding="utf-8"))}
    predicted = json.loads(args.predictions.read_text(encoding="utf-8"))
    pred_by_id = {p["id"]: p for p in predicted["predictions"]}

    missing = set(labeled) - set(pred_by_id)
    if missing:
        preview = sorted(missing)[:10]
        print(f"warning: {len(missing)} ids never got a prediction: {preview}...", file=sys.stderr)

    scored_ids = sorted(set(labeled) & set(pred_by_id))
    print(f"scoring {len(scored_ids)} of {len(labeled)} sampled examples")

    results = {}
    for dimension, classes in (("time", TIME_CLASSES), ("space", SPACE_CLASSES)):
        true_y = [labeled[i][f"{dimension}_class"] for i in scored_ids]
        pred_y = [pred_by_id[i][f"{dimension}_class"] for i in scored_ids]
        metrics = compute_metrics(dimension, true_y, pred_y, tuple(classes))
        results[dimension] = metrics.as_dict()
        print(json.dumps(results[dimension], indent=2))

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
