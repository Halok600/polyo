#!/usr/bin/env python3
"""Samples N held-out test-set records (both time_class and space_class
present) for the LLM zero-shot baseline (see MODEL_CARD.md). Fixed seed for
reproducibility; no oversampling of rare languages/classes -- this has to
match the same natural distribution rung3's own headline number is computed
over (dominated by Python), or the two aren't a fair comparison.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from data.corpus import read_jsonl


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/llm_baseline_sample.json"))
    parser.add_argument(
        "--out-unlabeled",
        type=Path,
        default=Path("eval/llm_baseline_unlabeled.json"),
        help="Same records without time_class/space_class -- the file the "
        "classifying agents actually read, so there's no answer key sitting "
        "in the same file they're told to look at.",
    )
    args = parser.parse_args(argv)

    test_path = args.processed_dir / "split_test.jsonl"
    if not test_path.is_file():
        print(f"error: {test_path} not found -- run `python -m data.build` first", file=sys.stderr)
        return 1

    records = read_jsonl(test_path)
    both_labeled = [r for r in records if r.time_class is not None and r.space_class is not None]
    print(f"{len(both_labeled)} of {len(records)} test records have both dimensions labelled")

    rng = random.Random(args.seed)
    sample = rng.sample(both_labeled, min(args.n, len(both_labeled)))

    payload = [
        {
            "id": i,
            "language": r.language,
            "code": r.code,
            "time_class": r.time_class,
            "space_class": r.space_class,
        }
        for i, r in enumerate(sample)
    ]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(payload)} sampled records (with labels) to {args.out}")

    unlabeled = [{"id": p["id"], "language": p["language"], "code": p["code"]} for p in payload]
    args.out_unlabeled.write_text(json.dumps(unlabeled, indent=2), encoding="utf-8")
    print(f"wrote {len(unlabeled)} unlabeled records to {args.out_unlabeled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
