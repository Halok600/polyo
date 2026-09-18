#!/usr/bin/env python3
"""Standalone runner for `eval.transfer.run_raw_token_transfer_experiment`.

Deliberately NOT folded into `eval/phase5_report.py`'s `main()`: that script
retrains rung 3's GNN from scratch (needs the GPU, takes real wall-clock
time) to produce numbers already committed in `PHASE5_REPORT.md`. This only
adds one cheap TF-IDF+LogisticRegression baseline on the exact same
`data/processed/split_*.jsonl` split, so its output is directly comparable
to that report's existing GNN transfer table without retraining anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.corpus import read_jsonl
from eval.transfer import run_raw_token_transfer_experiment
from models.dataset import build_examples, space_labels, time_labels


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("eval/raw_token_transfer.json"))
    args = parser.parse_args(argv)

    train_path = args.processed_dir / "split_train.jsonl"
    test_path = args.processed_dir / "split_test.jsonl"
    for path in (train_path, test_path):
        if not path.is_file():
            print(f"error: {path} not found -- run `python -m data.build` first", file=sys.stderr)
            return 1

    print("building train examples...", flush=True)
    train_ex, _ = build_examples(read_jsonl(train_path))
    print("building test examples...", flush=True)
    test_ex, _ = build_examples(read_jsonl(test_path))
    train_time_y, train_space_y = time_labels(train_ex), space_labels(train_ex)
    test_time_y, test_space_y = time_labels(test_ex), space_labels(test_ex)

    results: dict[str, dict[str, object]] = {}
    for dimension, train_y, test_y in (
        ("time", train_time_y, test_time_y),
        ("space", train_space_y, test_space_y),
    ):
        print(f"=== raw-token transfer: {dimension} ===", flush=True)
        metrics_by_language = run_raw_token_transfer_experiment(
            dimension, train_ex, train_y, test_ex, test_y
        )
        results[dimension] = {lang: m.as_dict() for lang, m in metrics_by_language.items()}
        print(json.dumps(results[dimension], indent=2), flush=True)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
