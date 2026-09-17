#!/usr/bin/env python3
"""Dedupe, problem-level train/val/test split, and class-distribution
reporting (plan §7, §9, §12).

**Split by problem, never by solution** -- solutions to the same problem are
near-duplicates, and splitting by solution leaks (plan §9: "the single
easiest way to accidentally publish a fraudulent accuracy"). The plan's own
wording is "no problem_id in more than one split", but different *sources*
use overlapping problem_id spellings (BigO(Bench)'s are bare small integers;
CodeComplex's are "round_problemid"), so this splits by (source, problem_id)
pairs -- strictly stronger, and what `tests/test_data_splits.py` enforces.

Assignment is a deterministic hash of (source, problem_id) -> [0, 100), not
`random.shuffle`, so the split is reproducible across machines and Python
versions without needing to pin a seed anywhere (`hash()` on strings is
randomised per-process in CPython; `hashlib.sha256` is not).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.corpus import CorpusRecord, read_jsonl, write_jsonl

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# test gets the remainder

_RAW_PROCESSED_DIR = Path("data/processed")
_SPLIT_NAMES = ("train", "val", "test")


def _bucket(key: str, num_buckets: int = 100) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def _split_name(source: str, problem_id: str, train_frac: float, val_frac: float) -> str:
    bucket = _bucket(f"{source}:{problem_id}")
    train_cut = round(train_frac * 100)
    val_cut = train_cut + round(val_frac * 100)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def dedupe(records: list[CorpusRecord]) -> list[CorpusRecord]:
    """Drops exact-duplicate (source, code) pairs, keeping the first."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for r in records:
        key = (r.source, r.code)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def stratify_cap(records: list[CorpusRecord], max_per_class: int) -> list[CorpusRecord]:
    """Caps each time_class bucket (None included) at `max_per_class`,
    keeping the lowest-hash records in each bucket -- deterministic, and
    keeps rare classes from being crowded out by a dominant one (plan §9's
    class-imbalance risk) the way a plain head(N) truncation would."""
    by_class: dict[str | None, list[CorpusRecord]] = defaultdict(list)
    for r in records:
        by_class[r.time_class].append(r)
    kept = []
    for group in by_class.values():
        group.sort(key=lambda r: _bucket(f"{r.source}:{r.solution_id}", num_buckets=1_000_003))
        kept.extend(group[:max_per_class])
    return kept


def split(
    records: list[CorpusRecord],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict[str, list[CorpusRecord]]:
    result: dict[str, list[CorpusRecord]] = {name: [] for name in _SPLIT_NAMES}
    for r in records:
        name = _split_name(r.source, r.problem_id, train_frac, val_frac)
        result[name].append(r)
    return result


def class_distribution(records: list[CorpusRecord]) -> dict[str, Counter[str]]:
    time_c: Counter[str] = Counter(r.time_class or "<none>" for r in records)
    space_c: Counter[str] = Counter(r.space_class or "<none>" for r in records)
    return {"time": time_c, "space": space_c}


def _load_processed_corpora(processed_dir: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        if path.stem.startswith("split_"):
            continue  # this script's own prior output, not a source corpus
        records.extend(read_jsonl(path))
    return records


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=_RAW_PROCESSED_DIR)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="cap each time_class bucket at N records (stratified, deterministic) before splitting",
    )
    args = parser.parse_args(argv)

    records = _load_processed_corpora(args.processed_dir)
    if not records:
        print(f"error: no source corpus files found under {args.processed_dir}", file=sys.stderr)
        return 1

    records = dedupe(records)
    if args.max_per_class is not None:
        records = stratify_cap(records, args.max_per_class)

    splits = split(records, args.train_frac, args.val_frac)

    report: dict[str, object] = {}
    for name, split_records in splits.items():
        out_path = args.processed_dir / f"split_{name}.jsonl"
        write_jsonl(split_records, out_path)
        dist = class_distribution(split_records)
        report[name] = {
            "records": len(split_records),
            "problems": len({(r.source, r.problem_id) for r in split_records}),
            "time_class": dict(dist["time"]),
            "space_class": dict(dist["space"]),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
