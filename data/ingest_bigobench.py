#!/usr/bin/env python3
"""Ingest BigO(Bench) (plan §7) into the shared labelled-corpus schema.

**Real, verified schema** (checked against the released files, not guessed):
  problem_and_human_solutions_list.jsonl
    {"problem_id": ..., "correct_solution_list": [{"solution_id", "solution_code"}, ...]}
  complexity_labels_light.jsonl
    {"problem_id", "solution_id", "time_complexity_inferred", "space_complexity_inferred"}
Both from https://huggingface.co/datasets/facebook/BigOBench (CC-BY-NC-4.0 --
see NOTICE.md). Download with (no `datasets` library needed, plain files):
    curl -L -o data/raw/bigobench/problem_and_human_solutions_list.jsonl \\
        https://huggingface.co/datasets/facebook/BigOBench/resolve/main/data/problem_and_human_solutions_list.jsonl
    curl -L -o data/raw/bigobench/complexity_labels_light.jsonl \\
        https://huggingface.co/datasets/facebook/BigOBench/resolve/main/data/complexity_labels_light.jsonl
(~915MB and ~307MB respectively -- `data/raw/` is gitignored.)

**Non-obvious limitation, found while implementing this script:** BigO(Bench)
labels each solution's complexity *per input parameter*, e.g. "O(n+m**2+k)"
for a 3-parameter problem -- not against one dominant size like our taxonomy
(core/taxonomy.py, which is single-variable by design, see plan §3). Only
labels already expressed in a single variable (e.g. "O(n)", "O(n**2)") map
cleanly; every multivariate label is dropped, loudly, via the same
`UnmappableLabelError` path any unmapped label from any source takes. This
script's printed stats report exactly how much of the raw data that costs --
it is a real, measured fraction, not a guess, and belongs in the model card.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from core.taxonomy import (
    SpaceClass,
    TimeClass,
    UnmappableLabelError,
    map_space_label,
    map_time_label,
    register_space_source,
    register_time_source,
)
from data.corpus import CorpusRecord, write_jsonl

SOURCE = "bigobench"

# Verified against real rows of complexity_labels_light.jsonl -- only the
# single-variable forms our taxonomy can express (see module docstring).
_TIME_LABELS: dict[str, TimeClass] = {
    "O(1)": TimeClass.O_1,
    "O(logn)": TimeClass.O_LOG_N,
    "O(n)": TimeClass.O_N,
    "O(nlogn)": TimeClass.O_N_LOG_N,
    "O(n**2)": TimeClass.O_N2,
    "O(n**3)": TimeClass.O_N3,
    "O(2**n)": TimeClass.O_2N,
}
_SPACE_LABELS: dict[str, SpaceClass] = {
    "O(1)": SpaceClass.O_1,
    "O(logn)": SpaceClass.O_LOG_N,
    "O(n)": SpaceClass.O_N,
    "O(nlogn)": SpaceClass.O_N_LOG_N,
    "O(n**2)": SpaceClass.O_N2,
}
register_time_source(SOURCE, _TIME_LABELS)
register_space_source(SOURCE, _SPACE_LABELS)


def _load_solutions(path: Path) -> dict[tuple[str, str], str]:
    code_by_key: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            problem_id = row["problem_id"]
            for sol in row.get("correct_solution_list", []):
                code_by_key[(problem_id, sol["solution_id"])] = sol["solution_code"]
    return code_by_key


def _map_time_or_none(raw: str | None, stats: Counter[str]) -> TimeClass | None:
    if raw is None:
        stats["time_null"] += 1
        return None
    try:
        return map_time_label(SOURCE, raw)
    except UnmappableLabelError:
        stats["time_unmapped"] += 1
        return None


def _map_space_or_none(raw: str | None, stats: Counter[str]) -> SpaceClass | None:
    if raw is None:
        stats["space_null"] += 1
        return None
    try:
        return map_space_label(SOURCE, raw)
    except UnmappableLabelError:
        stats["space_unmapped"] += 1
        return None


def ingest(solutions_path: Path, labels_path: Path, out_path: Path) -> Counter[str]:
    code_by_key = _load_solutions(solutions_path)
    records: list[CorpusRecord] = []
    stats: Counter[str] = Counter()

    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["problem_id"], row["solution_id"])
            code = code_by_key.get(key)
            if code is None:
                stats["missing_code"] += 1
                continue

            time_class = _map_time_or_none(row.get("time_complexity_inferred"), stats)
            space_class = _map_space_or_none(row.get("space_complexity_inferred"), stats)
            if time_class is None and space_class is None:
                stats["dropped_no_usable_label"] += 1
                continue

            records.append(
                CorpusRecord(
                    problem_id=key[0],
                    solution_id=key[1],
                    source=SOURCE,
                    language="python",
                    code=code,
                    time_class=time_class.value if time_class else None,
                    space_class=space_class.value if space_class else None,
                )
            )
            stats["kept"] += 1

    write_jsonl(records, out_path)
    return stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ingest BigO(Bench) into the labelled corpus.")
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("data/raw/bigobench/problem_and_human_solutions_list.jsonl"),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/raw/bigobench/complexity_labels_light.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/bigobench_python.jsonl"))
    args = parser.parse_args(argv)

    if not args.solutions.is_file() or not args.labels.is_file():
        print(
            "error: raw BigO(Bench) files not found -- see this script's "
            "module docstring for the download command",
            file=sys.stderr,
        )
        return 1

    stats = ingest(args.solutions, args.labels, args.out)
    print(json.dumps(dict(stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
