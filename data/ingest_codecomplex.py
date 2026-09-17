#!/usr/bin/env python3
"""Ingest CodeComplex-Data's Python split (plan §7) into the shared
labelled-corpus schema. Time-only -- CodeComplex has no space labels, so
every record's `space_class` is `None`.

**Real, verified schema** (checked against the released file, not guessed):
  python_data.jsonl: {"src", "complexity", "problem", "from", "tags"}
  `complexity` is one of: constant, logn, linear, nlogn, quadratic, cubic,
  np. `problem` is a contest round + problem id, not unique per submission
  -- there is no solution-level id, so one is synthesised from the file's
  row index (see `ingest`).

  Note: the repo's own README documents this 7th class as "exponential",
  but the actual released python_data.jsonl (checked directly, all 4,900
  rows) uses "np" instead -- both are registered below so this survives
  either spelling.
From https://github.com/sybaik1/CodeComplex-Data (arXiv:2401.08719; see
NOTICE.md for the license). Download with:
    curl -L -o data/raw/codecomplex/python_data.jsonl \\
        https://raw.githubusercontent.com/sybaik1/CodeComplex-Data/main/python_data.jsonl
(~5MB, 4,900 rows -- `data/raw/` is gitignored.)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from core.taxonomy import TimeClass, UnmappableLabelError, map_time_label, register_time_source
from data.corpus import CorpusRecord, write_jsonl

SOURCE = "codecomplex"

# Verified against the real python_data.jsonl rows (not just the README --
# see the module docstring's note on "np" vs "exponential").
_TIME_LABELS: dict[str, TimeClass] = {
    "constant": TimeClass.O_1,
    "logn": TimeClass.O_LOG_N,
    "linear": TimeClass.O_N,
    "nlogn": TimeClass.O_N_LOG_N,
    "quadratic": TimeClass.O_N2,
    "cubic": TimeClass.O_N3,
    "np": TimeClass.O_2N,
    "exponential": TimeClass.O_2N,
}
register_time_source(SOURCE, _TIME_LABELS)


def _map_time_or_none(raw: str | None, stats: Counter[str]) -> TimeClass | None:
    if raw is None:
        stats["time_null"] += 1
        return None
    try:
        return map_time_label(SOURCE, raw)
    except UnmappableLabelError:
        stats["time_unmapped"] += 1
        return None


def ingest(python_data_path: Path, out_path: Path) -> Counter[str]:
    records: list[CorpusRecord] = []
    stats: Counter[str] = Counter()

    with python_data_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            time_class = _map_time_or_none(row.get("complexity"), stats)
            if time_class is None:
                stats["dropped_no_usable_label"] += 1
                continue

            problem_id = row["problem"]
            records.append(
                CorpusRecord(
                    problem_id=problem_id,
                    solution_id=f"{problem_id}_{i}",
                    source=SOURCE,
                    language="python",
                    code=row["src"],
                    time_class=time_class.value,
                    space_class=None,
                )
            )
            stats["kept"] += 1

    write_jsonl(records, out_path)
    return stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ingest CodeComplex-Python into the corpus.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/codecomplex/python_data.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/codecomplex_python.jsonl"))
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(
            "error: raw CodeComplex file not found -- see this script's "
            "module docstring for the download command",
            file=sys.stderr,
        )
        return 1

    stats = ingest(args.data, args.out)
    print(json.dumps(dict(stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
