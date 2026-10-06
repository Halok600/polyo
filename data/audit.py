#!/usr/bin/env python3
"""What is actually in the training corpus: the numbers behind the plan's diagnosis, recomputed.

    python -m data.audit [--sample 20000]

Reports, for the train / val / test splits under data/processed: size and problems, language and
source mix, how many records have no usable time or space label, the label distribution per
language, how concentrated the corpus is (solutions per problem), and (on a deterministic sample,
because parsing every record is slow) how many records are near-duplicates of another solution of
the same problem (same IR shape, same labels). Prints JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.build import ir_signature
from data.corpus import CorpusRecord, read_jsonl


def _bucket(record: CorpusRecord) -> int:
    key = f"{record.source}:{record.solution_id}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def describe(records: list[CorpusRecord]) -> dict[str, object]:
    per_problem = Counter((r.source, r.problem_id) for r in records)
    sizes = sorted(per_problem.values())
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        by_language[r.language][r.time_class or "<none>"] += 1
    return {
        "records": len(records),
        "problems": len(per_problem),
        "languages": dict(Counter(r.language for r in records)),
        "sources": dict(Counter(r.source for r in records)),
        "time_label_missing": sum(r.time_class is None for r in records),
        "space_label_missing": sum(r.space_class is None for r in records),
        "no_usable_label": sum(r.time_class is None and r.space_class is None for r in records),
        "time_labels_by_language": {k: dict(v) for k, v in by_language.items()},
        "solutions_per_problem": {
            "max": sizes[-1] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "p95": sizes[int(len(sizes) * 0.95)] if sizes else 0,
            "problems_with_over_50": sum(s > 50 for s in sizes),
        },
    }


def near_duplicates(records: list[CorpusRecord], sample: int) -> dict[str, object]:
    """Within a sample, the share of records that have another record of the same problem with the
    same IR shape and labels. A lower bound for the whole corpus (a sample sees fewer pairs)."""
    chosen = sorted(records, key=_bucket)[:sample]
    groups: Counter[tuple[str, str, str, str | None, str | None]] = Counter()
    unparsed = 0
    for r in chosen:
        signature = ir_signature(r)
        if signature is None:
            unparsed += 1
            continue
        groups[(r.source, r.problem_id, signature, r.time_class, r.space_class)] += 1
    redundant = sum(count - 1 for count in groups.values())
    return {
        "sampled": len(chosen),
        "unparsed": unparsed,
        "redundant_in_sample": redundant,
        "redundant_share": round(redundant / max(1, len(chosen) - unparsed), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample", type=int, default=20000)
    args = parser.parse_args(argv)
    report: dict[str, object] = {}
    train: list[CorpusRecord] = []
    for name in ("train", "val", "test"):
        path = args.processed_dir / f"split_{name}.jsonl"
        if not path.is_file():
            print(f"error: {path} not found -- run `python -m data.build` first", file=sys.stderr)
            return 1
        records = read_jsonl(path)
        report[name] = describe(records)
        if name == "train":
            train = records
    report["train_near_duplicates"] = near_duplicates(train, args.sample)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
