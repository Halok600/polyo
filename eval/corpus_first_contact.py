"""The symbolic engine's first contact with the REAL corpus (BigO(Bench), CodeComplex test splits).

The engine was written against hand-made golden programs and checked on programs labelled by the
same author, then on a blind corpus written by independent agents. The corpus the ML arm trains on
is a third, very different kind of code: competitive-programming solutions that read standard
input, labelled by measurement (BigO(Bench)) or by contest tags (CodeComplex). The engine never saw
a record of it. This script runs it over a deterministic sample of the TEST split once and records
what it answered (`eval/corpus_first_contact.json`); like the blind corpus, that record is evidence
and is never re-recorded after the engine has been fixed against it.

    python -m eval.corpus_first_contact record     # once
    python -m eval.corpus_first_contact summary    # agreement with the corpus labels, per certainty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_TEST = ROOT / "data" / "processed" / "split_test.jsonl"
RECORD = ROOT / "eval" / "corpus_first_contact.json"
SAMPLE_PER_LANGUAGE = {"python": 1500, "java": 500}


def _bucket(solution_id: str) -> int:
    return int(hashlib.sha256(solution_id.encode("utf-8")).hexdigest()[:12], 16)


def sample_records() -> list[dict[str, object]]:
    """A deterministic sample of the test split: the records with the smallest hash of their id,
    per language (so it does not depend on file order or on a random seed)."""
    from data.corpus import read_jsonl

    by_language: dict[str, list] = defaultdict(list)
    for record in read_jsonl(SPLIT_TEST):
        if record.language in SAMPLE_PER_LANGUAGE and (
            record.time_class is not None or record.space_class is not None
        ):
            by_language[record.language].append(record)
    chosen = []
    for language, limit in SAMPLE_PER_LANGUAGE.items():
        ranked = sorted(by_language[language], key=lambda r: _bucket(f"{r.source}:{r.solution_id}"))
        chosen.extend(ranked[:limit])
    return [
        {
            "id": f"{r.source}:{r.solution_id}",
            "language": r.language,
            "source": r.source,
            "problem_id": r.problem_id,
            "label_time": r.time_class,
            "label_space": r.space_class,
            "code": r.code,
        }
        for r in chosen
    ]


def first_contact(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    from analysis.engine import analyze

    record: dict[str, dict[str, object]] = {}
    for row in rows:
        entry: dict[str, object] = {
            "language": row["language"],
            "source": row["source"],
            "problem_id": row["problem_id"],
            "label_time": row["label_time"],
            "label_space": row["label_space"],
        }
        try:
            result = analyze(str(row["code"]), str(row["language"]))
        except Exception as error:  # noqa: BLE001 -- a crash is data
            entry.update(certainty="error", error=f"{type(error).__name__}: {error}")
        else:
            entry.update(
                time_class=result.time.cls,
                space_class=result.space.cls,
                time_expr=result.time.text,
                space_expr=result.space.text,
                certainty=result.certainty,
            )
        record[str(row["id"])] = entry
    return record


def summarise(record: dict[str, dict[str, object]]) -> str:
    lines = []
    for dimension in ("time", "space"):
        per_certainty: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for entry in record.values():
            label = entry.get(f"label_{dimension}")
            if label is None or entry["certainty"] == "error":
                continue
            cell = per_certainty[str(entry["certainty"])]
            cell[1] += 1
            cell[0] += entry.get(f"{dimension}_class") == label
        lines.append(f"{dimension}:")
        for certainty in ("certain", "assumed", "unknown"):
            correct, total = per_certainty.get(certainty, [0, 0])
            share = f"{correct / total:.1%}" if total else "n/a"
            lines.append(f"  {certainty:8} {correct}/{total} = {share}")
    lines.append(f"certainty: {dict(Counter(str(e['certainty']) for e in record.values()))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["record", "summary"])
    args = parser.parse_args(argv)
    if args.command == "record":
        if RECORD.exists():
            print(f"{RECORD} exists: first contact happens once")
            return 1
        record = first_contact(sample_records())
        RECORD.write_text(json.dumps(record, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(record)} records -> {RECORD}")
        print(summarise(record))
        return 0
    print(summarise(json.loads(RECORD.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
