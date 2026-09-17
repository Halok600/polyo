"""Shared labelled-corpus record schema for every ingestion script (plan §7,
§12). One record is one (problem, solution) pair with source provenance and
taxonomy-mapped labels. `time_class`/`space_class` are `None` when that
source doesn't label that dimension at all (e.g. CodeComplex is time-only)
or when the source's raw label didn't map to our taxonomy and was dropped
(see `core.taxonomy.UnmappableLabelError`) -- a record survives as long as
*at least one* dimension is usable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    problem_id: str
    solution_id: str
    source: str
    language: str
    code: str
    time_class: str | None
    space_class: str | None


def write_jsonl(records: list[CorpusRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_jsonl(path: Path) -> list[CorpusRecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(CorpusRecord(**json.loads(line)))
    return records
