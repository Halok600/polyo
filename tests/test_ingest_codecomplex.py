"""CodeComplex-Data ingestion tests (plan §7), against a small synthetic
fixture that mirrors the real, verified `python_data.jsonl` schema (see
data/ingest_codecomplex.py's module docstring).
"""
from __future__ import annotations

import json
from pathlib import Path

from data.ingest_codecomplex import ingest

_FIXTURE = Path(__file__).parent / "fixtures" / "codecomplex_python_sample.jsonl"
_JAVA_FIXTURE = Path(__file__).parent / "fixtures" / "codecomplex_java_sample.jsonl"


def test_ingest_maps_kept_and_drops_unmapped_labels(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = ingest(_FIXTURE, out)

    assert stats["kept"] == 2
    assert stats["time_unmapped"] == 1
    assert stats["dropped_no_usable_label"] == 1

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2

    linear = next(r for r in records if r["time_class"] == "O(n)")
    assert linear["problem_id"] == "0001_A"
    assert linear["source"] == "codecomplex"
    assert linear["space_class"] is None

    quadratic = next(r for r in records if r["time_class"] == "O(n^2)")
    assert quadratic["problem_id"] == "0002_B"

    # No solution-level id in the source -- synthesised, and must be unique.
    assert len({r["solution_id"] for r in records}) == len(records)


def test_ingest_java_tags_language_and_avoids_solution_id_collision(tmp_path):
    py_out = tmp_path / "py.jsonl"
    java_out = tmp_path / "java.jsonl"
    ingest(_FIXTURE, py_out, language="python")
    ingest(_JAVA_FIXTURE, java_out, language="java")

    py_records = [json.loads(line) for line in py_out.read_text(encoding="utf-8").splitlines()]
    java_records = [json.loads(line) for line in java_out.read_text(encoding="utf-8").splitlines()]

    assert {r["language"] for r in java_records} == {"java"}
    assert {r["language"] for r in py_records} == {"python"}

    # Same fixture shape (and problem_ids) in both files -- solution_ids must
    # still be globally unique once both are ingested into one corpus, since
    # dedupe() keys on (source, code), not solution_id (data/build.py).
    all_solution_ids = [r["solution_id"] for r in py_records + java_records]
    assert len(set(all_solution_ids)) == len(all_solution_ids)
