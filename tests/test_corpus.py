"""Labelled-corpus record schema round-trip (plan §7, §12)."""
from __future__ import annotations

from data.corpus import CorpusRecord, read_jsonl, write_jsonl


def test_write_then_read_round_trips(tmp_path):
    records = [
        CorpusRecord(
            problem_id="0",
            solution_id="0_0",
            source="bigobench",
            language="python",
            code="def f(): pass\n",
            time_class="O(n)",
            space_class=None,
        ),
    ]
    path = tmp_path / "corpus.jsonl"
    write_jsonl(records, path)
    assert read_jsonl(path) == records


def test_write_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "corpus.jsonl"
    write_jsonl([], path)
    assert path.is_file()
