"""BigO(Bench) ingestion tests (plan §7), against a small synthetic fixture
that mirrors the real, verified schema of `problem_and_human_solutions_list.jsonl`
and `complexity_labels_light.jsonl` (see data/ingest_bigobench.py's module
docstring) -- not real dataset content, to keep this repo free of any
redistribution question around BigO(Bench)'s CC-BY-NC license.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.ingest_bigobench import ingest

_FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_maps_kept_and_drops_the_rest(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = ingest(
        _FIXTURES / "bigobench_solutions_sample.jsonl",
        _FIXTURES / "bigobench_labels_sample.jsonl",
        out,
    )

    # 0_0: both labels map cleanly. 0_1: both labels null. 0_2: time is
    # multivariate (unmapped), space maps -- kept on the space dimension
    # alone. 0_3: labelled but has no matching solution code.
    assert stats["kept"] == 2
    assert stats["missing_code"] == 1
    assert stats["dropped_no_usable_label"] == 1
    assert stats["time_null"] == 1
    assert stats["space_null"] == 1
    assert stats["time_unmapped"] == 1

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_id = {r["solution_id"]: r for r in records}

    assert by_id["0_0"]["time_class"] == "O(n)"
    assert by_id["0_0"]["space_class"] == "O(1)"
    assert by_id["0_0"]["source"] == "bigobench"
    assert by_id["0_0"]["language"] == "python"

    assert by_id["0_2"]["time_class"] is None
    assert by_id["0_2"]["space_class"] == "O(n^2)"  # canonical form, not BigOBench's "O(n**2)"

    assert "0_1" not in by_id
    assert "0_3" not in by_id
