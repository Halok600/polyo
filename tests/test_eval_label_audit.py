"""The label audit's tables, on made-up entries, and the committed evidence behind them."""

from __future__ import annotations

import json
from pathlib import Path

from eval.label_audit import (
    AUDIT,
    by_source,
    disagreement_table,
    random_sample_table,
    render,
    usable,
)


def _entry(
    kind: str,
    agent: str,
    corpus: str | None,
    engine: str,
    confidence: str = "high",
    source: str = "bigobench",
) -> dict[str, object]:
    return {
        "kind": kind,
        "confidence": confidence,
        "source": source,
        "agent_time": agent,
        "corpus_time": corpus,
        "engine_time": engine,
        "agent_space": agent,
        "corpus_space": corpus,
        "engine_space": engine,
    }


def test_low_confidence_answers_are_left_out() -> None:
    entries = [
        _entry("random", "O(n)", "O(n)", "O(n)"),
        _entry("random", "O(n)", "O(1)", "O(n)", "low"),
    ]
    assert len(usable(entries, "random")) == 1


def test_the_random_table_counts_who_agrees_with_the_independent_reading() -> None:
    entries = [
        _entry("random", "O(n)", "O(n)", "O(n)"),  # both right
        _entry("random", "O(n)", "O(1)", "O(n)"),  # engine right
        _entry("random", "O(n)", "O(n)", "O(n^2)"),  # corpus right
        _entry("random", "O(n)", "O(1)", "O(n^2)"),  # neither
        _entry("random", "O(n)", None, "O(n)"),  # no corpus label: not counted
    ]
    table = random_sample_table(entries, "time")
    assert table == {
        "rows": 4,
        "corpus right": 2,
        "engine right": 2,
        "both right": 1,
        "neither right": 1,
    }


def test_the_disagreement_table_says_which_side_the_independent_reading_took() -> None:
    entries = [
        _entry("disagreement", "O(n)", "O(1)", "O(n)"),
        _entry("disagreement", "O(1)", "O(1)", "O(n)"),
        _entry("disagreement", "O(n^2)", "O(1)", "O(n)"),
    ]
    table = disagreement_table(entries, "time")
    assert table["rows"] == 3
    assert table["neither"] == 1
    assert sum(v for k, v in table.items() if k.startswith("engine right")) == 1
    assert sum(v for k, v in table.items() if k.startswith("corpus right")) == 1


def test_the_corpus_label_accuracy_is_split_by_source() -> None:
    entries = [
        _entry("random", "O(n)", "O(n)", "O(n)", source="bigobench"),
        _entry("random", "O(n)", "O(1)", "O(n)", source="bigobench"),
        _entry("random", "O(n)", "O(n)", "O(n)", source="codecomplex"),
    ]
    assert by_source(entries, "time") == {"bigobench": (1, 2), "codecomplex": (1, 1)}


def test_the_report_has_every_section() -> None:
    entries = [
        _entry("random", "O(n)", "O(n)", "O(n)"),
        _entry("disagreement", "O(n)", "O(1)", "O(n)"),
    ]
    text = render(entries)
    assert "Random sample" in text and "Disagreements" in text and "Reading the numbers" in text


def test_the_committed_evidence_is_well_formed_and_holds_no_dataset_code() -> None:
    entries = json.loads(Path(AUDIT).read_text("utf-8"))
    assert len(entries) >= 100
    kinds = {e["kind"] for e in entries}
    assert kinds == {"random", "disagreement"}
    for e in entries:
        assert len(e["sha256"]) == 64 and "code" not in e
        assert e["confidence"] in ("high", "medium", "low")
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "an id may appear in only one sample"
