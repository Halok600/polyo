"""Corpus -> parsed-example pipeline tests (plan §8, `models/dataset.py`)."""
from __future__ import annotations

from data.corpus import CorpusRecord
from models.dataset import build_examples, space_labels, time_labels, with_label


def _record(code: str, language: str = "python", time_class: str | None = "O(n)", **kw):
    return CorpusRecord(
        problem_id=kw.pop("problem_id", "0"),
        solution_id=kw.pop("solution_id", "0_0"),
        source=kw.pop("source", "test"),
        language=language,
        code=code,
        time_class=time_class,
        space_class=kw.pop("space_class", None),
    )


def test_build_examples_parses_valid_python():
    records = [_record("def f(x):\n    return x + 1\n")]
    examples, stats = build_examples(records)
    assert len(examples) == 1
    assert stats.total == 1
    assert stats.parsed == 1
    assert stats.failed == 0
    assert examples[0].features.max_loop_nesting_depth == 0


def test_build_examples_counts_unsupported_language_separately():
    # "go" was this test's original example of an unsupported language;
    # Phase 4 added Go support, so this now uses a language that still has
    # no `parsing/lang/*.toml` mapping.
    records = [_record("fn f() {}", language="rust")]
    examples, stats = build_examples(records)
    assert examples == []
    assert stats.unsupported_language == 1
    assert stats.parsed == 0


def test_build_examples_survives_and_counts_a_bad_example():
    good = _record("def f(x):\n    return x + 1\n")
    # A lone UTF-16 surrogate is valid in a Python str but cannot be
    # UTF-8-encoded -- parsing/parse.py's `source.encode("utf-8")` raises,
    # deterministically exercising the broad-except path.
    bad = _record("def f():\n    x = '\ud800'\n")
    examples, stats = build_examples([good, bad])
    assert len(examples) == 1
    assert stats.total == 2
    assert stats.parsed == 1
    assert stats.failed == 1
    assert "UnicodeEncodeError" in stats.failure_types


def test_time_and_space_labels_read_off_the_record():
    records = [_record("def f(): pass\n", time_class="O(1)", space_class="O(n)")]
    examples, _ = build_examples(records)
    assert time_labels(examples) == ["O(1)"]
    assert space_labels(examples) == ["O(n)"]


def test_with_label_drops_none_labelled_examples():
    records = [
        _record("def f(): pass\n", time_class="O(1)"),
        _record("def g(): pass\n", time_class=None),
    ]
    examples, _ = build_examples(records)
    labels = time_labels(examples)
    kept_examples, kept_labels = with_label(examples, labels)
    assert kept_labels == ["O(1)"]
    assert len(kept_examples) == 1
