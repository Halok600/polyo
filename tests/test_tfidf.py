"""Rung 1 tests: TF-IDF over IR symbol n-grams + logistic regression
(plan §8)."""
from __future__ import annotations

import pytest

from data.corpus import CorpusRecord
from models.dataset import build_examples
from models.tfidf import fit, symbol_sequence_text

_CONSTANT = "def f(x):\n    return x + 1\n"
_LINEAR = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
_QUADRATIC = (
    "def f(xs):\n"
    "    total = 0\n"
    "    for x in xs:\n"
    "        for y in xs:\n"
    "            total += x * y\n"
    "    return total\n"
)


def _record(code: str, i: int, label: str) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=label,
        space_class=None,
    )


def _synthetic_examples(n_per_class: int = 12):
    records = []
    i = 0
    for code, label in ((_CONSTANT, "O(1)"), (_LINEAR, "O(n)"), (_QUADRATIC, "O(n^2)")):
        for _ in range(n_per_class):
            records.append(_record(code, i, label))
            i += 1
    examples, stats = build_examples(records)
    assert stats.failed == 0
    labels = [r.time_class for r in records]
    return examples, labels


def test_symbol_sequence_text_is_space_joined_ir_symbols():
    examples, _ = _synthetic_examples(n_per_class=1)
    text = symbol_sequence_text(examples[0])
    assert "FUNC_DEF" in text
    assert " " in text


def test_fit_rejects_mismatched_lengths():
    examples, labels = _synthetic_examples(n_per_class=1)
    with pytest.raises(ValueError):
        fit(examples, labels[:-1])


def test_fit_separates_obviously_different_shapes():
    examples, labels = _synthetic_examples(n_per_class=15)
    model = fit(examples, labels)

    held_out_examples, held_out_labels = _synthetic_examples(n_per_class=3)
    predictions = model.predict(held_out_examples)
    accuracy = sum(
        p == y for p, y in zip(predictions, held_out_labels, strict=True)
    ) / len(held_out_labels)
    # These three shapes are maximally distinct in IR-symbol n-gram space (no
    # loop vs. one loop vs. nested loops) -- this should be trivially
    # separable, so a broken pipeline (not a weak model) is what a failure
    # here would mean.
    assert accuracy == 1.0


def test_predict_proba_rows_sum_to_one():
    examples, labels = _synthetic_examples(n_per_class=10)
    model = fit(examples, labels)
    distributions = model.predict_proba(examples[:5])
    for dist in distributions:
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        assert set(dist.keys()) == set(model.classes)


def test_decision_function_shape_matches_classes():
    examples, labels = _synthetic_examples(n_per_class=10)
    model = fit(examples, labels)
    scores = model.decision_function(examples[:4])
    assert scores.shape == (4, len(model.classes))
