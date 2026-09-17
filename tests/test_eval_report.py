"""Evaluation metrics tests (plan §9): accuracy/macro-F1/ordinal
distance/ECE/confusion matrix always computed together, never bare
accuracy."""
from __future__ import annotations

import numpy as np
import pytest

from eval.report import (
    SPACE_CLASSES,
    TIME_CLASSES,
    compute_metrics,
    per_language_metrics,
    plot_confusion_matrix,
)


def test_perfect_predictions_have_zero_ordinal_distance_and_full_accuracy():
    # macro-F1 is computed over the *full* taxonomy (`labels=classes`), not
    # just whatever classes happen to appear -- a class absent from this
    # test set would otherwise silently count as F1=0 against the average,
    # so "perfect predictions" here means at least one of every class.
    labels = list(TIME_CLASSES)
    metrics = compute_metrics("time", labels, labels, TIME_CLASSES)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.mean_ordinal_distance == 0.0


def test_ordinal_distance_penalises_far_errors_more_than_near_ones():
    true_labels = ["O(1)", "O(1)"]
    near_miss = ["O(log n)", "O(log n)"]  # adjacent rank
    far_miss = ["O(2^n)", "O(2^n)"]  # farthest rank

    near_metrics = compute_metrics("time", true_labels, near_miss, TIME_CLASSES)
    far_metrics = compute_metrics("time", true_labels, far_miss, TIME_CLASSES)
    assert near_metrics.mean_ordinal_distance < far_metrics.mean_ordinal_distance
    assert near_metrics.accuracy == far_metrics.accuracy == 0.0


def test_ece_is_none_without_confidences():
    labels = ["O(1)", "O(n)"]
    metrics = compute_metrics("time", labels, labels, TIME_CLASSES)
    assert metrics.ece is None


def test_ece_is_computed_when_confidences_are_given():
    labels = ["O(1)", "O(n)", "O(1)", "O(n)"]
    predictions = ["O(1)", "O(n)", "O(n)", "O(n)"]
    confidences = np.array([0.9, 0.9, 0.9, 0.9])
    metrics = compute_metrics("time", labels, predictions, TIME_CLASSES, confidences)
    assert metrics.ece is not None
    assert metrics.ece > 0


def test_confusion_matrix_shape_matches_class_count():
    labels = ["O(1)", "O(n)"]
    metrics = compute_metrics("time", labels, labels, TIME_CLASSES)
    assert metrics.confusion.shape == (len(TIME_CLASSES), len(TIME_CLASSES))


def test_space_dimension_uses_space_taxonomy():
    labels = ["O(1)", "O(n)", "O(n^2)"]
    metrics = compute_metrics("space", labels, labels, SPACE_CLASSES)
    assert metrics.classes == SPACE_CLASSES
    assert metrics.accuracy == 1.0


def test_compute_metrics_requires_at_least_one_example():
    with pytest.raises(ValueError):
        compute_metrics("time", [], [], TIME_CLASSES)


def test_per_language_metrics_splits_correctly():
    true_labels = ["O(1)", "O(1)", "O(n)", "O(n)"]
    predicted = ["O(1)", "O(n)", "O(n)", "O(n)"]  # python: 1/2 right, cpp: 2/2 right
    languages = ["python", "python", "cpp", "cpp"]
    by_language = per_language_metrics("time", true_labels, predicted, languages, TIME_CLASSES)
    assert set(by_language) == {"python", "cpp"}
    assert by_language["python"].accuracy == 0.5
    assert by_language["cpp"].accuracy == 1.0


def test_as_dict_rounds_and_serialises_cleanly():
    labels = ["O(1)", "O(n)"]
    metrics = compute_metrics("time", labels, labels, TIME_CLASSES)
    d = metrics.as_dict()
    assert d["accuracy"] == 1.0
    assert d["dimension"] == "time"
    assert isinstance(d["per_class_f1"], dict)


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    labels = ["O(1)", "O(n)", "O(n^2)", "O(1)"]
    predictions = ["O(1)", "O(n)", "O(n)", "O(1)"]
    metrics = compute_metrics("time", labels, predictions, TIME_CLASSES)
    out_path = tmp_path / "confusion.png"
    plot_confusion_matrix(metrics, "test chart", out_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
