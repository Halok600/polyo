"""Temperature scaling / ECE tests (plan §8, §9)."""
from __future__ import annotations

import numpy as np

from models.calibrate import expected_calibration_error, fit_temperature

_CLASSES = ("a", "b", "c")


def _overconfident_scores(
    true_idx: np.ndarray, boost: float, noise_scale: float, seed: int, scale: float = 1.0
):
    rng = np.random.default_rng(seed)
    n = len(true_idx)
    scores = rng.normal(0, noise_scale, (n, len(_CLASSES)))
    scores[np.arange(n), true_idx] += boost
    return scores * scale


def test_temperature_scaling_reduces_ece_for_overconfident_scores():
    rng = np.random.default_rng(0)
    n = 600
    true_idx = rng.integers(0, len(_CLASSES), n)
    # A modest boost keeps *accuracy* well below 100% (so genuine confidence
    # should be middling), then a large uniform scale inflates the softmax
    # confidence far past that -- genuine overconfidence, not just "a very
    # accurate classifier that's rightly confident".
    scores = _overconfident_scores(true_idx, boost=1.5, noise_scale=1.0, seed=1, scale=8.0)
    labels = [_CLASSES[i] for i in true_idx]

    uncalibrated = np.exp(scores - scores.max(axis=1, keepdims=True))
    uncalibrated /= uncalibrated.sum(axis=1, keepdims=True)
    uncal_conf = uncalibrated.max(axis=1)
    uncal_correct = uncalibrated.argmax(axis=1) == true_idx
    ece_before = expected_calibration_error(uncal_conf, uncal_correct)

    calibrator = fit_temperature(scores, labels, _CLASSES)
    proba = calibrator.calibrate(scores)
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == true_idx
    ece_after = expected_calibration_error(conf, correct)

    assert calibrator.temperature > 1.0  # overconfident scores need softening
    assert ece_after < ece_before


def test_calibration_does_not_change_the_argmax_prediction():
    rng = np.random.default_rng(2)
    n = 300
    true_idx = rng.integers(0, len(_CLASSES), n)
    scores = _overconfident_scores(true_idx, boost=4.0, noise_scale=1.0, seed=3)
    labels = [_CLASSES[i] for i in true_idx]

    calibrator = fit_temperature(scores, labels, _CLASSES)
    uncalibrated_pred = scores.argmax(axis=1)
    calibrated_pred = calibrator.calibrate(scores).argmax(axis=1)
    assert np.array_equal(uncalibrated_pred, calibrated_pred)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_predictions():
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correct = np.array([True] * 9 + [False])  # 90% accuracy matches 90% confidence
    ece = expected_calibration_error(confidences, correct)
    assert ece < 1e-9


def test_expected_calibration_error_is_positive_for_overconfident_predictions():
    confidences = np.full(10, 0.99)
    correct = np.array([True] * 5 + [False] * 5)  # 50% accuracy, 99% confidence
    ece = expected_calibration_error(confidences, correct)
    assert ece > 0.4


def test_expected_calibration_error_handles_empty_input():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_calibrator_calibrate_returns_a_valid_probability_distribution():
    rng = np.random.default_rng(4)
    scores = rng.normal(0, 2, (50, 3))
    labels = [_CLASSES[i] for i in rng.integers(0, 3, 50)]
    calibrator = fit_temperature(scores, labels, _CLASSES)
    proba = calibrator.calibrate(scores)
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)
