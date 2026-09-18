"""Split-conformal prediction tests -- the actual property under test is
marginal coverage: fit on one split, check empirical coverage on a held-out
split drawn from the same distribution, same as `models/train_production.py`
does for real (calibrate on val, report on test).
"""
from __future__ import annotations

import numpy as np
import pytest

from models.conformal import evaluate_conformal, fit_conformal

_CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)")


def _synthetic_proba(true_idx: np.ndarray, sharpness: float, seed: int) -> np.ndarray:
    """A believable-but-imperfect softmax: peaked around the true rank, with
    some genuine mass spread to neighbours (an ordinal classifier's errors
    cluster near the true class, which is exactly the structure conformal
    sets are supposed to exploit)."""
    rng = np.random.default_rng(seed)
    n, k = len(true_idx), len(_CLASSES)
    ranks = np.arange(k)
    logits = -sharpness * (ranks[None, :] - true_idx[:, None]) ** 2
    logits += rng.normal(0, 0.5, (n, k))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_predict_set_is_always_contiguous_in_ordinal_rank():
    rng = np.random.default_rng(0)
    n = 400
    true_idx = rng.integers(0, len(_CLASSES), n)
    proba = _synthetic_proba(true_idx, sharpness=0.3, seed=1)  # noisy -- exercises wide sets too
    labels = [_CLASSES[i] for i in true_idx]

    cal_proba, test_proba = proba[:200], proba[200:]
    cal_labels = labels[:200]
    calibration = fit_conformal(cal_proba, cal_labels, _CLASSES, alphas=(0.1,))

    for row in test_proba:
        pred_set = calibration.predict_set(row, alpha=0.1)
        ranks = sorted(_CLASSES.index(c) for c in pred_set)
        assert ranks == list(range(ranks[0], ranks[-1] + 1)), pred_set
        # ordinal order, not probability order
        assert list(pred_set) == [_CLASSES[r] for r in ranks]


def test_empirical_coverage_meets_the_target_on_held_out_data():
    rng = np.random.default_rng(2)
    n = 4000
    true_idx = rng.integers(0, len(_CLASSES), n)
    proba = _synthetic_proba(true_idx, sharpness=0.6, seed=3)
    labels = [_CLASSES[i] for i in true_idx]

    cal_proba, test_proba = proba[:2000], proba[2000:]
    cal_labels, test_labels = labels[:2000], labels[2000:]

    for alpha in (0.05, 0.10, 0.20):
        calibration = fit_conformal(cal_proba, cal_labels, _CLASSES, alphas=(alpha,))
        result = evaluate_conformal(calibration, test_proba, test_labels, alpha)
        # Marginal coverage is a population guarantee, not per-run exact --
        # a few points of slack for finite-sample (n=2000) fluctuation.
        assert result.empirical_coverage >= (1 - alpha) - 0.03, (alpha, result.empirical_coverage)


def test_smaller_alpha_never_produces_a_smaller_average_set():
    rng = np.random.default_rng(4)
    n = 2000
    true_idx = rng.integers(0, len(_CLASSES), n)
    proba = _synthetic_proba(true_idx, sharpness=0.5, seed=5)
    labels = [_CLASSES[i] for i in true_idx]
    cal_proba, test_proba = proba[:1000], proba[1000:]
    cal_labels, test_labels = labels[:1000], labels[1000:]

    calibration = fit_conformal(cal_proba, cal_labels, _CLASSES, alphas=(0.05, 0.10, 0.20))
    sizes = {
        alpha: evaluate_conformal(calibration, test_proba, test_labels, alpha).average_set_size
        for alpha in (0.05, 0.10, 0.20)
    }
    # Higher required coverage (smaller alpha) demands sets at least as wide.
    assert sizes[0.05] >= sizes[0.10] >= sizes[0.20]


def test_predict_set_always_includes_the_point_prediction():
    rng = np.random.default_rng(6)
    n = 500
    true_idx = rng.integers(0, len(_CLASSES), n)
    proba = _synthetic_proba(true_idx, sharpness=0.4, seed=7)
    labels = [_CLASSES[i] for i in true_idx]
    calibration = fit_conformal(proba, labels, _CLASSES, alphas=(0.1,))

    for row in proba:
        pred_set = calibration.predict_set(row, alpha=0.1)
        assert _CLASSES[int(row.argmax())] in pred_set


def test_predict_set_raises_for_an_alpha_never_fit():
    calibration = fit_conformal(
        np.tile(np.eye(len(_CLASSES))[0], (10, 1)), [_CLASSES[0]] * 10, _CLASSES, alphas=(0.1,)
    )
    with pytest.raises(KeyError):
        calibration.predict_set(np.eye(len(_CLASSES))[0], alpha=0.5)


def test_evaluate_conformal_matches_a_hand_computed_example():
    classes = ("a", "b", "c")
    # Perfectly confident, always correct -- the minimal-alpha set is just
    # the single point prediction, every time, 100% coverage.
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = ["a", "b", "c"]
    calibration = fit_conformal(proba, labels, classes, alphas=(0.1,))
    result = evaluate_conformal(calibration, proba, labels, alpha=0.1)
    assert result.empirical_coverage == 1.0
    assert result.average_set_size == 1.0
    assert result.n == 3
