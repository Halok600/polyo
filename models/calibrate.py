"""Temperature scaling and Expected Calibration Error (plan §8, §9).

"The UI shows a confidence number, so it has to mean something" (plan §8) --
temperature scaling rescales a model's raw (pre-softmax) scores by one
scalar T, fit on a held-out validation set, so its predicted-class
probability actually tracks its empirical accuracy. ECE is what's reported
to show whether that worked.

Fit by grid search + local refinement over T, not `scipy.optimize`, so this
module needs only numpy (already a dependency via scikit-learn/lightgbm)
rather than adding scipy as a direct one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _log_loss(scores: np.ndarray, temperature: float, true_index: np.ndarray) -> float:
    proba = _softmax(scores / temperature)
    eps = 1e-12
    true_proba = proba[np.arange(len(true_index)), true_index]
    return float(-np.mean(np.log(np.clip(true_proba, eps, 1.0))))


@dataclass(frozen=True, slots=True)
class Calibrator:
    temperature: float
    classes: tuple[str, ...]

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """`scores`: (n_examples, n_classes) raw/pre-softmax scores, in the
        same class order as `self.classes`. Returns calibrated
        probabilities, same shape."""
        return _softmax(scores / self.temperature)


def fit_temperature(
    scores: np.ndarray,
    true_labels: list[str],
    classes: tuple[str, ...],
    *,
    search_range: tuple[float, float] = (0.05, 10.0),
    coarse_steps: int = 200,
    refine_rounds: int = 6,
) -> Calibrator:
    """Fits T minimising validation-set negative log-likelihood: coarse grid
    search over `search_range`, then a few rounds of bisection-style
    refinement around the best point -- NLL(T) is smooth and unimodal in T
    for a single scalar temperature, so this converges without needing a
    general-purpose optimiser."""
    class_index = {c: i for i, c in enumerate(classes)}
    true_index = np.array([class_index[label] for label in true_labels])

    lo, hi = search_range
    candidates = np.linspace(lo, hi, coarse_steps)
    losses = [_log_loss(scores, float(t), true_index) for t in candidates]
    best_t = float(candidates[int(np.argmin(losses))])

    step = (hi - lo) / coarse_steps
    for _ in range(refine_rounds):
        step /= 4
        refined = np.linspace(max(best_t - 2 * step, 1e-3), best_t + 2 * step, 9)
        refined_losses = [_log_loss(scores, float(t), true_index) for t in refined]
        best_t = float(refined[int(np.argmin(refined_losses))])

    return Calibrator(temperature=best_t, classes=classes)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10
) -> float:
    """Standard ECE: bin by predicted-class confidence, weight each bin's
    |accuracy - mean confidence| by its share of examples."""
    if len(confidences) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        above_lo = confidences >= lo if i == 0 else confidences > lo
        in_bin = above_lo & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_accuracy = float(np.mean(correct[in_bin]))
        bin_confidence = float(np.mean(confidences[in_bin]))
        ece += (np.sum(in_bin) / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
