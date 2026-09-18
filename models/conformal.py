"""Split-conformal prediction over PolyO's ordinal taxonomy.

Instead of one predicted class + a raw softmax confidence, this outputs the
smallest CONTIGUOUS interval of ordinally-adjacent classes that's
guaranteed -- marginally, distribution-free, regardless of whether the
model itself is well-calibrated -- to contain the true class at least
(1 - alpha) of the time on exchangeable data. This is the actual reframe of
mean_ordinal_distance (already reported everywhere else in this project):
the model is usually within one class even when its single top pick is
"wrong," and a set that says so honestly is a better statement than one
number that reads like a coin flip.

**Deliberately not the textbook Adaptive Prediction Sets (APS) construction**
(Romano, Sesia & Candes 2020): APS sorts classes by predicted probability
and greedily adds the most-probable remaining class first, which for an
ORDINAL taxonomy can return a non-contiguous set -- e.g. {O(1), O(n log n),
O(2^n)}, skipping the "boring middle" classes even though they sit between
the included ones. Unreadable, and undercuts the entire point of a set
prediction being a legible answer. Since `core.taxonomy` classes are
ordinally ranked, the nonconformity score here is defined over CONTIGUOUS
intervals around the model's own point prediction instead: the interval
grows outward, one adjacent rank at a time (whichever of the current low-1
or high+1 rank currently holds more of the predicted probability mass),
until its own cumulative mass reaches a calibrated threshold -- the same
"walk the cumulative distribution" idea APS uses, just constrained to
intervals a human reads as one answer ("O(n log n) - O(n^2)"), never a
scattered set. The calibrated threshold is a probability MASS, not a
fixed class count -- that's what makes the returned set size adaptive per
example: a confident, peaked prediction reaches the threshold within a
class or two, an unsure, flat one needs more. Calibrating a fixed number
of classes instead (this module's first version did) produces a
constant-width set for nearly every input regardless of how sure the
model actually is on that input -- a real bug, not a stylistic choice.

**Calibration-set note, stated plainly**: this project has train/val/test,
not a fourth calibration-only split. `models/train_production.py` fits both
temperature scaling AND this conformal quantile on the same val split --
conformal's coverage guarantee only requires the calibration and test sets
be exchangeable, not that calibration be untouched by any other tuning
step, so this is standard-enough practice, not a shortcut with an unstated
cost. Coverage and average set size are always reported on the held-out
*test* split, which neither step ever touches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)


def _interval_for_mass(
    proba_row: np.ndarray, point_rank: int, target_mass: float
) -> tuple[int, int]:
    """Grows a [lo, hi] rank interval outward from `point_rank`, absorbing
    whichever adjacent rank (lo-1 or hi+1, whichever exists) currently
    holds the higher probability in `proba_row` -- ties broken toward the
    lower rank, arbitrary but deterministic -- until the interval's own
    cumulative mass reaches `target_mass`. This is what makes the set size
    adaptive per example: a peaked distribution (the model is confident)
    reaches the threshold within one or two classes, a flat one (the model
    is unsure) needs more -- a FIXED number of expansion steps, calibrated
    once and reused for every example, would return the same-sized set
    for a confident and an unsure prediction alike."""
    lo = hi = point_rank
    n = len(proba_row)
    mass = float(proba_row[point_rank])
    while mass < target_mass and (lo > 0 or hi < n - 1):
        can_left = lo > 0
        can_right = hi < n - 1
        if can_left and (not can_right or proba_row[lo - 1] >= proba_row[hi + 1]):
            lo -= 1
            mass += float(proba_row[lo])
        else:
            hi += 1
            mass += float(proba_row[hi])
    return lo, hi


def _mass_to_include(proba_row: np.ndarray, point_rank: int, true_rank: int) -> float:
    """This example's nonconformity score: the cumulative probability mass
    of the smallest interval -- grown by the same absorb-the-higher-
    adjacent-mass rule as `_interval_for_mass` -- that reaches `true_rank`.
    Always terminates: growing to both boundaries always covers [0, n-1]."""
    lo = hi = point_rank
    n = len(proba_row)
    mass = float(proba_row[point_rank])
    while not (lo <= true_rank <= hi):
        can_left = lo > 0
        can_right = hi < n - 1
        if can_left and (not can_right or proba_row[lo - 1] >= proba_row[hi + 1]):
            lo -= 1
            mass += float(proba_row[lo])
        else:
            hi += 1
            mass += float(proba_row[hi])
    return mass


def _alpha_key(alpha: float) -> str:
    return f"{alpha:.2f}"


@dataclass(frozen=True, slots=True)
class ConformalCalibration:
    classes: tuple[str, ...]  # ordinal order, core.taxonomy's own ordering
    mass_threshold_by_alpha: dict[str, float]  # e.g. {"0.05": 0.91, "0.10": 0.74, "0.20": 0.55}

    def predict_set(self, proba: np.ndarray, alpha: float) -> list[str]:
        """`proba`: this example's calibrated probability vector, same
        class order as `self.classes`. Returns the classes in the
        conformal interval, in ordinal order."""
        key = _alpha_key(alpha)
        if key not in self.mass_threshold_by_alpha:
            raise KeyError(
                f"no calibrated quantile for alpha={alpha!r} -- fit_conformal wasn't run with it"
            )
        point_rank = int(proba.argmax())
        lo, hi = _interval_for_mass(proba, point_rank, self.mass_threshold_by_alpha[key])
        return list(self.classes[lo : hi + 1])


def fit_conformal(
    val_proba: np.ndarray,
    true_labels: list[str],
    classes: tuple[str, ...],
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> ConformalCalibration:
    """`val_proba`: (n_examples, n_classes) CALIBRATED probabilities (i.e.
    already through `models.calibrate.Calibrator`) on the calibration
    (val) split, same class order as `classes`."""
    class_index = {c: i for i, c in enumerate(classes)}
    true_ranks = [class_index[label] for label in true_labels]
    point_ranks = val_proba.argmax(axis=1)

    scores = np.array(
        [
            _mass_to_include(val_proba[i], int(point_ranks[i]), true_ranks[i])
            for i in range(len(true_labels))
        ]
    )

    n = len(scores)
    mass_threshold_by_alpha: dict[str, float] = {}
    for alpha in alphas:
        # Standard split-conformal finite-sample quantile correction (see
        # e.g. Angelopoulos & Bates' conformal prediction tutorial): the
        # ceil((n+1)(1-alpha))/n empirical quantile, "higher" interpolation
        # so the result is always one of the observed scores.
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        mass_threshold_by_alpha[_alpha_key(alpha)] = float(
            np.quantile(scores, q_level, method="higher")
        )

    return ConformalCalibration(classes=classes, mass_threshold_by_alpha=mass_threshold_by_alpha)


@dataclass(frozen=True, slots=True)
class ConformalEvaluation:
    alpha: float
    n: int
    empirical_coverage: float
    average_set_size: float

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "n": self.n,
            "empirical_coverage": round(self.empirical_coverage, 4),
            "average_set_size": round(self.average_set_size, 4),
        }


def evaluate_conformal(
    calibration: ConformalCalibration,
    test_proba: np.ndarray,
    true_labels: list[str],
    alpha: float,
) -> ConformalEvaluation:
    """Empirical coverage + average set size on a held-out split the
    calibration never saw -- the actual check that the theoretical
    guarantee held in practice, not just a claim."""
    covered = 0
    total_size = 0
    n = len(true_labels)
    for i, label in enumerate(true_labels):
        pred_set = calibration.predict_set(test_proba[i], alpha)
        total_size += len(pred_set)
        if label in pred_set:
            covered += 1
    return ConformalEvaluation(
        alpha=alpha,
        n=n,
        empirical_coverage=covered / n if n else 0.0,
        average_set_size=total_size / n if n else 0.0,
    )
