"""Canonical growth-curve generation for the `curve` response field (plan
§10, §11). NOT a measured curve -- the API never executes user code (plan
§10's hard invariant, enforced by `tests/test_isolation.py`) -- this is the
textbook shape of each complexity class over a representative *n* range, so
the frontend can draw "the predicted class bold, the two adjacent classes
ghosted behind it" (plan §11) without an oracle measurement, which only
ever exists for the offline-labelled training corpus, never for arbitrary
user-submitted code.
"""
from __future__ import annotations

import math

from core.taxonomy import SpaceClass, TimeClass

_N_MIN = 8
_N_MAX = 2**16
_NUM_POINTS = 12
# 2^n is capped, not left to overflow: a real 2^65536 is meaningless in a
# JSON payload, and this curve is illustrative shape only (plan §11), not a
# measurement.
_EXP_CAP = 2.0**64

_CLASSES_BY_DIMENSION: dict[str, tuple[str, ...]] = {
    "time": tuple(c.value for c in TimeClass),
    "space": tuple(c.value for c in SpaceClass),
}


def _n_grid() -> list[int]:
    log_min, log_max = math.log2(_N_MIN), math.log2(_N_MAX)
    step = (log_max - log_min) / (_NUM_POINTS - 1)
    return [round(2 ** (log_min + i * step)) for i in range(_NUM_POINTS)]


def _canonical_value(class_name: str, n: int) -> float:
    if class_name == "O(1)":
        return 1.0
    if class_name == "O(log n)":
        return math.log2(n)
    if class_name == "O(n)":
        return float(n)
    if class_name == "O(n log n)":
        return n * math.log2(n)
    if class_name == "O(n^2)":
        return float(n**2)
    if class_name == "O(n^3)":
        return float(n**3)
    if class_name == "O(2^n)":
        try:
            return min(2.0**n, _EXP_CAP)
        except OverflowError:
            # 2.0**n raises rather than returning inf once it overflows a
            # float64 (n gets well past 1024 within `_n_grid`'s range) --
            # the cap is exactly what would have been returned anyway.
            return _EXP_CAP
    raise ValueError(f"unknown class: {class_name!r}")


def _neighbour_classes(predicted_class: str, classes: tuple[str, ...]) -> list[str]:
    idx = classes.index(predicted_class)
    neighbours = [predicted_class]
    if idx > 0:
        neighbours.append(classes[idx - 1])
    if idx < len(classes) - 1:
        neighbours.append(classes[idx + 1])
    return neighbours


def growth_curve(dimension: str, predicted_class: str) -> dict[str, object]:
    """One shared *n* grid, one series per class in `{predicted, one rank
    down, one rank up}` (fewer at either taxonomy edge) -- plan §11's
    "where your code sits among its neighbours"."""
    classes = _CLASSES_BY_DIMENSION[dimension]
    n = _n_grid()
    series = {
        cls: [_canonical_value(cls, ni) for ni in n]
        for cls in _neighbour_classes(predicted_class, classes)
    }
    return {"n": n, "predicted_class": predicted_class, "series": series}
