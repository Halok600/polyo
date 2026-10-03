"""The richer time taxonomy the symbolic engine answers in (plan section 2: "symbolic inside,
richer classes out"): O(sqrt n) between O(log n) and O(n), O(n^2 log n) between O(n^2) and O(n^3),
and O(n!) after O(2^n). The ML arm keeps its 7-class head; every extended class is reported in the
legacy fields as a legacy class, rounded up where one exists."""

from __future__ import annotations

import pytest

from analysis.engine import analyze
from core.taxonomy import (
    ExtendedTimeClass,
    TimeClass,
    extended_time_rank,
    legacy_time_class,
)


def test_the_extended_classes_are_ordered_by_growth() -> None:
    order = [c.value for c in sorted(ExtendedTimeClass, key=extended_time_rank)]
    assert order == [
        "O(1)",
        "O(log n)",
        "O(sqrt n)",
        "O(n)",
        "O(n log n)",
        "O(n^2)",
        "O(n^2 log n)",
        "O(n^3)",
        "O(2^n)",
        "O(n!)",
    ]


def test_every_legacy_class_is_an_extended_class() -> None:
    assert {c.value for c in TimeClass} <= {c.value for c in ExtendedTimeClass}


@pytest.mark.parametrize(
    ("extended", "legacy"),
    [
        ("O(sqrt n)", "O(n)"),
        ("O(n^2 log n)", "O(n^3)"),
        ("O(n!)", "O(2^n)"),
        ("O(n^2)", "O(n^2)"),
        ("O(1)", "O(1)"),
    ],
)
def test_an_extended_class_rounds_up_to_a_legacy_one(extended: str, legacy: str) -> None:
    assert legacy_time_class(ExtendedTimeClass(extended)) == TimeClass(legacy)


def test_rounding_goes_up_until_the_legacy_taxonomy_runs_out() -> None:
    """O(n!) is beyond the largest legacy class, so it saturates there (the answer is flagged
    lossy); every other extended class rounds up to a legacy class at or above it."""
    for cls in ExtendedTimeClass:
        legacy = legacy_time_class(cls)
        if cls is ExtendedTimeClass.O_FACT_N:
            assert legacy is TimeClass.O_2N
        else:
            assert extended_time_rank(ExtendedTimeClass(legacy.value)) >= extended_time_rank(cls)


TRIAL_DIVISION = """\
def is_prime(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
"""

SORT_IN_LOOP = """\
def f(nums):
    for i in range(len(nums)):
        nums.sort()
"""

PERMUTE_BY_SIZE = """\
def f(n):
    if n == 0:
        return 1
    total = 0
    for i in range(n):
        total += f(n - 1)
    return total
"""

NESTED = "def f(a):\n    for x in a:\n        for y in a:\n            pass\n"


@pytest.mark.parametrize(
    ("source", "extended", "legacy", "lossy"),
    [
        (TRIAL_DIVISION, "O(sqrt n)", "O(n)", True),
        (SORT_IN_LOOP, "O(n^2 log n)", "O(n^3)", True),
        (PERMUTE_BY_SIZE, "O(n!)", "O(2^n)", True),
        (NESTED, "O(n^2)", "O(n^2)", False),
    ],
)
def test_the_engine_reports_the_extended_class_next_to_the_legacy_one(
    source: str, extended: str, legacy: str, lossy: bool
) -> None:
    result = analyze(source, "python").time
    assert (result.ext_cls, result.cls, result.lossy) == (extended, legacy, lossy)


def test_space_keeps_the_legacy_taxonomy() -> None:
    result = analyze("def f(n):\n    return [0] * n\n", "python").space
    assert result.ext_cls == result.cls == "O(n)"
