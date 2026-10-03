"""Tests for the canonical growth-curve generator (plan §10/§11)."""
from __future__ import annotations

import math

import pytest

from api.curve import growth_curve
from core.taxonomy import ExtendedTimeClass

_TIME_CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)")
_SPACE_CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)")


def test_growth_curve_includes_predicted_class_and_both_neighbours_when_in_the_middle():
    curve = growth_curve("time", "O(n)")
    assert set(curve["series"].keys()) == {"O(log n)", "O(n)", "O(n log n)"}


def test_growth_curve_has_no_lower_neighbour_at_the_bottom_of_the_taxonomy():
    curve = growth_curve("time", "O(1)")
    assert set(curve["series"].keys()) == {"O(1)", "O(log n)"}


def test_growth_curve_has_no_upper_neighbour_at_the_top_of_the_taxonomy():
    curve = growth_curve("time", "O(2^n)")
    assert set(curve["series"].keys()) == {"O(n^3)", "O(2^n)"}


def test_growth_curve_space_dimension_uses_the_five_class_taxonomy():
    curve = growth_curve("space", "O(n^2)")
    assert set(curve["series"].keys()) <= set(_SPACE_CLASSES)
    assert "O(n^2)" in curve["series"]


def test_growth_curve_series_values_are_monotonically_increasing_with_n():
    curve = growth_curve("time", "O(n^2)")
    for series in curve["series"].values():
        assert all(a <= b for a, b in zip(series, series[1:], strict=False))


def test_growth_curve_n_grid_is_shared_across_every_series():
    curve = growth_curve("time", "O(n)")
    for series in curve["series"].values():
        assert len(series) == len(curve["n"])


def test_growth_curve_orders_time_classes_by_rank_not_declared_dict_order():
    curve = growth_curve("time", "O(n^2)")
    # O(n^2)'s neighbours are O(n log n) below and O(n^3) above -- not, say,
    # O(1) or O(2^n), which would indicate the rank lookup is broken.
    assert set(curve["series"].keys()) == {"O(n log n)", "O(n^2)", "O(n^3)"}


@pytest.mark.parametrize("cls", _TIME_CLASSES)
def test_growth_curve_accepts_every_declared_time_class(cls):
    curve = growth_curve("time", cls)
    assert cls in curve["series"]
    assert curve["predicted_class"] == cls


def test_extended_time_curve_uses_the_richer_taxonomy_for_its_neighbours():
    curve = growth_curve("time_extended", "O(n^2 log n)")
    assert set(curve["series"].keys()) == {"O(n^2)", "O(n^2 log n)", "O(n^3)"}
    assert set(growth_curve("time_extended", "O(sqrt n)")["series"]) == {
        "O(log n)",
        "O(sqrt n)",
        "O(n)",
    }
    assert set(growth_curve("time_extended", "O(n!)")["series"]) == {"O(2^n)", "O(n!)"}


@pytest.mark.parametrize("cls", [c.value for c in ExtendedTimeClass])
def test_every_extended_class_has_a_finite_monotone_growth_function(cls):
    series = growth_curve("time_extended", cls)["series"][cls]
    assert all(a <= b for a, b in zip(series, series[1:], strict=False))
    assert all(math.isfinite(v) for v in series)


def test_sqrt_n_grows_between_log_n_and_n_beyond_the_crossover():
    curve = growth_curve("time_extended", "O(sqrt n)")
    series = curve["series"]
    for n, log_v, sqrt_v, lin_v in zip(
        curve["n"], series["O(log n)"], series["O(sqrt n)"], series["O(n)"], strict=True
    ):
        if n > 16:  # sqrt n only overtakes log n at n = 16
            assert log_v < sqrt_v < lin_v


def test_the_legacy_curve_dimension_is_unchanged_by_the_extension():
    assert set(growth_curve("time", "O(n^3)")["series"]) == {"O(n^2)", "O(n^3)", "O(2^n)"}
