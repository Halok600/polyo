"""Oracle test-spec schema validation (plan §6, `oracle/spec.py`)."""
from __future__ import annotations

import pytest

from oracle.spec import InvalidSpecError, NGrid, ParamDist, ParamSpec, ParamType, TestSpec


def test_default_n_grid_is_geometric_powers_of_two():
    grid = NGrid(start_exp=2, stop_exp=5)
    assert grid.values() == [4, 8, 16, 32]


def test_explicit_n_grid_values_are_used_verbatim():
    grid = NGrid(explicit_values=(3, 5, 9))
    assert grid.values() == [3, 5, 9]


def test_n_grid_rejects_stop_before_start():
    with pytest.raises(InvalidSpecError):
        NGrid(start_exp=10, stop_exp=5)


def test_n_grid_rejects_n_below_two():
    with pytest.raises(InvalidSpecError):
        NGrid(explicit_values=(1, 4))


def test_spec_requires_a_param_sized_on_the_grid_variable():
    with pytest.raises(InvalidSpecError):
        TestSpec(
            entrypoint="f",
            params=(ParamSpec(type=ParamType.INT, dist=ParamDist.CONST, const=1),),
        )


def test_spec_accepts_a_correctly_sized_param():
    spec = TestSpec(
        entrypoint="f",
        params=(ParamSpec(type=ParamType.INT_ARRAY, size="n"),),
    )
    assert spec.n_grid.variable == "n"


def test_const_param_without_a_size_needs_a_const_value():
    with pytest.raises(InvalidSpecError):
        ParamSpec(type=ParamType.INT, dist=ParamDist.CONST)


def test_int_array_without_size_needs_a_const_length():
    with pytest.raises(InvalidSpecError):
        ParamSpec(type=ParamType.INT_ARRAY)
