"""Oracle runner + codegen smoke tests (plan §6): a rendered Python driver
actually runs in a subprocess and reports real measurements.
"""
from __future__ import annotations

import pytest

from oracle.runner import OracleRunError, run
from oracle.spec import NGrid, ParamDist, ParamSpec, ParamType, TestSpec

_LINEAR_SCAN = """
def linear_scan(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1
"""


def test_runner_produces_one_sample_per_n():
    spec = TestSpec(
        entrypoint="linear_scan",
        params=(
            ParamSpec(type=ParamType.INT_ARRAY, size="n"),
            ParamSpec(type=ParamType.INT, dist=ParamDist.CONST, const=999_999),
        ),
        n_grid=NGrid(start_exp=4, stop_exp=8),
        budget_ms=2000,
    )
    samples = run(spec, _LINEAR_SCAN)
    assert [s.n for s in samples] == spec.n_grid.values()
    assert all(s.time_ns > 0 for s in samples)
    assert all(s.peak_bytes >= 0 for s in samples)
    assert all(s.max_call_depth >= 1 for s in samples)


def test_runner_raises_when_the_solution_has_no_matching_entrypoint():
    spec = TestSpec(
        entrypoint="does_not_exist",
        params=(ParamSpec(type=ParamType.INT_ARRAY, size="n"),),
        n_grid=NGrid(start_exp=4, stop_exp=5),
    )
    with pytest.raises(OracleRunError):
        run(spec, _LINEAR_SCAN)


def test_runner_skips_ns_where_the_solution_raises():
    broken = """
def f(arr):
    raise ValueError("boom")
"""
    spec = TestSpec(
        entrypoint="f",
        params=(ParamSpec(type=ParamType.INT_ARRAY, size="n"),),
        n_grid=NGrid(start_exp=4, stop_exp=5),
    )
    with pytest.raises(OracleRunError):
        run(spec, broken)
