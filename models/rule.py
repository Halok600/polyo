"""Rung 0: the rule baseline (plan §8).

Time class from max loop-nesting depth; space class from max allocation
nesting depth. No recursion handling, no loop-bound shape, no library-call
awareness -- those are rung-2 features. This rung exists to be beaten, and
to prove that beating it is nontrivial.
"""
from __future__ import annotations

from core.taxonomy import SpaceClass, TimeClass
from features.tabular import TabularFeatures

_TIME_BY_LOOP_DEPTH: dict[int, TimeClass] = {
    0: TimeClass.O_1,
    1: TimeClass.O_N,
    2: TimeClass.O_N2,
    3: TimeClass.O_N3,
}
_MAX_LOOP_DEPTH_RUNG_0 = max(_TIME_BY_LOOP_DEPTH)

_SPACE_BY_ALLOC_DEPTH: dict[int, SpaceClass] = {
    0: SpaceClass.O_1,
    1: SpaceClass.O_N,
    2: SpaceClass.O_N2,
}
_MAX_ALLOC_DEPTH_RUNG_0 = max(_SPACE_BY_ALLOC_DEPTH)


def predict_time(features: TabularFeatures) -> TimeClass:
    depth = min(features.max_loop_nesting_depth, _MAX_LOOP_DEPTH_RUNG_0)
    return _TIME_BY_LOOP_DEPTH[depth]


def predict_space(features: TabularFeatures) -> SpaceClass:
    depth = min(features.max_alloc_nesting_depth, _MAX_ALLOC_DEPTH_RUNG_0)
    return _SPACE_BY_ALLOC_DEPTH[depth]
