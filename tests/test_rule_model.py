"""Rung-0 rule baseline tests (plan §8 rung 0: `models/rule.py`).

Time class from max loop-nesting depth; space class from max allocation
nesting depth. Deliberately naive -- it is the honest floor the later rungs
must beat, not a real predictor (plan §8).
"""
from __future__ import annotations

from core.taxonomy import SpaceClass, TimeClass
from features.tabular import TabularFeatures
from models.rule import predict_space, predict_time


def test_zero_loop_depth_predicts_constant_time():
    assert predict_time(TabularFeatures(0, 0)) == TimeClass.O_1


def test_one_loop_depth_predicts_linear_time():
    assert predict_time(TabularFeatures(1, 0)) == TimeClass.O_N


def test_two_loop_depth_predicts_quadratic_time():
    assert predict_time(TabularFeatures(2, 0)) == TimeClass.O_N2


def test_three_loop_depth_predicts_cubic_time():
    assert predict_time(TabularFeatures(3, 0)) == TimeClass.O_N3


def test_four_loop_depth_caps_at_cubic_time():
    # Rung-0 has no evidence for anything past O(n^3) from loop nesting
    # alone -- O(2^n) is a recursion signature, not a loop-nesting one, and
    # is out of scope for this rule (plan §8).
    assert predict_time(TabularFeatures(4, 0)) == TimeClass.O_N3


def test_zero_alloc_depth_predicts_constant_space():
    assert predict_space(TabularFeatures(0, 0)) == SpaceClass.O_1


def test_one_alloc_depth_predicts_linear_space():
    assert predict_space(TabularFeatures(0, 1)) == SpaceClass.O_N


def test_two_alloc_depth_predicts_quadratic_space():
    assert predict_space(TabularFeatures(0, 2)) == SpaceClass.O_N2


def test_three_alloc_depth_caps_at_quadratic_space():
    # SpaceClass has no O(n^3) rung -- cap at the taxonomy's ceiling.
    assert predict_space(TabularFeatures(0, 3)) == SpaceClass.O_N2
