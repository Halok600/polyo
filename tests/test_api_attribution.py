"""Tests for per-prediction feature attribution (plan §10). Uses a
precomputed, gain-weighted feature-importance heuristic, not live
LightGBM SHAP -- see `api/attribution.py`'s module docstring for why.
"""
from __future__ import annotations

from api.attribution import spans_for_feature, top_contributions
from core.ir import IRGraph, IRNode
from models.gbdt import FEATURE_NAMES

_SPAN_A = (1, 0, 1, 10)
_SPAN_B = (2, 4, 3, 8)


def _ir_with(symbols_and_spans):
    nodes = [
        IRNode(id=i, symbol=symbol, span=span, text=symbol)
        for i, (symbol, span) in enumerate(symbols_and_spans)
    ]
    return IRGraph(nodes=nodes, edges=[])


def _zero_values() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def test_spans_for_feature_returns_spans_of_the_mapped_symbols_only():
    ir = _ir_with([("LOOP_FOR", _SPAN_A), ("BRANCH", _SPAN_B)])
    assert spans_for_feature(ir, "loop_count") == [_SPAN_A]
    assert spans_for_feature(ir, "branch_count") == [_SPAN_B]


def test_spans_for_feature_returns_empty_for_a_feature_with_no_symbol_mapping():
    ir = _ir_with([("LOOP_FOR", _SPAN_A)])
    assert spans_for_feature(ir, "node_count") == []


def test_spans_for_feature_respects_max_spans():
    ir = _ir_with([("LOOP_FOR", _SPAN_A), ("LOOP_WHILE", _SPAN_B), ("LOOP_FOR", _SPAN_A)])
    assert len(spans_for_feature(ir, "loop_count", max_spans=2)) == 2


def test_every_feature_name_has_a_span_mapping_entry():
    # The module asserts this at import time already -- this test documents
    # the invariant explicitly rather than relying on import-time luck.
    from api.attribution import _FEATURE_SYMBOLS

    assert set(_FEATURE_SYMBOLS) == set(FEATURE_NAMES)


def test_top_contributions_ranks_by_absolute_value_and_attaches_spans():
    ir = _ir_with([("LOOP_FOR", _SPAN_A)])
    values = _zero_values() | {"loop_count": 3.0, "branch_count": 1.0}
    weights = {"loop_count": -0.9, "branch_count": 0.1}
    scales = {"loop_count": 1.0, "branch_count": 1.0}

    contributions = top_contributions(values, weights, scales, ir, top_k=5)
    assert contributions[0]["feature"] == "loop_count"
    assert contributions[0]["contribution"] == -2.7
    assert contributions[0]["spans"] == [list(_SPAN_A)]


def test_top_contributions_divides_by_scale_before_weighting():
    # Same value and weight for both features, but "big_scale" varies far
    # more in typical code -- once divided by its scale, it should rank
    # BELOW "small_scale", even though its raw value*weight product alone
    # would rank it first.
    ir = _ir_with([])
    values = _zero_values() | {"node_count": 100.0, "max_loop_nesting_depth": 2.0}
    weights = {"node_count": 0.5, "max_loop_nesting_depth": 0.5}
    scales = {"node_count": 200.0, "max_loop_nesting_depth": 1.0}

    contributions = top_contributions(values, weights, scales, ir, top_k=2)
    assert contributions[0]["feature"] == "max_loop_nesting_depth"
    assert contributions[0]["contribution"] == 1.0
    assert contributions[1]["feature"] == "node_count"
    assert contributions[1]["contribution"] == 0.25


def test_top_contributions_drops_near_zero_values():
    ir = _ir_with([])
    values = _zero_values() | {"call_count": 1e-9}
    weights = {"call_count": 1.0}
    scales = {"call_count": 1.0}
    assert top_contributions(values, weights, scales, ir, min_abs_contribution=1e-6) == []


def test_top_contributions_treats_a_missing_weight_or_scale_as_a_safe_default():
    ir = _ir_with([])
    values = _zero_values() | {"call_count": 100.0}
    # No entry for "call_count" in weights -- must not crash or default to
    # something nonzero.
    assert top_contributions(values, {}, {}, ir) == []


def test_top_contributions_respects_top_k():
    ir = _ir_with([])
    values = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}
    weights = dict.fromkeys(FEATURE_NAMES, 1.0)
    scales = dict.fromkeys(FEATURE_NAMES, 1.0)
    contributions = top_contributions(values, weights, scales, ir, top_k=3)
    assert len(contributions) == 3
