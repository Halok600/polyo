"""Per-prediction feature attribution (plan §10's `attribution` response
field): each request's own tabular feature VALUES (`features/tabular.py`),
weighted by a global, precomputed feature-importance weight from the
production GBDT (rung 2 -- plan §8's "feature engineering,
interpretability" rung), mapped back to the spans of the IR nodes each
feature is derived from.

**Deliberately not real per-prediction SHAP, and why:** LightGBM's own
per-prediction SHAP (`Booster.predict(..., pred_contrib=True)`) needs
`lightgbm` importable at request time, which unconditionally imports
`scipy` (confirmed by reading `lightgbm/basic.py`: `import scipy.sparse`
is a bare top-level import, not inside a `try/except` the way its sklearn
integration is) -- scipy alone measures ~115MB installed, more than a
third of the entire 300MB serving-image budget (plan §13), for an
interpretability side-feature, not the headline prediction. A gain-
weighted feature salience (`(value / scale) * global_importance_weight`)
still varies per request -- values do, even though weights and scales
don't -- and needs only a small JSON file (`models/artifacts/
feature_importance.json`), no runtime ML library at all. Dividing by each
feature's training-set scale before weighting matters: without it, a
large-magnitude, whole-program feature (`node_count`) systematically
dominates a small-magnitude, span-bearing one (`max_loop_nesting_depth`)
on raw numbers alone, regardless of which one the model actually weighted
higher. This is a heuristic salience score, not a calibrated SHAP
contribution: it has no "sums to the model's output minus a baseline"
property. That tradeoff is the point, not an oversight.
"""
from __future__ import annotations

from core.ir import IRGraph
from models.gbdt import FEATURE_NAMES

Span = tuple[int, int, int, int]

# Which IR symbols a tabular feature's count is derived from
# (`features/tabular.py`'s own extraction) -- used only to look up spans to
# report alongside a feature's contribution, not to recompute it.
_FEATURE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "max_loop_nesting_depth": ("LOOP_FOR", "LOOP_WHILE"),
    "loop_count": ("LOOP_FOR", "LOOP_WHILE"),
    "loop_count_depth_0": ("LOOP_FOR", "LOOP_WHILE"),
    "loop_count_depth_1": ("LOOP_FOR", "LOOP_WHILE"),
    "loop_count_depth_2": ("LOOP_FOR", "LOOP_WHILE"),
    "loop_count_depth_3": ("LOOP_FOR", "LOOP_WHILE"),
    "max_alloc_nesting_depth": ("ARRAY_ALLOC", "HASH_ALLOC"),
    "alloc_count": ("ARRAY_ALLOC", "HASH_ALLOC"),
    "alloc_inside_loop_count": ("ARRAY_ALLOC", "HASH_ALLOC"),
    "recursion_call_count": ("RECURSE",),
    "recursion_shape_none": ("RECURSE",),
    "recursion_shape_single": ("RECURSE",),
    "recursion_shape_multiple": ("RECURSE",),
    "branch_count": ("BRANCH",),
    "break_count": ("BREAK",),
    "continue_count": ("CONTINUE",),
    "sort_call_count": ("SORT",),
    "binary_search_call_count": ("BINARY_SEARCH",),
    "heap_op_count": ("HEAP_PUSH", "HEAP_POP"),
    "math_op_count": ("MATH_OP",),
    "call_count": ("CALL",),
    "node_count": (),
    "edge_count": (),
}
assert set(_FEATURE_SYMBOLS) == set(FEATURE_NAMES), "every FEATURE_NAMES entry needs a span mapping"


def spans_for_feature(ir: IRGraph, feature_name: str, max_spans: int = 5) -> list[Span]:
    symbols = _FEATURE_SYMBOLS.get(feature_name, ())
    if not symbols:
        return []
    return [n.span for n in ir.nodes if n.symbol in symbols][:max_spans]


def top_contributions(
    feature_values: dict[str, float],
    importance_weights: dict[str, float],
    feature_scales: dict[str, float],
    ir: IRGraph,
    *,
    top_k: int = 5,
    min_abs_contribution: float = 1e-9,
) -> list[dict[str, object]]:
    """`importance_weights` are the production GBDT's normalised (summing
    to 1) global feature importances; `feature_scales` are each feature's
    training-set standard deviation (both from `models/artifacts/
    feature_importance.json`); `feature_values` are this one request's own
    `features/tabular.py` values. Dividing by scale before weighting keeps
    a large-magnitude, whole-program feature (`node_count`) from
    dominating a small-magnitude, span-bearing one
    (`max_loop_nesting_depth`) purely on raw numbers."""
    scored = [
        (name, importance_weights.get(name, 0.0) * (value / feature_scales.get(name, 1.0)))
        for name, value in feature_values.items()
    ]
    ranked = sorted(scored, key=lambda kv: -abs(kv[1]))
    contributions: list[dict[str, object]] = []
    for name, value in ranked:
        if abs(value) < min_abs_contribution:
            continue
        contributions.append(
            {
                "feature": name,
                "contribution": round(float(value), 6),
                "spans": [list(span) for span in spans_for_feature(ir, name)],
            }
        )
        if len(contributions) >= top_k:
            break
    return contributions
