"""Tests for Phase 5 report's data-driven "reading these numbers" prose
(plan §9-adjacent: interpretation must follow the actual numbers, not be
hardcoded for one run -- these functions must give the right verdict
whichever way the numbers land, so tests exercise both directions).
"""
from __future__ import annotations

import numpy as np

from eval.ablations import AblationResult
from eval.phase5_report import (
    _reading_edge_types,
    _reading_feature_families,
    _reading_ir_vs_raw_tokens,
    _reading_multitask_vs_singletask,
)
from eval.report import Metrics


def _metrics(macro_f1: float) -> Metrics:
    return Metrics(
        dimension="time",
        accuracy=0.5,
        macro_f1=macro_f1,
        mean_ordinal_distance=1.0,
        ece=None,
        n=10,
        per_class_f1={},
        confusion=np.zeros((1, 1)),
        classes=("O(1)",),
    )


def test_reading_multitask_vs_singletask_names_whichever_side_wins():
    results = [
        AblationResult("multi_task", _metrics(0.30)),
        AblationResult("single_task", _metrics(0.40)),
    ]
    assert "**single_task**" in _reading_multitask_vs_singletask(results)

    results_flipped = [
        AblationResult("multi_task", _metrics(0.40)),
        AblationResult("single_task", _metrics(0.30)),
    ]
    assert "**multi_task**" in _reading_multitask_vs_singletask(results_flipped)


def test_reading_edge_types_says_all_edges_wins_when_it_does():
    results = [
        AblationResult("structural_only", _metrics(0.20)),
        AblationResult("all_edges", _metrics(0.50)),
    ]
    text = _reading_edge_types(results)
    assert "all_edges" in text
    assert "structural_only" not in text


def test_reading_edge_types_names_the_actual_winner_when_all_edges_loses():
    results = [
        AblationResult("structural_only", _metrics(0.20)),
        AblationResult("structural_plus_loop_carry", _metrics(0.60)),
        AblationResult("all_edges", _metrics(0.45)),
    ]
    text = _reading_edge_types(results)
    assert "**structural_plus_loop_carry**" in text
    assert "0.450" in text  # all_edges' own score is still quoted, not hidden


def test_reading_ir_vs_raw_tokens_names_whichever_side_wins_and_always_caveats():
    results = [
        AblationResult("ir_symbol_tfidf", _metrics(0.30)),
        AblationResult("raw_token_tfidf", _metrics(0.40)),
    ]
    text = _reading_ir_vs_raw_tokens(results)
    assert "**raw_token_tfidf**" in text
    assert "cross-language" in text  # the caveat is present regardless of winner


def test_reading_feature_families_names_the_family_with_the_biggest_drop():
    results = [
        AblationResult("full", _metrics(0.50)),
        AblationResult("without_loop", _metrics(0.45)),
        AblationResult("without_library_calls", _metrics(0.20)),
        AblationResult("without_recursion", _metrics(0.48)),
    ]
    text = _reading_feature_families(results)
    assert "**library_calls**" in text
