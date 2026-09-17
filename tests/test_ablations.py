"""Tests for Phase 5's ablation harness (plan §9). Kept CPU-fast: tiny GNN
configs, few epochs, a small synthetic corpus -- these check that each
ablation runs end-to-end and returns the right shape, not that it converges.
"""
from __future__ import annotations

from data.corpus import CorpusRecord
from eval.ablations import (
    FEATURE_FAMILIES,
    edge_type_ablation,
    feature_family_ablation,
    ir_symbols_vs_raw_tokens_ablation,
    multitask_vs_singletask_ablation,
)
from eval.report import SPACE_CLASSES, TIME_CLASSES
from models.dataset import build_examples, space_labels, time_labels
from models.gbdt import FEATURE_NAMES

_CONSTANT = "def f(x):\n    return x + 1\n"
_LINEAR = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
_QUADRATIC = (
    "def f(xs):\n"
    "    total = 0\n"
    "    for x in xs:\n"
    "        for y in xs:\n"
    "            total += x * y\n"
    "    return total\n"
)
_GNN_KWARGS = {"hidden_dim": 8, "num_layers": 2, "batch_size": 8, "max_epochs": 2, "patience": 1}


def _record(code: str, i: int, time_label: str) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=time_label,
        space_class="O(1)",
    )


def _synthetic_examples(n_per_class: int = 4):
    records = []
    i = 0
    for code, label in ((_CONSTANT, "O(1)"), (_LINEAR, "O(n)"), (_QUADRATIC, "O(n^2)")):
        for _ in range(n_per_class):
            records.append(_record(code, i, label))
            i += 1
    examples, stats = build_examples(records)
    assert stats.failed == 0
    return examples, time_labels(examples), space_labels(examples)


def test_feature_families_exactly_partition_feature_names():
    covered = sorted(name for names in FEATURE_FAMILIES.values() for name in names)
    assert covered == sorted(FEATURE_NAMES)


def test_feature_family_ablation_returns_full_plus_one_per_family():
    examples, time_y, _ = _synthetic_examples()
    results = feature_family_ablation("time", TIME_CLASSES, examples, time_y, examples, time_y)
    names = {r.name for r in results}
    assert names == {"full"} | {f"without_{family}" for family in FEATURE_FAMILIES}
    for r in results:
        assert 0.0 <= r.metrics.accuracy <= 1.0


def test_ir_symbols_vs_raw_tokens_ablation_returns_both_variants():
    examples, time_y, _ = _synthetic_examples()
    results = ir_symbols_vs_raw_tokens_ablation(
        "time", TIME_CLASSES, examples, time_y, examples, time_y
    )
    assert {r.name for r in results} == {"ir_symbol_tfidf", "raw_token_tfidf"}


def test_multitask_vs_singletask_ablation_returns_two_variants_per_dimension():
    examples, time_y, space_y = _synthetic_examples()
    results = multitask_vs_singletask_ablation(
        examples, time_y, space_y, examples, time_y, space_y, gnn_kwargs=_GNN_KWARGS
    )
    assert set(results.keys()) == {"time", "space"}
    for dimension_results in results.values():
        assert {r.name for r in dimension_results} == {"multi_task", "single_task"}
        for r in dimension_results:
            assert r.metrics.classes == TIME_CLASSES or r.metrics.classes == SPACE_CLASSES


def test_edge_type_ablation_covers_every_configured_edge_set():
    from eval.ablations import EDGE_TYPE_CONFIGS

    examples, time_y, space_y = _synthetic_examples()
    results = edge_type_ablation(
        examples, time_y, space_y, examples, time_y, space_y, gnn_kwargs=_GNN_KWARGS
    )
    for dimension_results in results.values():
        assert {r.name for r in dimension_results} == set(EDGE_TYPE_CONFIGS)
