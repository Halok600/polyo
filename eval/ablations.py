"""Ablation studies (plan §9): feature families · IR symbols vs raw tokens ·
graph edge types · multi-task vs single-task. Each ablation retrains a
variant and compares macro-F1 against a full/baseline configuration on the
same held-out test split `eval/model_card.py` uses -- these are comparative
diagnostics, not calibrated served models, so none of them run temperature
scaling (plan §8's calibration story is rung-level, not ablation-level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from eval.report import Metrics, compute_metrics
from models import gbdt, gnn, tfidf
from models.dataset import ParsedExample, with_label
from models.gbdt import FEATURE_NAMES
from models.gnn import ALL_EDGE_KINDS

# Groups every name in `models.gbdt.FEATURE_NAMES` into the feature
# families plan §8 names for rung 2 -- checked below to be an exact
# partition (no leftover name, no name in two families), so this stays
# correct if `features/tabular.py`'s field set ever changes.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "loop": (
        "max_loop_nesting_depth",
        "loop_count",
        "loop_count_depth_0",
        "loop_count_depth_1",
        "loop_count_depth_2",
        "loop_count_depth_3",
    ),
    "allocation": ("max_alloc_nesting_depth", "alloc_count", "alloc_inside_loop_count"),
    "recursion": (
        "recursion_call_count",
        "recursion_shape_none",
        "recursion_shape_single",
        "recursion_shape_multiple",
    ),
    "control_flow": ("branch_count", "break_count", "continue_count"),
    "library_calls": (
        "sort_call_count",
        "binary_search_call_count",
        "heap_op_count",
        "math_op_count",
        "call_count",
    ),
    "graph_size": ("node_count", "edge_count"),
}

_partitioned = sorted(name for names in FEATURE_FAMILIES.values() for name in names)
if _partitioned != sorted(FEATURE_NAMES):
    raise AssertionError("FEATURE_FAMILIES must exactly partition models.gbdt.FEATURE_NAMES")

EDGE_TYPE_CONFIGS: dict[str, tuple[str, ...]] = {
    "structural_only": ("AST_CHILD", "NEXT_SIBLING"),
    "structural_plus_data_dep": ("AST_CHILD", "NEXT_SIBLING", "DATA_DEP"),
    "structural_plus_loop_carry": ("AST_CHILD", "NEXT_SIBLING", "LOOP_CARRY"),
    "structural_plus_call_edge": ("AST_CHILD", "NEXT_SIBLING", "CALL_EDGE"),
    "all_edges": ALL_EDGE_KINDS,
}


@dataclass(frozen=True, slots=True)
class AblationResult:
    name: str
    metrics: Metrics


def feature_family_ablation(
    dimension: str,
    classes: tuple[str, ...],
    train_ex: list[ParsedExample],
    train_y: list[str],
    test_ex: list[ParsedExample],
    test_y: list[str],
) -> list[AblationResult]:
    """Leave-one-family-out: for each feature family, trains rung 2 without
    it and reports the macro-F1 against the full-feature baseline -- which
    family's removal hurts most is the actual answer this ablation exists
    to give."""
    full_model = gbdt.fit(train_ex, train_y)
    full_metrics = compute_metrics(dimension, test_y, full_model.predict(test_ex), classes)
    results = [AblationResult("full", full_metrics)]
    for family, dropped in FEATURE_FAMILIES.items():
        subset = tuple(name for name in FEATURE_NAMES if name not in dropped)
        model = gbdt.fit(train_ex, train_y, feature_names=subset)
        metrics = compute_metrics(dimension, test_y, model.predict(test_ex), classes)
        results.append(AblationResult(f"without_{family}", metrics))
    return results


def _raw_source_text(example: ParsedExample) -> str:
    return example.record.code


def ir_symbols_vs_raw_tokens_ablation(
    dimension: str,
    classes: tuple[str, ...],
    train_ex: list[ParsedExample],
    train_y: list[str],
    test_ex: list[ParsedExample],
    test_y: list[str],
) -> list[AblationResult]:
    """Compares rung 1's IR-symbol-sequence TF-IDF against a raw-source-text
    TF-IDF baseline of the same shape (identical vectoriser/classifier
    hyperparameters -- only the input text differs) -- the ablation that
    actually tests the project's central thesis: does the normalised IR's
    structure carry the signal, or would raw source tokens do just as well?
    """
    ir_model = tfidf.fit(train_ex, train_y)
    ir_pred = ir_model.predict(test_ex)

    raw_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    raw_x = raw_vectorizer.fit_transform([_raw_source_text(e) for e in train_ex])
    raw_classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    raw_classifier.fit(raw_x, train_y)
    raw_test_x = raw_vectorizer.transform([_raw_source_text(e) for e in test_ex])
    raw_pred = list(raw_classifier.predict(raw_test_x))

    return [
        AblationResult("ir_symbol_tfidf", compute_metrics(dimension, test_y, ir_pred, classes)),
        AblationResult("raw_token_tfidf", compute_metrics(dimension, test_y, raw_pred, classes)),
    ]


def edge_type_ablation(
    train_ex: list[ParsedExample],
    train_time_y: list[str | None],
    train_space_y: list[str | None],
    test_ex: list[ParsedExample],
    test_time_y: list[str | None],
    test_space_y: list[str | None],
    *,
    val_ex: list[ParsedExample] | None = None,
    val_time_y: list[str | None] | None = None,
    val_space_y: list[str | None] | None = None,
    gnn_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[AblationResult]]:
    """Trains a multi-task GNN once per edge-kind subset (plan §9's "graph
    edge types" ablation) -- which edge kinds beyond the plain AST actually
    earn their keep."""
    kwargs = gnn_kwargs or {}
    time_test_ex, time_test_y = with_label(test_ex, test_time_y)
    space_test_ex, space_test_y = with_label(test_ex, test_space_y)

    results: dict[str, list[AblationResult]] = {"time": [], "space": []}
    for name, edge_kinds in EDGE_TYPE_CONFIGS.items():
        time_model, space_model = gnn.fit_multitask(
            train_ex,
            train_time_y,
            train_space_y,
            val_examples=val_ex,
            val_time_labels=val_time_y,
            val_space_labels=val_space_y,
            edge_kinds=edge_kinds,
            **kwargs,
        )
        time_metrics = compute_metrics(
            "time", time_test_y, time_model.predict(time_test_ex), tuple(time_model.classes)
        )
        space_metrics = compute_metrics(
            "space", space_test_y, space_model.predict(space_test_ex), tuple(space_model.classes)
        )
        results["time"].append(AblationResult(name, time_metrics))
        results["space"].append(AblationResult(name, space_metrics))
    return results


def multitask_vs_singletask_ablation(
    train_ex: list[ParsedExample],
    train_time_y: list[str | None],
    train_space_y: list[str | None],
    test_ex: list[ParsedExample],
    test_time_y: list[str | None],
    test_space_y: list[str | None],
    *,
    val_ex: list[ParsedExample] | None = None,
    val_time_y: list[str | None] | None = None,
    val_space_y: list[str | None] | None = None,
    gnn_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[AblationResult]]:
    """Multi-task (shared encoder, plan §8) vs. two independent single-task
    GNNs -- plan §9's ablation: "does joint training beat two independent
    models? Either answer is a result." Single-task training/eval uses
    `with_label`-filtered examples per dimension (rung 1/2's own
    convention); multi-task uses the full, unfiltered example set, since a
    partially-labelled example (no space_class, say) can still help its
    other head -- that's the actual capability being tested here."""
    kwargs = gnn_kwargs or {}
    time_test_ex, time_test_y = with_label(test_ex, test_time_y)
    space_test_ex, space_test_y = with_label(test_ex, test_space_y)

    mt_time_model, mt_space_model = gnn.fit_multitask(
        train_ex,
        train_time_y,
        train_space_y,
        val_examples=val_ex,
        val_time_labels=val_time_y,
        val_space_labels=val_space_y,
        **kwargs,
    )
    mt_time_metrics = compute_metrics(
        "time", time_test_y, mt_time_model.predict(time_test_ex), tuple(mt_time_model.classes)
    )
    mt_space_metrics = compute_metrics(
        "space", space_test_y, mt_space_model.predict(space_test_ex), tuple(mt_space_model.classes)
    )

    time_train_ex, time_train_y = with_label(train_ex, train_time_y)
    space_train_ex, space_train_y = with_label(train_ex, train_space_y)
    time_val_ex, time_val_y = (
        with_label(val_ex, val_time_y)
        if val_ex is not None and val_time_y is not None
        else (None, None)
    )
    space_val_ex, space_val_y = (
        with_label(val_ex, val_space_y)
        if val_ex is not None and val_space_y is not None
        else (None, None)
    )

    st_time_model = gnn.fit_single_task(
        time_train_ex,
        time_train_y,
        "time",
        val_examples=time_val_ex,
        val_labels=time_val_y,
        **kwargs,
    )
    st_space_model = gnn.fit_single_task(
        space_train_ex,
        space_train_y,
        "space",
        val_examples=space_val_ex,
        val_labels=space_val_y,
        **kwargs,
    )
    st_time_metrics = compute_metrics(
        "time", time_test_y, st_time_model.predict(time_test_ex), tuple(st_time_model.classes)
    )
    st_space_metrics = compute_metrics(
        "space", space_test_y, st_space_model.predict(space_test_ex), tuple(st_space_model.classes)
    )

    return {
        "time": [
            AblationResult("multi_task", mt_time_metrics),
            AblationResult("single_task", st_time_metrics),
        ],
        "space": [
            AblationResult("multi_task", mt_space_metrics),
            AblationResult("single_task", st_space_metrics),
        ],
    }
