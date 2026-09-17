"""Rung 3 tests: GNN message-passing + multi-task heads (plan §8).

Kept CPU-fast on purpose (tiny hidden_dim, few epochs, a few dozen examples)
-- there is no GPU in CI (see `.github/workflows/ci.yml`), and these must
still complete quickly there. Real full-corpus training runs are driven
separately via `eval/model_card.py`/`eval/ablations.py` on this dev
machine's GPU, never through pytest.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from data.corpus import CorpusRecord
from models.dataset import build_examples
from models.gnn import (
    _CLASSES_BY_DIMENSION,
    _GnnCore,
    fit_multitask,
    fit_single_task,
    ordinal_cross_entropy,
)
from models.graph_batch import collate, to_example_graph

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
_FIT_KWARGS = dict(hidden_dim=8, num_layers=2, batch_size=8, max_epochs=3, patience=2)


def _record(code: str, i: int, time_label: str, space_label: str | None) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=time_label,
        space_class=space_label,
    )


def _synthetic_examples(n_per_class: int = 6, with_space: bool = True):
    records = []
    i = 0
    for code, label in ((_CONSTANT, "O(1)"), (_LINEAR, "O(n)"), (_QUADRATIC, "O(n^2)")):
        for _ in range(n_per_class):
            space_label = "O(1)" if with_space else None
            records.append(_record(code, i, label, space_label))
            i += 1
    examples, stats = build_examples(records)
    assert stats.failed == 0
    time_labels = [r.time_class for r in records]
    space_labels = [r.space_class for r in records]
    return examples, time_labels, space_labels


def test_ordinal_cross_entropy_soft_targets_sum_to_one():
    logits = torch.randn(5, len(_CLASSES_BY_DIMENSION["time"]))
    true_idx = torch.tensor([0, 1, 2, 3, 4])
    loss = ordinal_cross_entropy(logits, true_idx, tau=1.0)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_ordinal_cross_entropy_prefers_correct_logits_over_wrong_ones():
    num_classes = len(_CLASSES_BY_DIMENSION["time"])
    true_idx = torch.tensor([2])
    confident_correct = F.one_hot(true_idx, num_classes).float() * 10
    confident_wrong = F.one_hot(torch.tensor([6]), num_classes).float() * 10
    assert ordinal_cross_entropy(confident_correct, true_idx, tau=1.0) < ordinal_cross_entropy(
        confident_wrong, true_idx, tau=1.0
    )


def test_ordinal_cross_entropy_penalizes_a_distant_wrong_class_more_than_a_neighbour():
    # True class is O(n) (rank 2 of 7). A neighbour prediction (O(log n),
    # rank 1) must cost less loss than a distant one (O(2^n), rank 6) --
    # that's the entire point of using an ordinal loss over plain CE.
    num_classes = len(_CLASSES_BY_DIMENSION["time"])
    true_idx = torch.tensor([2])
    neighbour_logits = F.one_hot(torch.tensor([1]), num_classes).float() * 10
    distant_logits = F.one_hot(torch.tensor([6]), num_classes).float() * 10
    assert ordinal_cross_entropy(neighbour_logits, true_idx, tau=1.0) < ordinal_cross_entropy(
        distant_logits, true_idx, tau=1.0
    )


def test_gnn_core_forward_produces_correct_head_shapes():
    examples, _, _ = _synthetic_examples(n_per_class=2)
    graphs = [to_example_graph(e.ir) for e in examples]
    core = _GnnCore(hidden_dim=8, num_layers=2, edge_kinds=("AST_CHILD",), heads=("time", "space"))
    batch = collate(graphs, ("AST_CHILD",), torch.device("cpu"))
    outputs = core(batch)
    assert outputs["time"].shape == (len(examples), len(_CLASSES_BY_DIMENSION["time"]))
    assert outputs["space"].shape == (len(examples), len(_CLASSES_BY_DIMENSION["space"]))


def test_fit_single_task_rejects_mismatched_lengths():
    examples, time_labels, _ = _synthetic_examples(n_per_class=1)
    with pytest.raises(ValueError):
        fit_single_task(examples, time_labels[:-1], "time", **_FIT_KWARGS)


def test_fit_single_task_rejects_unknown_dimension():
    examples, time_labels, _ = _synthetic_examples(n_per_class=1)
    with pytest.raises(ValueError):
        fit_single_task(examples, time_labels, "volume", **_FIT_KWARGS)


def test_fit_single_task_produces_a_model_with_the_full_taxonomy_as_classes():
    examples, time_labels, _ = _synthetic_examples()
    model = fit_single_task(examples, time_labels, "time", **_FIT_KWARGS)
    assert model.classes == list(_CLASSES_BY_DIMENSION["time"])


def test_fit_single_task_predict_proba_rows_sum_to_one():
    examples, time_labels, _ = _synthetic_examples()
    model = fit_single_task(examples, time_labels, "time", **_FIT_KWARGS)
    for dist in model.predict_proba(examples[:4]):
        assert abs(sum(dist.values()) - 1.0) < 1e-5


def test_fit_single_task_decision_function_shape_matches_classes():
    examples, time_labels, _ = _synthetic_examples()
    model = fit_single_task(examples, time_labels, "time", **_FIT_KWARGS)
    scores = model.decision_function(examples[:5])
    assert scores.shape == (5, len(model.classes))


def test_fit_single_task_with_validation_set_does_not_crash():
    examples, time_labels, _ = _synthetic_examples()
    model = fit_single_task(
        examples[:12], time_labels[:12], "time",
        val_examples=examples[12:], val_labels=time_labels[12:], **_FIT_KWARGS,
    )
    assert len(model.predict(examples[:3])) == 3


def test_fit_multitask_returns_two_models_over_a_shared_core():
    examples, time_labels, space_labels = _synthetic_examples()
    time_model, space_model = fit_multitask(examples, time_labels, space_labels, **_FIT_KWARGS)
    assert time_model.core is space_model.core
    assert time_model.dimension == "time"
    assert space_model.dimension == "space"


def test_fit_multitask_handles_examples_missing_one_dimensions_label():
    # Half the corpus has no space label at all (mirrors CodeComplex having
    # no space_class) -- the joint model must still train without crashing
    # and produce full-shaped predictions for both heads on every example.
    examples, time_labels, space_labels = _synthetic_examples(with_space=True)
    space_labels = [label if i % 2 == 0 else None for i, label in enumerate(space_labels)]
    time_model, space_model = fit_multitask(examples, time_labels, space_labels, **_FIT_KWARGS)
    time_scores = time_model.decision_function(examples)
    space_scores = space_model.decision_function(examples)
    assert time_scores.shape == (len(examples), len(time_model.classes))
    assert space_scores.shape == (len(examples), len(space_model.classes))


def test_fit_respects_a_restricted_edge_kind_subset():
    examples, time_labels, _ = _synthetic_examples(n_per_class=2)
    model = fit_single_task(examples, time_labels, "time", edge_kinds=("AST_CHILD",), **_FIT_KWARGS)
    assert model.edge_kinds == ("AST_CHILD",)
    assert model.decision_function(examples).shape[0] == len(examples)
