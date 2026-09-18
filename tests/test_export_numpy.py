"""Tests for the numpy GNN export/reimplementation (plan §8). The whole
point of `models/export_numpy.py` is that it computes the *identical*
forward pass to the trained torch model -- these tests train a tiny real
model and assert the numpy path matches torch's own output on the same
examples, not just that it runs.
"""
from __future__ import annotations

import numpy as np

from data.corpus import CorpusRecord
from models.dataset import build_examples
from models.export_numpy import NumpyGnnModel, export_gnn
from models.gnn import fit_multitask, fit_single_task
from models.graph_batch import to_example_graph

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
_RECURSIVE = "def f(n):\n    if n <= 1:\n        return 1\n    return f(n - 1) + f(n - 2)\n"
_FIT_KWARGS = {"hidden_dim": 8, "num_layers": 3, "batch_size": 8, "max_epochs": 3, "patience": 2}


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


def _synthetic_examples():
    codes_and_labels = [
        (_CONSTANT, "O(1)"),
        (_LINEAR, "O(n)"),
        (_QUADRATIC, "O(n^2)"),
        (_RECURSIVE, "O(2^n)"),
    ]
    records = [
        _record(code, i * 5 + j, label)
        for i, (code, label) in enumerate(codes_and_labels)
        for j in range(5)
    ]
    examples, stats = build_examples(records)
    assert stats.failed == 0
    return examples, [r.time_class for r in records]


def test_numpy_forward_matches_torch_single_task_model(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)

    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path)

    torch_scores = model.decision_function(examples)
    for i, example in enumerate(examples):
        graph = to_example_graph(example.ir)
        numpy_out = numpy_model.forward(graph.symbol_ids, graph.edges_by_kind)
        np.testing.assert_allclose(numpy_out["time"], torch_scores[i], atol=1e-4, rtol=1e-4)


def test_numpy_forward_matches_torch_multitask_model_on_both_heads(tmp_path):
    examples, time_labels = _synthetic_examples()
    space_labels = ["O(1)"] * len(examples)
    time_model, space_model = fit_multitask(examples, time_labels, space_labels, **_FIT_KWARGS)

    npz_path = tmp_path / "gnn.npz"
    export_gnn(time_model.core, time_model.edge_kinds, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path)
    assert set(numpy_model.heads) == {"time", "space"}

    torch_time_scores = time_model.decision_function(examples)
    torch_space_scores = space_model.decision_function(examples)
    for i, example in enumerate(examples):
        graph = to_example_graph(example.ir)
        numpy_out = numpy_model.forward(graph.symbol_ids, graph.edges_by_kind)
        np.testing.assert_allclose(numpy_out["time"], torch_time_scores[i], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(numpy_out["space"], torch_space_scores[i], atol=1e-4, rtol=1e-4)


def test_numpy_forward_matches_with_a_restricted_edge_kind_subset(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", edge_kinds=("AST_CHILD",), **_FIT_KWARGS)

    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path)
    assert numpy_model.edge_kinds == ("AST_CHILD",)

    torch_scores = model.decision_function(examples)
    for i, example in enumerate(examples):
        graph = to_example_graph(example.ir)
        numpy_out = numpy_model.forward(graph.symbol_ids, graph.edges_by_kind)
        np.testing.assert_allclose(numpy_out["time"], torch_scores[i], atol=1e-4, rtol=1e-4)


def test_export_gnn_writes_a_loadable_file(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "subdir" / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, npz_path)
    assert npz_path.is_file()
    loaded = NumpyGnnModel.load(npz_path)
    assert loaded.num_layers == 3


def test_numpy_model_never_imports_torch():
    import ast
    from pathlib import Path

    source = Path("models/export_numpy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "torch"
