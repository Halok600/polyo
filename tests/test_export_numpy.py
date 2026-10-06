"""Tests for the numpy GNN export/reimplementation (plan §8). The whole
point of `models/export_numpy.py` is that it computes the *identical*
forward pass to the trained torch model -- these tests train a tiny real
model and assert the numpy path matches torch's own output on the same
examples, not just that it runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.ir import IR_SYMBOLS
from data.corpus import CorpusRecord
from models.dataset import build_examples
from models.export_numpy import NumpyGnnModel, VocabularyMismatchError, export_gnn
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
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
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
    export_gnn(time_model.core, time_model.edge_kinds, IR_SYMBOLS, npz_path)
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
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
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
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
    assert npz_path.is_file()
    loaded = NumpyGnnModel.load(npz_path)
    assert loaded.num_layers == 3


def test_load_succeeds_when_saved_vocabulary_matches_expected(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path, expected_symbols=IR_SYMBOLS)
    assert numpy_model.num_layers == 3


def test_load_raises_when_saved_vocabulary_differs_from_expected(tmp_path):
    # The embedding table's row i means "whatever symbol was at position i
    # in IR_SYMBOLS at export time" -- if that order has since changed
    # (a symbol renamed, removed, or added), row lookups at serve time
    # would silently read the wrong symbol's embedding. This must be
    # caught loudly at load, not discovered later as a quality regression.
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    stale_vocabulary = ("SOME", "STALE", "VOCABULARY")
    export_gnn(model.core, model.edge_kinds, stale_vocabulary, npz_path)
    with pytest.raises(VocabularyMismatchError):
        NumpyGnnModel.load(npz_path, expected_symbols=IR_SYMBOLS)


def test_load_raises_for_an_artifact_exported_before_vocabulary_fingerprinting(tmp_path):
    # Simulates an artifact exported by an older `export_gnn` that never
    # wrote `_ir_symbols` at all -- must fail loudly and clearly, not with
    # a bare KeyError, and must not silently skip validation just because
    # the key happens to be absent.
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    state = model.core.state_dict()
    arrays = {key: tensor.detach().cpu().numpy() for key, tensor in state.items()}
    arrays["_edge_kinds"] = np.array(model.edge_kinds)
    np.savez(npz_path, **arrays)  # deliberately no "_ir_symbols" key
    with pytest.raises(VocabularyMismatchError):
        NumpyGnnModel.load(npz_path, expected_symbols=IR_SYMBOLS)


def test_load_skips_validation_when_no_expected_symbols_given(tmp_path):
    # Callers that don't care (or are exercising unrelated behavior, like
    # every other test in this file) must not be forced to pass this.
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, ("ANYTHING",), npz_path)
    NumpyGnnModel.load(npz_path)  # must not raise


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


# ------------------------------------------------------------------ pooling variants (phase 6)
def test_numpy_forward_matches_torch_for_mean_and_max_pooling(tmp_path):
    examples, time_labels = _synthetic_examples()
    space_labels = ["O(1)"] * len(examples)
    time_model, space_model = fit_multitask(
        examples, time_labels, space_labels, pooling="meanmax", **_FIT_KWARGS
    )
    npz_path = tmp_path / "gnn.npz"
    export_gnn(time_model.core, time_model.edge_kinds, IR_SYMBOLS, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path)
    assert numpy_model.pooling == "meanmax"

    torch_scores = time_model.decision_function(examples)
    for i, example in enumerate(examples):
        graph = to_example_graph(example.ir)
        numpy_out = numpy_model.forward(graph.symbol_ids, graph.edges_by_kind)
        np.testing.assert_allclose(numpy_out["time"], torch_scores[i], atol=1e-4, rtol=1e-4)


def test_a_mean_pooled_export_says_so_and_an_old_file_without_the_entry_is_mean_pooled(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
    assert NumpyGnnModel.load(npz_path).pooling == "mean"

    stripped = tmp_path / "old.npz"
    data = np.load(npz_path, allow_pickle=False)
    np.savez(stripped, **{k: data[k] for k in data.files if k != "_pooling"})
    assert NumpyGnnModel.load(stripped).pooling == "mean"


def test_a_single_task_model_can_use_mean_and_max_pooling_too(tmp_path):
    examples, labels = _synthetic_examples()
    model = fit_single_task(examples, labels, "time", pooling="meanmax", **_FIT_KWARGS)
    npz_path = tmp_path / "gnn.npz"
    export_gnn(model.core, model.edge_kinds, IR_SYMBOLS, npz_path)
    numpy_model = NumpyGnnModel.load(npz_path)
    torch_scores = model.decision_function(examples)
    graph = to_example_graph(examples[0].ir)
    out = numpy_model.forward(graph.symbol_ids, graph.edges_by_kind)
    np.testing.assert_allclose(out["time"], torch_scores[0], atol=1e-4, rtol=1e-4)


def test_the_served_artifacts_are_mean_pooled_or_say_otherwise():
    from api.models_registry import ARTIFACTS_DIR

    loaded = NumpyGnnModel.load(ARTIFACTS_DIR / "gnn.npz", expected_symbols=IR_SYMBOLS)
    assert loaded.pooling in ("mean", "meanmax")
