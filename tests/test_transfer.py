"""Tests for the zero-shot cross-language transfer experiment (plan §9).
CPU-fast: tiny GNN config, a handful of examples across three languages.
"""
from __future__ import annotations

from pathlib import Path

from data.corpus import CorpusRecord
from eval.transfer import (
    TRAIN_LANGUAGES,
    filter_by_language,
    plot_transfer_heatmap,
    run_transfer_experiment,
)
from models.dataset import build_examples

_LINEAR_BY_LANG = {
    "python": "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n",
    "java": (
        "int f(int[] xs) {\n"
        "    int total = 0;\n"
        "    for (int x : xs) {\n"
        "        total += x;\n"
        "    }\n"
        "    return total;\n"
        "}\n"
    ),
    "cpp": (
        "int f(std::vector<int> xs) {\n"
        "    int total = 0;\n"
        "    for (int x : xs) {\n"
        "        total += x;\n"
        "    }\n"
        "    return total;\n"
        "}\n"
    ),
}
_GNN_KWARGS = {"hidden_dim": 8, "num_layers": 2, "batch_size": 8, "max_epochs": 2, "patience": 1}


def _record(language: str, i: int) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language=language,
        code=_LINEAR_BY_LANG[language],
        time_class="O(n)",
        space_class="O(1)",
    )


def _examples(languages: list[str], per_language: int = 6):
    records = []
    i = 0
    for language in languages:
        for _ in range(per_language):
            records.append(_record(language, i))
            i += 1
    examples, stats = build_examples(records)
    assert stats.failed == 0
    return examples


def test_filter_by_language_keeps_only_requested_languages():
    examples = _examples(["python", "java", "cpp"], per_language=2)
    filtered = filter_by_language(examples, TRAIN_LANGUAGES)
    assert {e.record.language for e in filtered} == {"python", "java"}
    assert len(filtered) == 4


def test_run_transfer_experiment_scores_every_language_present_in_test():
    train_examples = _examples(["python", "java"], per_language=6)
    test_examples = _examples(["python", "java", "cpp"], per_language=3)
    train_y = [e.record.time_class for e in train_examples]
    test_y = [e.record.time_class for e in test_examples]

    results = run_transfer_experiment(
        "time", train_examples, train_y, test_examples, test_y, gnn_kwargs=_GNN_KWARGS
    )
    assert set(results.keys()) == {"python", "java", "cpp"}
    assert results["cpp"].n == 3  # the genuinely zero-shot language


def test_plot_transfer_heatmap_writes_a_file(tmp_path):
    train_examples = _examples(["python", "java"], per_language=6)
    test_examples = _examples(["python", "cpp"], per_language=3)
    train_y = [e.record.time_class for e in train_examples]
    test_y = [e.record.time_class for e in test_examples]
    results = run_transfer_experiment(
        "time", train_examples, train_y, test_examples, test_y, gnn_kwargs=_GNN_KWARGS
    )
    out_path = Path(tmp_path) / "heatmap.png"
    plot_transfer_heatmap({"time": results}, out_path)
    assert out_path.is_file()
