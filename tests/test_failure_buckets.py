"""Tests for the failure-bucket heuristics (plan §9)."""
from __future__ import annotations

from data.corpus import CorpusRecord
from eval.failure_buckets import BUCKETS, analyze_failures
from models.dataset import build_examples

_MEMOISED_FIB = (
    "def fib(n, cache):\n"
    "    if n in cache:\n"
    "        return cache[n]\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    result = fib(n - 1, cache) + fib(n - 2, cache)\n"
    "    cache[n] = result\n"
    "    return result\n"
)
_LIBRARY_SORT = "def f(xs):\n    return sorted(xs)\n"
_EARLY_EXIT = (
    "def f(xs, target):\n"
    "    for x in xs:\n"
    "        if x == target:\n"
    "            break\n"
    "    return 0\n"
)
_PLAIN_LINEAR = (
    "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
)


def _record(code: str, i: int, time_label: str) -> CorpusRecord:
    return CorpusRecord(
        problem_id=str(i),
        solution_id=f"{i}_0",
        source="test",
        language="python",
        code=code,
        time_class=time_label,
        space_class=None,
    )


def _build(codes_and_labels: list[tuple[str, str]]):
    records = [_record(code, i, label) for i, (code, label) in enumerate(codes_and_labels)]
    examples, stats = build_examples(records)
    assert stats.failed == 0
    return examples


def test_correctly_classified_examples_are_never_counted():
    examples = _build([(_PLAIN_LINEAR, "O(n)")])
    report = analyze_failures("time", examples, ["O(n)"], ["O(n)"])
    assert report.total_misclassified == 0
    assert all(count == 0 for count in report.counts.values())


def test_memoised_recursion_bucket_matches_a_recursive_hash_lookup_pattern():
    examples = _build([(_MEMOISED_FIB, "O(n)")])
    report = analyze_failures("time", examples, ["O(n)"], ["O(2^n)"])
    assert report.total_misclassified == 1
    assert report.counts["memoised_recursion"] == 1


def test_hidden_in_library_call_bucket_matches_a_sort_call():
    examples = _build([(_LIBRARY_SORT, "O(n log n)")])
    report = analyze_failures("time", examples, ["O(n log n)"], ["O(n)"])
    assert report.counts["hidden_in_library_call"] == 1


def test_input_dependent_early_exit_bucket_matches_a_break_in_a_loop():
    examples = _build([(_EARLY_EXIT, "O(n)")])
    report = analyze_failures("time", examples, ["O(n)"], ["O(1)"])
    assert report.counts["input_dependent_early_exit"] == 1


def test_on_vs_onlogn_boundary_bucket_only_fires_for_that_exact_pair():
    examples = _build([(_PLAIN_LINEAR, "O(n)")])
    report = analyze_failures("time", examples, ["O(n)"], ["O(n log n)"])
    assert report.counts["on_vs_onlogn_boundary"] == 1

    report_other_pair = analyze_failures("time", examples, ["O(n)"], ["O(n^2)"])
    assert report_other_pair.counts["on_vs_onlogn_boundary"] == 0


def test_on_vs_onlogn_boundary_bucket_never_fires_for_space_dimension():
    examples = _build([(_PLAIN_LINEAR, "O(n)")])
    report = analyze_failures("space", examples, ["O(n)"], ["O(n log n)"])
    assert report.counts["on_vs_onlogn_boundary"] == 0


def test_buckets_can_overlap_on_a_single_example():
    # Sorts inside a memoised-looking recursive call -- both buckets fire.
    combined_src = (
        "def f(n, cache, xs):\n"
        "    if n in cache:\n"
        "        return cache[n]\n"
        "    result = sorted(xs)\n"
        "    cache[n] = result\n"
        "    return f(n - 1, cache, xs)\n"
    )
    examples = _build([(combined_src, "O(n log n)")])
    report = analyze_failures("time", examples, ["O(n log n)"], ["O(1)"])
    assert report.counts["memoised_recursion"] == 1
    assert report.counts["hidden_in_library_call"] == 1


def test_report_covers_every_declared_bucket():
    examples = _build([(_PLAIN_LINEAR, "O(n)")])
    report = analyze_failures("time", examples, ["O(n)"], ["O(1)"])
    assert set(report.counts.keys()) == set(BUCKETS)
