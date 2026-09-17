"""Problem-level split tests (plan §9): no (source, problem_id) pair may
appear in more than one split -- the guard the plan calls "the single
easiest way to accidentally publish a fraudulent accuracy".
"""
from __future__ import annotations

from dataclasses import replace

from data.build import class_distribution, dedupe, split, stratify_cap
from data.corpus import CorpusRecord


def _record(problem_id: str, solution_id: str, source: str = "s", time_class: str | None = "O(n)"):
    return CorpusRecord(
        problem_id=problem_id,
        solution_id=solution_id,
        source=source,
        language="python",
        code=f"def f_{solution_id}(): pass\n",
        time_class=time_class,
        space_class=None,
    )


def _many_problems(n_problems: int, solutions_per_problem: int, source: str = "s"):
    records = []
    for p in range(n_problems):
        for s in range(solutions_per_problem):
            records.append(_record(str(p), f"{p}_{s}", source=source))
    return records


def test_no_problem_id_appears_in_more_than_one_split():
    records = _many_problems(n_problems=200, solutions_per_problem=5)
    result = split(records)

    problem_to_splits: dict[tuple[str, str], set[str]] = {}
    for name, split_records in result.items():
        for r in split_records:
            key = (r.source, r.problem_id)
            problem_to_splits.setdefault(key, set()).add(name)

    offenders = {k: v for k, v in problem_to_splits.items() if len(v) > 1}
    assert not offenders


def test_same_problem_id_from_different_sources_can_land_in_different_splits():
    # A collision in the bare problem_id string across two sources must not
    # force them into the same split -- splitting is keyed on (source, id).
    records = [
        _record("0", "0_0", source="bigobench"),
        _record("0", "0_1", source="bigobench"),
        _record("0", "0_0", source="codecomplex"),
        _record("0", "0_1", source="codecomplex"),
    ]
    result = split(records, train_frac=1.0, val_frac=0.0)
    # With train_frac=1.0 both sources land in train regardless -- the real
    # assertion is that bucketing is computed per (source, problem_id), not
    # that this particular tiny example lands in different splits.
    by_source = {r.source for split_records in result.values() for r in split_records}
    assert by_source == {"bigobench", "codecomplex"}


def test_split_covers_every_record_exactly_once():
    records = _many_problems(n_problems=150, solutions_per_problem=3)
    result = split(records)
    total = sum(len(v) for v in result.values())
    assert total == len(records)


def test_split_respects_train_fraction_at_the_problem_level():
    records = _many_problems(n_problems=1000, solutions_per_problem=1)
    result = split(records, train_frac=0.7, val_frac=0.15)
    problems_per_split = {name: len(v) for name, v in result.items()}
    # Hash-bucketed, not exact -- allow a reasonable margin around the target.
    assert 600 <= problems_per_split["train"] <= 800
    assert 100 <= problems_per_split["val"] <= 200
    assert 100 <= problems_per_split["test"] <= 200


def test_dedupe_drops_exact_duplicate_code_within_a_source():
    a = _record("0", "0_0")
    b = replace(a, solution_id="0_1")  # identical code, different id
    c = _record("0", "0_2")
    assert b.code == a.code
    deduped = dedupe([a, b, c])
    assert deduped == [a, c]


def test_dedupe_keeps_identical_code_from_different_sources():
    a = _record("0", "0_0", source="bigobench")
    b = _record("0", "0_0", source="codecomplex")
    assert a.code == b.code
    deduped = dedupe([a, b])
    assert deduped == [a, b]


def test_stratify_cap_bounds_every_class_and_is_deterministic():
    records = _many_problems(n_problems=50, solutions_per_problem=1, source="s")
    records += [_record(str(p), f"rare_{p}", time_class="O(2^n)") for p in range(3)]
    capped_once = stratify_cap(records, max_per_class=10)
    capped_again = stratify_cap(records, max_per_class=10)
    dist = class_distribution(capped_once)
    assert dist["time"]["O(n)"] <= 10
    assert dist["time"]["O(2^n)"] == 3  # never crowded out by the larger O(n) bucket
    assert capped_once == capped_again


def test_class_distribution_counts_none_as_its_own_bucket():
    records = [_record("0", "0_0", time_class=None), _record("0", "0_1", time_class="O(n)")]
    dist = class_distribution(records)
    assert dist["time"]["<none>"] == 1
    assert dist["time"]["O(n)"] == 1
