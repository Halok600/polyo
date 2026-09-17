"""Tests for the parallel synthetic generator (plan §7, §14 Phase 4,
`data/synth.py`).

`build_records` doesn't run the oracle -- labels are exact by construction --
but every record's code IS run through the real parser for its language
(parsing/normalize.py), which needs no compiler and so is fully checkable on
this dev machine even for C/C++/Java/Go: a snippet that isn't valid,
parseable code for its declared language is caught here, well before any of
those languages' toolchains would be needed to catch it another way.
"""
from __future__ import annotations

from core.taxonomy import SpaceClass, TimeClass
from data.synth import ALGORITHMS, build_records
from parsing.normalize import normalize_source

_LANGUAGES = ("python", "cpp", "c", "java", "javascript", "go")


def test_every_algorithm_covers_all_six_languages():
    for algo in ALGORITHMS:
        assert set(algo.code_by_language) == set(_LANGUAGES), algo.name


def test_build_records_count_matches_algorithms_times_languages():
    records = build_records()
    assert len(records) == len(ALGORITHMS) * len(_LANGUAGES)


def test_build_records_shares_problem_id_across_languages_per_algorithm():
    records = build_records()
    by_algo: dict[str, set[str]] = {}
    for r in records:
        by_algo.setdefault(r.problem_id, set()).add(r.language)
    assert len(by_algo) == len(ALGORITHMS)
    for problem_id, languages in by_algo.items():
        assert languages == set(_LANGUAGES), problem_id


def test_build_records_solution_id_is_unique():
    records = build_records()
    solution_ids = [r.solution_id for r in records]
    assert len(solution_ids) == len(set(solution_ids))


def test_build_records_labels_are_valid_taxonomy_values():
    records = build_records()
    time_values = {c.value for c in TimeClass}
    space_values = {c.value for c in SpaceClass}
    for r in records:
        assert r.time_class in time_values
        assert r.space_class is None or r.space_class in space_values


def test_build_records_source_is_synth():
    records = build_records()
    assert all(r.source == "synth" for r in records)


def test_every_algorithm_every_language_actually_parses_and_has_a_func_def():
    for algo in ALGORITHMS:
        for language, code in algo.code_by_language.items():
            ir = normalize_source(code, language)
            hist = ir.symbol_histogram()
            assert hist["FUNC_DEF"] >= 1, f"{algo.name}/{language}: no FUNC_DEF parsed"


def test_recursive_algorithm_ir_shows_recursion():
    fib = next(a for a in ALGORITHMS if a.name == "naive_fibonacci")
    for language, code in fib.code_by_language.items():
        ir = normalize_source(code, language)
        hist = ir.symbol_histogram()
        assert hist["RECURSE"] >= 1, f"naive_fibonacci/{language}: no RECURSE detected"


def test_taxonomy_coverage_includes_rare_classes():
    # plan §7's stated purpose: fill the classes real corpora barely
    # contain. Verified here, not assumed.
    time_classes = {a.time_class for a in ALGORITHMS}
    space_classes = {a.space_class for a in ALGORITHMS if a.space_class is not None}
    assert TimeClass.O_N3 in time_classes
    assert TimeClass.O_2N in time_classes
    assert SpaceClass.O_N_LOG_N in space_classes
