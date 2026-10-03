"""Integrity of the real golden suite (`eval/golden/*.toml`).

These tests are what make "fullproof" mean something: every source must be syntactically valid in
its own language (a typo in one language's translation would otherwise surface as a bogus engine
failure, or worse, hide a real one), and the suite must cover every complexity class and the
constructs that actually caused the shipped bugs.
"""
from __future__ import annotations

import ast
from collections import Counter

import pytest

from eval.golden_suite import ALL_LANGUAGES, GoldenCase, load_cases
from parsing.parse import parse_source

# The suite may only grow. Raise these floors whenever cases are added; never lower them.
MIN_CASES = 140
MIN_PER_TIME_CLASS = {
    "O(1)": 6,
    "O(log n)": 8,
    "O(n)": 15,
    "O(n log n)": 8,
    "O(n^2)": 14,
    "O(n^3)": 5,
    "O(2^n)": 5,
}
MIN_PER_SPACE_CLASS = {"O(1)": 20, "O(log n)": 4, "O(n)": 25, "O(n log n)": 0, "O(n^2)": 5}

_CASES = load_cases()
_SOURCE_PARAMS = [
    pytest.param(case, language, id=f"{case.id}-{language}")
    for case in _CASES
    for language in case.languages
]


def test_suite_is_not_shrinking() -> None:
    assert len(_CASES) >= MIN_CASES, f"golden suite has {len(_CASES)} cases, floor is {MIN_CASES}"


@pytest.mark.parametrize(("case", "language"), _SOURCE_PARAMS)
def test_source_is_syntactically_valid(case: GoldenCase, language: str) -> None:
    source = case.sources[language]
    if language == "python":
        ast.parse(source)  # raises SyntaxError with a precise location
        return
    tree = parse_source(source, language)
    assert not tree.root_node.has_error, (
        f"{case.id} [{language}] does not parse cleanly with tree-sitter"
    )


def test_every_time_class_is_covered() -> None:
    counts = Counter(case.time_class for case in _CASES)
    short = {c: (counts[c], need) for c, need in MIN_PER_TIME_CLASS.items() if counts[c] < need}
    assert not short, f"time classes below their floor (have, need): {short}"


def test_every_space_class_is_covered() -> None:
    counts = Counter(case.space_class for case in _CASES)
    short = {c: (counts[c], need) for c, need in MIN_PER_SPACE_CLASS.items() if counts[c] < need}
    assert not short, f"space classes below their floor (have, need): {short}"


def test_c_is_present_often_enough_to_matter() -> None:
    # C is optional per case (no hash maps, no growable strings), but if it were nearly absent a
    # C-only regression would hide, exactly like the 9-row C++ test cell that hid the GNN bug.
    with_c = sum(1 for case in _CASES if "c" in case.sources)
    assert with_c >= 30, f"only {with_c} cases include a C source"


def test_suite_has_cross_language_semantic_overrides_documented() -> None:
    # Overrides are legitimate (Go slices are views, C++ strings are mutable) but each one is a
    # place where "same algorithm" stops meaning "same label", so it must be explained.
    for case in _CASES:
        if case.overrides:
            assert "override" in case.rationale.lower() or "differs" in case.rationale.lower(), (
                f"{case.id} has per-language overrides but its rationale does not explain them"
            )


def test_all_languages_constant_matches_parser_support() -> None:
    from parsing.parse import SUPPORTED_LANGUAGES

    assert set(ALL_LANGUAGES) <= SUPPORTED_LANGUAGES
