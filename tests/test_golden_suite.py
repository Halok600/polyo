"""Loader contract for the golden suite (`eval/golden_suite.py`).

The suite is the safety net PolyO v2 is built on: the served GNN once predicted O(n^3) for
O(n^2) C++/Java/JS code because nothing in the test set could see a cross-language failure.
These tests pin the loader's validation so a malformed case can never silently weaken that net.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.golden_suite import GoldenCase, GoldenSuiteError, load_cases

_VALID = """
[[case]]
id = "sum_loop"
title = "Sum of an array"
time_class = "O(n)"
space_class = "O(1)"
time_expr = "O(n)"
space_expr = "O(1)"
tags = ["single-loop"]
rationale = "One pass over the input."

[case.sources]
python = '''
def total(nums):
    s = 0
    for x in nums:
        s += x
    return s
'''
cpp = "int total(vector<int>& a) { int s = 0; for (int x : a) s += x; return s; }"
java = "class S { int total(int[] a) { int s = 0; for (int x : a) s += x; return s; } }"
javascript = "function total(a) { let s = 0; for (const x of a) s += x; return s; }"
go = "func total(a []int) int { s := 0; for _, x := range a { s += x }; return s }"
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_cases_parses_every_field(tmp_path: Path) -> None:
    _write(tmp_path, "loops.toml", _VALID)
    (case,) = load_cases(tmp_path)
    assert isinstance(case, GoldenCase)
    assert case.id == "sum_loop"
    assert case.group == "loops"  # taken from the file stem
    assert case.tags == ("single-loop",)
    assert (case.time_class, case.space_class) == ("O(n)", "O(1)")
    assert set(case.sources) == {"python", "cpp", "java", "javascript", "go"}
    assert "for x in nums" in case.sources["python"]


def test_expected_applies_per_language_override(tmp_path: Path) -> None:
    text = _VALID + '\n[case.overrides.cpp]\ntime_class = "O(1)"\ntime_expr = "O(1)"\n'
    _write(tmp_path, "loops.toml", text)
    (case,) = load_cases(tmp_path)
    assert case.expected("python").time_class == "O(n)"
    cpp = case.expected("cpp")
    assert (cpp.time_class, cpp.time_expr) == ("O(1)", "O(1)")
    assert cpp.space_class == "O(1)"  # un-overridden fields fall through to the case default


def test_duplicate_ids_across_files_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "a.toml", _VALID)
    _write(tmp_path, "b.toml", _VALID)
    with pytest.raises(GoldenSuiteError, match="duplicate case id"):
        load_cases(tmp_path)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ('time_class = "O(n)"', 'time_class = "O(n^9)"', "time_class"),
        ('space_class = "O(1)"', 'space_class = "O(2^n)"', "space_class"),
        ('id = "sum_loop"', 'id = "Sum Loop"', "id"),
        ('time_expr = "O(n)"', 'time_expr = "linear"', "time_expr"),
    ],
)
def test_invalid_fields_are_rejected(
    tmp_path: Path, needle: str, replacement: str, message: str
) -> None:
    _write(tmp_path, "loops.toml", _VALID.replace(needle, replacement))
    with pytest.raises(GoldenSuiteError, match=message):
        load_cases(tmp_path)


def test_unknown_language_key_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "loops.toml", _VALID.replace("go = ", "cobol = "))
    with pytest.raises(GoldenSuiteError, match="language"):
        load_cases(tmp_path)


def test_every_case_must_cover_the_five_core_languages(tmp_path: Path) -> None:
    # Dropping Go would let a Go-only regression (the exact kind of gap that hid the original
    # bug) pass unnoticed, so the five core languages are mandatory; C is optional.
    text = "\n".join(line for line in _VALID.splitlines() if not line.startswith("go = "))
    _write(tmp_path, "loops.toml", text)
    with pytest.raises(GoldenSuiteError, match="missing"):
        load_cases(tmp_path)


def test_override_for_unknown_language_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "loops.toml", _VALID + '\n[case.overrides.cobol]\ntime_class = "O(1)"\n')
    with pytest.raises(GoldenSuiteError, match="override"):
        load_cases(tmp_path)
