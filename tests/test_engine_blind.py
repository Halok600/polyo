"""The blind corpus (`tests/data/blind_v1_cases.json`): 144 programs, 24 per language, written and
labelled by independent agents that never saw the engine (conventions in the labelling brief, not in
the engine's code).

First contact (recorded in `tests/data/blind_v1_first_contact.json`, which is never edited and is
what the served confidence is calibrated on): 117 of 144 programs had both classes right. Every
miss was an engine gap (none was a label slip); the ones that could be fixed were fixed test-first
(`tests/test_engine_blind_fixes.py`), so these tests now pin the fixed ones. What is still wrong is
listed in `KNOWN_LIMITATIONS` with the reason, as strict xfails: when one is fixed the XPASS fails
the suite and the entry has to go.

After the fixes this corpus is a regression net, not a measurement: the honest generalisation
number is the first-contact one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.engine import analyze

_DATA = Path(__file__).with_name("data")
_CASES = json.loads(_DATA.joinpath("blind_v1_cases.json").read_text("utf-8"))
_FIRST_CONTACT = json.loads(_DATA.joinpath("blind_v1_first_contact.json").read_text("utf-8"))

KNOWN_LIMITATIONS: dict[str, str] = {
    "reverse_words_backward_scan_cpp": "a backward scan that hands its pointer on and copies each "
    "word with substr: the outer scan's progress is not proven, so the answer is an assumed "
    "over-estimate",
    "all_subsequences_strings_javascript": "a list that doubles every round (`push` while "
    "iterating over its length at the start of the round): exponential growth through a "
    "container that feeds itself is not modelled",
    "heap_level_sums_python": "level widths that double every round form a geometric series the "
    "loop planner does not sum",
    "prime_factors_trial_python": "the bound `d * d <= n` while `n` shrinks inside the loop: the "
    "current value of a changing variable is not tracked",
    "add_digits_javascript": "a value that collapses to its digit sum each round: not tracked",
    "zigzag_tree_path_go": "loops over bit lengths of a value (`1 << depth <= label`) and a second "
    "loop over the depth the first one found",
    "number_of_provinces_java": "an explicit stack kept in an array with a top index "
    "(`stack[top++] = x`): the worklist is not recognised, so the answer is an over-estimate",
    "sorted_list_to_bst_javascript": "list halving by walking to the middle: reported unknown",
    "sort_list_go": "list halving with slow and fast pointers: the two halves look like one node "
    "reached twice, so the answer is a loose over-estimate",
    "merge_sort_tree_go": "the merge work of each call is read from rows of a table the same "
    "function fills: their length is assumed as long as the input, an over-estimate in time",
}


def _cases() -> list:
    params = []
    for case in _CASES:
        marks = []
        if case["id"] in KNOWN_LIMITATIONS:
            marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_LIMITATIONS[case["id"]]))
        params.append(pytest.param(case, id=case["id"], marks=marks))
    return params


@pytest.mark.parametrize("case", _cases())
def test_blind_program(case: dict) -> None:
    result = analyze(case["source"], case["language"])
    assert (result.time.cls, result.space.cls) == (case["time_class"], case["space_class"])


def test_the_corpus_is_balanced_and_the_first_contact_record_covers_it() -> None:
    assert len(_CASES) == 144
    assert {case["language"] for case in _CASES} == {
        "python",
        "cpp",
        "java",
        "javascript",
        "go",
        "c",
    }
    per_language = {case["language"] for case in _CASES}
    assert all(sum(c["language"] == lang for c in _CASES) == 24 for lang in per_language)
    assert set(_FIRST_CONTACT) == {case["id"] for case in _CASES}


def test_every_known_limitation_is_a_real_case() -> None:
    assert set(KNOWN_LIMITATIONS) <= {case["id"] for case in _CASES}


def test_the_first_contact_record_is_what_the_calibration_was_built_on() -> None:
    """The record is evidence: if someone re-records after fixing the engine, the served
    confidence would claim an accuracy the engine never had on unseen code."""
    from analysis.calibration import load_calibration

    certain = [r for r in _FIRST_CONTACT.values() if r["certainty"] == "certain"]
    assert load_calibration().stats("certain", "time").total == len(certain)
