"""Programs the engine was NOT tuned against when they were written, kept as a regression net.

`tests/data/heldout_cases.json` holds ~170 LeetCode-style programs in all six languages, labelled
before the engine ever saw them (time and space under the project conventions). Each batch was run
once, every miss was either a real engine bug (fixed test-first, with a test elsewhere) or a label
slip (corrected here, with the reason in `note`). Because they were used for fixing afterwards they
no longer measure generalisation; they keep what was fixed fixed, next to the golden suite.

`expect == "unknown"` marks recursions no static argument can bound (quick sort over list
comprehensions, collatz, flattening nested lists, backtracking with a `used` array): the engine must
say so (certainty "unknown") or answer with a bound that is not too low, never a confident low class.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.engine import analyze

_CASES = json.loads(
    Path(__file__).with_name("data").joinpath("heldout_cases.json").read_text("utf-8")
)


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] + ":" + c["language"] for c in _CASES])
def test_heldout_program(case: dict[str, str]) -> None:
    result = analyze(case["source"], case["language"])
    if case["expect"] == "unknown":
        assert result.certainty == "unknown" or result.time.cls in ("O(2^n)", "O(n^3)")
        return
    assert (result.time.cls, result.space.cls) == (case["time_class"], case["space_class"])


def test_the_corpus_covers_every_language() -> None:
    assert {c["language"] for c in _CASES} == {"python", "cpp", "java", "javascript", "go", "c"}
    assert len(_CASES) >= 150
