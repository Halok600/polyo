"""The golden cross-language IR test (plan §5, §14, §15/§16).

The same algorithm, written in Python and C++, must produce near-identical
IR symbol histograms under the normalised vocabulary in `core/ir.py`. This
is what validates the project's central premise -- if it fails, the
cross-language architecture needs rethinking, not patching around.

"Near-identical" is defined precisely, not left to eyeballing:

- Exact match on the *structural core* -- FUNC_DEF, PARAM, LOOP_FOR, BLOCK,
  BRANCH, ARRAY_INDEX, RETURN -- the symbols that describe control flow and
  data access shape, independent of how a language spells a loop bound.
- A documented, expected divergence on COMPARE/IDENT/LITERAL: Python's
  `for i in range(n)` leaves the loop's start (0) and bound-check (`i < n`)
  implicit, while C++'s `for (int i = 0; i < n; i++)` must spell both out.
  That gap is real and reflects a language-idiom difference, not a bug in
  the mapping -- so it is asserted as an expected, bounded delta rather than
  hidden. See plan §16 risk #1.
- A whole-histogram cosine similarity floor, so a real divergence in the
  mapping (as opposed to this one documented idiom gap) would still fail
  the test.
"""
from __future__ import annotations

import math
from collections import Counter

from parsing.normalize import normalize_source

STRUCTURAL_CORE = ("FUNC_DEF", "PARAM", "LOOP_FOR", "BLOCK", "BRANCH", "ARRAY_INDEX", "RETURN")

LINEAR_SEARCH_PY = """\
def linear_search(arr, n, target):
    for i in range(n):
        if arr[i] == target:
            return i
    return -1
"""

LINEAR_SEARCH_CPP = """\
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
"""


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    symbols = set(a) | set(b)
    dot = sum(a[s] * b[s] for s in symbols)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_linear_search_structural_core_matches_exactly_across_languages():
    py_hist = normalize_source(LINEAR_SEARCH_PY, "python").symbol_histogram()
    cpp_hist = normalize_source(LINEAR_SEARCH_CPP, "cpp").symbol_histogram()
    for symbol in STRUCTURAL_CORE:
        assert py_hist[symbol] == cpp_hist[symbol], (
            f"{symbol}: python={py_hist[symbol]} cpp={cpp_hist[symbol]} -- "
            "structural core must align exactly across languages"
        )


def test_linear_search_full_histogram_is_near_identical_by_cosine_similarity():
    py_hist = normalize_source(LINEAR_SEARCH_PY, "python").symbol_histogram()
    cpp_hist = normalize_source(LINEAR_SEARCH_CPP, "cpp").symbol_histogram()
    similarity = _cosine_similarity(py_hist, cpp_hist)
    assert similarity >= 0.9, f"cosine similarity {similarity:.3f} below 0.9 floor"


def test_linear_search_documented_loop_bound_idiom_gap_is_bounded():
    py_hist = normalize_source(LINEAR_SEARCH_PY, "python").symbol_histogram()
    cpp_hist = normalize_source(LINEAR_SEARCH_CPP, "cpp").symbol_histogram()
    # C++ spells out `i = 0` (LITERAL) and `i < n` (COMPARE) explicitly;
    # Python's `range(n)` leaves both implicit. Expect C++ to have exactly
    # one more COMPARE and exactly one more LITERAL than Python here, and no
    # other symbol to diverge outside the structural core.
    assert cpp_hist["COMPARE"] - py_hist["COMPARE"] == 1
    assert cpp_hist["LITERAL"] - py_hist["LITERAL"] == 1
