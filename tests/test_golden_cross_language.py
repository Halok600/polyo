"""The golden cross-language IR test (plan §5, §14, §15/§16).

The same algorithm, written in all six supported languages, must produce
near-identical IR symbol histograms under the normalised vocabulary in
`core/ir.py`. This is what validates the project's central premise -- if it
fails, the cross-language architecture needs rethinking, not patching
around. Phase 1 shipped this for Python + C++ only; Phase 4 extends it to
Java, JavaScript, C and Go now that all six have IR mappings.

"Near-identical" is defined precisely, not left to eyeballing:

- Exact match on the *structural core* -- FUNC_DEF, PARAM, LOOP_FOR, BLOCK,
  BRANCH, ARRAY_INDEX, RETURN -- the symbols that describe control flow and
  data access shape, independent of how a language spells a loop bound.
- A documented, expected divergence on COMPARE/LITERAL/IDENT/ASSIGN:
  Python's `for i in range(n)` leaves the loop's start (0) and bound-check
  (`i < n`) implicit, while every other language must spell out an explicit
  `i = 0` init and `i < n` condition -- one more COMPARE, one more LITERAL,
  and (since declaring `i` re-emits it as an IDENT beyond its uses in the
  condition/update) two more IDENT than Python. Java's typed local variable
  declaration (`int i = 0`) emits a third extra IDENT beyond that shared
  gap. Go's `i := 0` additionally maps to ASSIGN (like any other Go
  `:=`/`=`), a gap symbol none of the other five have. That gap is real and
  reflects a language-idiom difference, not a bug in the mapping -- so it is
  asserted as an expected, bounded delta rather than hidden. See plan §16
  risk #1.
- A whole-histogram cosine similarity floor per language, so a real
  divergence in a mapping (as opposed to a documented idiom gap) would still
  fail the test.
"""
from __future__ import annotations

import math
from collections import Counter

from parsing.normalize import normalize_source

STRUCTURAL_CORE = ("FUNC_DEF", "PARAM", "LOOP_FOR", "BLOCK", "BRANCH", "ARRAY_INDEX", "RETURN")

LINEAR_SEARCH_SOURCE: dict[str, str] = {
    "python": """\
def linear_search(arr, n, target):
    for i in range(n):
        if arr[i] == target:
            return i
    return -1
""",
    "cpp": """\
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
    "c": """\
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
    "java": """\
class Sol {
    int linearSearch(int[] arr, int n, int target) {
        for (int i = 0; i < n; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
""",
    "javascript": """\
function linearSearch(arr, n, target) {
    for (let i = 0; i < n; i++) {
        if (arr[i] === target) {
            return i;
        }
    }
    return -1;
}
""",
    "go": """\
func linearSearch(arr []int, n int, target int) int {
    for i := 0; i < n; i++ {
        if arr[i] == target {
            return i
        }
    }
    return -1
}
""",
}

# Every language's expected symbol-count delta relative to Python for this
# snippet (see module docstring). Any symbol not listed here is expected to
# have zero delta -- checked below, not just the listed ones.
EXPECTED_IDIOM_GAP_VS_PYTHON: dict[str, dict[str, int]] = {
    "cpp": {"COMPARE": 1, "LITERAL": 1, "IDENT": 2},
    "c": {"COMPARE": 1, "LITERAL": 1, "IDENT": 2},
    "java": {"COMPARE": 1, "LITERAL": 1, "IDENT": 3},
    "javascript": {"COMPARE": 1, "LITERAL": 1, "IDENT": 2},
    "go": {"COMPARE": 1, "LITERAL": 1, "IDENT": 2, "ASSIGN": 1},
}


def _histograms() -> dict[str, Counter[str]]:
    return {
        lang: normalize_source(src, lang).symbol_histogram()
        for lang, src in LINEAR_SEARCH_SOURCE.items()
    }


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    symbols = set(a) | set(b)
    dot = sum(a[s] * b[s] for s in symbols)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_linear_search_structural_core_matches_exactly_across_all_six_languages():
    hists = _histograms()
    reference = hists["python"]
    for lang, hist in hists.items():
        for symbol in STRUCTURAL_CORE:
            assert hist[symbol] == reference[symbol], (
                f"{lang}.{symbol}={hist[symbol]} vs python.{symbol}={reference[symbol]} -- "
                "structural core must align exactly across all six languages"
            )


def test_linear_search_full_histogram_is_near_identical_by_cosine_similarity():
    hists = _histograms()
    reference = hists["python"]
    for lang, hist in hists.items():
        if lang == "python":
            continue
        similarity = _cosine_similarity(reference, hist)
        assert similarity >= 0.9, (
            f"python vs {lang}: cosine similarity {similarity:.3f} below 0.9 floor"
        )


def test_linear_search_documented_loop_bound_idiom_gap_is_bounded():
    hists = _histograms()
    reference = hists["python"]
    for lang, expected_gap in EXPECTED_IDIOM_GAP_VS_PYTHON.items():
        hist = hists[lang]
        for symbol in set(reference) | set(hist):
            expected_delta = expected_gap.get(symbol, 0)
            actual_delta = hist[symbol] - reference[symbol]
            assert actual_delta == expected_delta, (
                f"{lang}.{symbol}: expected delta {expected_delta} vs python, got {actual_delta}"
            )
