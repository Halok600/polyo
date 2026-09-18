"""The golden cross-language IR test (plan §5, §14, §15/§16).

The same algorithm, written in all supported languages, must produce
near-identical IR symbol histograms under the normalised vocabulary in
`core/ir.py`. This is what validates the project's central premise -- if it
fails, the cross-language architecture needs rethinking, not patching
around. Phase 1 shipped this for Python + C++ only; Phase 4 extends it to
Java, JavaScript, C and Go (Tier 1/2); Phase 7 extends it again to
TypeScript, Rust, C#, Kotlin (Tier 3, plan §3: IR mapping file only, no
oracle) -- this test is exactly the "adding a language is a config file,
not a rewrite" claim, made concrete and checked in CI.

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

Tier 3's four languages introduce two new *kinds* of documented gap, both
measured against a real parse, not guessed:

- A **negative** delta: Rust's and Kotlin's `for x in <range>` loop has no
  explicit increment step at all (unlike a C-style `for`'s `i++`), so
  neither emits the ARITH that `i++` (or Python's own unary-minus-on-
  literal, which *is* mapped to ARITH in `python.toml`) contributes --
  `ARITH: -1` relative to Python for both.
- A **grammar-quirk** delta with no runtime-semantic meaning: TypeScript's
  tree-sitter grammar names its `number` *type* keyword's node the same
  as a numeric *literal*'s (`predefined_type -> number`, verified against
  a real parse) -- the return type annotation's `number` token gets mapped
  to LITERAL right along with genuine literals, one extra LITERAL that
  carries no algorithmic meaning. Parameter type annotations don't leak
  this way (a parameter's whole subtree is swallowed by `[params]`'s
  leaf-type match and never walked into) -- only a function's *return*
  type annotation is walked normally and can leak. Kotlin has the same
  shape of leak for its return type, but through IDENT instead: a
  `user_type` node wraps a bare `identifier` for the type name (`Int`),
  with no [nodes] entry of its own to stop it flattening through.
  Kotlin's `until` (its range-loop bound, an infix *function* named
  `until`, not a dedicated operator token) is a second, unrelated +1
  IDENT for the same underlying reason -- a keyword that is, structurally,
  just another identifier in this grammar.
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
    "typescript": """\
function linearSearch(arr: number[], n: number, target: number): number {
    for (let i = 0; i < n; i++) {
        if (arr[i] === target) {
            return i;
        }
    }
    return -1;
}
""",
    "rust": """\
fn linear_search(arr: &[i32], n: usize, target: i32) -> i32 {
    for i in 0..n {
        if arr[i] == target {
            return i as i32;
        }
    }
    return -1;
}
""",
    "csharp": """\
class Sol {
    int LinearSearch(int[] arr, int n, int target) {
        for (int i = 0; i < n; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
""",
    "kotlin": """\
fun linearSearch(arr: IntArray, n: Int, target: Int): Int {
    for (i in 0 until n) {
        if (arr[i] == target) {
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
    # TypeScript's explicit C-style for-loop matches javascript's own gap
    # exactly, plus one grammar-quirk LITERAL (see module docstring: the
    # return type annotation's `number` keyword shares a node type with a
    # numeric literal).
    "typescript": {"COMPARE": 1, "LITERAL": 2, "IDENT": 2},
    # Rust's `for i in 0..n` is a range-for, like Python's `for i in
    # range(n)` -- no redundant loop-variable redeclaration, so IDENT has
    # zero delta (unlike every C-style-for language above). LITERAL +1 is
    # the explicit range start (`0`, elided in Python's `range(n)`).
    # ARITH -1: Rust's range-for has no explicit increment step at all
    # (unlike a C-style `i++`), and Python's own ARITH=1 here comes from
    # its unary-minus-on-literal (`python.toml` maps `unary_operator` to
    # ARITH) -- Rust has no such mapping, so this snippet's "-1" is a bare
    # LITERAL for Rust, contributing nothing to ARITH.
    "rust": {"LITERAL": 1, "ARITH": -1},
    # C# matches java's own gap exactly, and for the same reason: both
    # need a wrapping class ("class Sol { ... }"), which is where the
    # extra +1 IDENT (the class name) beyond the shared C-style-for-loop
    # gap comes from.
    "csharp": {"COMPARE": 1, "LITERAL": 1, "IDENT": 3},
    # Kotlin's `for (i in 0 until n)` is a range-for like Rust's (ARITH -1,
    # same reason: no explicit increment step, and no unary-minus-to-ARITH
    # mapping here either). LITERAL +1 is the explicit range start, same
    # as Rust. IDENT +2 is two unrelated grammar-quirk leaks (module
    # docstring): the return type annotation's `Int` (a bare `identifier`
    # inside an unmapped `user_type` wrapper) and `until` (the range
    # bound's infix-function name, structurally just another identifier).
    "kotlin": {"IDENT": 2, "LITERAL": 1, "ARITH": -1},
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


def test_linear_search_structural_core_matches_exactly_across_every_language():
    hists = _histograms()
    reference = hists["python"]
    for lang, hist in hists.items():
        for symbol in STRUCTURAL_CORE:
            assert hist[symbol] == reference[symbol], (
                f"{lang}.{symbol}={hist[symbol]} vs python.{symbol}={reference[symbol]} -- "
                "structural core must align exactly across every language"
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
