"""CST -> IR normalisation tests for Go (plan §5, `parsing/normalize.py`
driven by `parsing/lang/go.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_GO = """\
func linearSearch(arr []int, n int, target int) int {
    for i := 0; i < n; i++ {
        if arr[i] == target {
            return i
        }
    }
    return -1
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_GO, "go")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_GO, "go")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_GO, "go")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    # `i < n` (loop) and `arr[i] == target` (branch): both comparisons.
    assert hist["COMPARE"] == 2


def test_normalize_disambiguates_binary_expression_by_operator():
    src = "func add(a int, b int) int { return a + b }\n"
    ir = normalize_source(src, "go")
    hist = ir.symbol_histogram()
    assert hist["ARITH"] == 1
    assert hist["COMPARE"] == 0


def test_normalize_detects_direct_recursion():
    src = "func fact(n int) int { if n <= 1 { return 1 }; return fact(n - 1) }\n"
    ir = normalize_source(src, "go")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "func s(a []int) { sort.Ints(a) }\n"
    ir = normalize_source(src, "go")
    assert ir.symbol_histogram()["SORT"] == 1
