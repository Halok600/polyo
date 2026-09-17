"""CST -> IR normalisation tests for C (plan §5, `parsing/normalize.py`
driven by `parsing/lang/c.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_C = """\
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_C, "c")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_C, "c")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_C, "c")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    # `i < n` (loop) and `arr[i] == target` (branch): both comparisons.
    assert hist["COMPARE"] == 2


def test_normalize_disambiguates_binary_expression_by_operator():
    src = "int add(int a, int b) { return a + b; }\n"
    ir = normalize_source(src, "c")
    hist = ir.symbol_histogram()
    assert hist["ARITH"] == 1
    assert hist["COMPARE"] == 0


def test_normalize_detects_direct_recursion():
    src = "int fact(int n) { if (n <= 1) { return 1; } return fact(n - 1); }\n"
    ir = normalize_source(src, "c")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "void f(int a[], int n) { qsort(a, n, 4, cmp); }\n"
    ir = normalize_source(src, "c")
    assert ir.symbol_histogram()["SORT"] == 1
