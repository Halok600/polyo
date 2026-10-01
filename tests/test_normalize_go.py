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
    assert hist["LOOP_N_BOUND"] == 1
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


def test_normalize_detects_recursion_through_a_method_receiver_call():
    # Two real bugs this regression guards, found together via a live
    # adversarial test: (1) `s.fib(...)` wasn't matched against the bare
    # method name by strict equality; (2) Go's `method_declaration` has a
    # RECEIVER parameter list (`(s *Solution)`) that precedes the method's
    # own name in document order -- `_function_name`'s generic preorder walk
    # found the receiver variable's identifier ("s") first and returned
    # that as the "function name" instead of "fib", so even the qualified-
    # call leniency fix alone couldn't have caught this without also fixing
    # name extraction itself (now via the grammar's own `name` field).
    src = (
        "type Solution struct{}\n\n"
        "func (s *Solution) fib(n int) int {\n"
        "    if n <= 1 {\n        return n\n    }\n"
        "    return s.fib(n-1) + s.fib(n-2)\n"
        "}\n"
    )
    ir = normalize_source(src, "go")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 2
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "func s(a []int) { sort.Ints(a) }\n"
    ir = normalize_source(src, "go")
    assert ir.symbol_histogram()["SORT"] == 1
