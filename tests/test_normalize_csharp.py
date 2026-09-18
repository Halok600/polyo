"""CST -> IR normalisation tests for C# (plan §5, Tier 3 plan §3,
`parsing/normalize.py` driven by `parsing/lang/csharp.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_CSHARP = """\
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
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_CSHARP, "csharp")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_CSHARP, "csharp")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_CSHARP, "csharp")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    assert hist["COMPARE"] == 2


def test_normalize_maps_postfix_increment_to_arith_directly():
    # No named "operator" field on postfix_unary_expression in this
    # grammar (unlike javascript.toml's update_expression) -- mapped
    # directly, same approach as java.toml's update_expression.
    ir = normalize_source(LINEAR_SEARCH_CSHARP, "csharp")
    assert ir.symbol_histogram()["ARITH"] == 1


def test_normalize_wrapping_class_name_is_the_only_extra_ident_beyond_java_parity():
    # C# requires a wrapping class ("class Sol { ... }"), same as Java --
    # this is the +1 IDENT (the class name) beyond the shared C-style-for
    # loop gap that gives C# the exact same total idiom gap as java.toml
    # (see tests/test_golden_cross_language.py).
    ir = normalize_source(LINEAR_SEARCH_CSHARP, "csharp")
    assert ir.symbol_histogram()["IDENT"] == 10


def test_normalize_detects_direct_recursion():
    src = "class C { int Fact(int n) { if (n <= 1) { return 1; } return Fact(n - 1); } }\n"
    ir = normalize_source(src, "csharp")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "class C { void f(int[] xs) { System.Array.Sort(xs); } }\n"
    ir = normalize_source(src, "csharp")
    assert ir.symbol_histogram()["SORT"] == 1
