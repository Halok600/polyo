"""CST -> IR normalisation tests for Kotlin (plan §5, Tier 3 plan §3,
`parsing/normalize.py` driven by `parsing/lang/kotlin.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_KOTLIN = """\
fun linearSearch(arr: IntArray, n: Int, target: Int): Int {
    for (i in 0 until n) {
        if (arr[i] == target) {
            return i
        }
    }
    return -1
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_KOTLIN, "kotlin")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_KOTLIN, "kotlin")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_KOTLIN, "kotlin")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    assert hist["COMPARE"] == 1


def test_normalize_range_for_loop_has_no_explicit_increment_arith():
    # `for (i in 0 until n)` is a range-for -- no explicit increment step,
    # unlike a C-style for-loop's `i++` (see rust.toml's own version of
    # this same property).
    ir = normalize_source(LINEAR_SEARCH_KOTLIN, "kotlin")
    assert ir.symbol_histogram()["ARITH"] == 0


def test_normalize_grammar_quirks_leak_return_type_and_until_as_ident():
    # Two unrelated, documented idiom quirks (see
    # tests/test_golden_cross_language.py): the return type annotation's
    # `Int` (a bare `identifier` inside an unmapped `user_type` wrapper)
    # and `until` (the range bound's infix-function name, structurally
    # just another identifier in this grammar) both leak into IDENT.
    ir = normalize_source(LINEAR_SEARCH_KOTLIN, "kotlin")
    assert ir.symbol_histogram()["IDENT"] == 9


def test_normalize_disambiguates_binary_expression_by_operator():
    src = "fun add(a: Int, b: Int): Int { return a + b }\n"
    ir = normalize_source(src, "kotlin")
    hist = ir.symbol_histogram()
    assert hist["ARITH"] == 1
    assert hist["COMPARE"] == 0


def test_normalize_detects_direct_recursion():
    src = "fun fact(n: Int): Int {\n    if (n <= 1) { return 1 }\n    return fact(n - 1)\n}\n"
    ir = normalize_source(src, "kotlin")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "fun f(xs: MutableList<Int>) { xs.sort() }\n"
    ir = normalize_source(src, "kotlin")
    assert ir.symbol_histogram()["SORT"] == 1
