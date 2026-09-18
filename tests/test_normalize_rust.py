"""CST -> IR normalisation tests for Rust (plan §5, Tier 3 plan §3,
`parsing/normalize.py` driven by `parsing/lang/rust.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_RUST = """\
fn linear_search(arr: &[i32], n: usize, target: i32) -> i32 {
    for i in 0..n {
        if arr[i] == target {
            return i as i32;
        }
    }
    return -1;
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_RUST, "rust")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_RUST, "rust")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_RUST, "rust")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    assert hist["COMPARE"] == 1


def test_normalize_range_for_loop_has_no_redundant_loop_variable_identifiers():
    # `for i in 0..n` is a range-for (like Python's `for i in range(n)`),
    # not a C-style for-loop -- no explicit init/increment re-mentions the
    # loop variable, so IDENT should not carry the +2/+3 bump every
    # C-style-for language in this project shows (see
    # tests/test_golden_cross_language.py's idiom-gap table).
    ir = normalize_source(LINEAR_SEARCH_RUST, "rust")
    assert ir.symbol_histogram()["IDENT"] == 7


def test_normalize_range_for_loop_has_no_explicit_increment_arith():
    # No `i++`-equivalent at all in a range-for -- ARITH stays at 0 for
    # this snippet, unlike every C-style-for language.
    ir = normalize_source(LINEAR_SEARCH_RUST, "rust")
    assert ir.symbol_histogram()["ARITH"] == 0


def test_normalize_disambiguates_binary_expression_by_operator():
    src = "fn add(a: i32, b: i32) -> i32 { return a + b; }\n"
    ir = normalize_source(src, "rust")
    hist = ir.symbol_histogram()
    assert hist["ARITH"] == 1
    assert hist["COMPARE"] == 0


def test_normalize_detects_direct_recursion():
    src = "fn fact(n: i32) -> i32 {\n    if n <= 1 { return 1; }\n    return fact(n - 1);\n}\n"
    ir = normalize_source(src, "rust")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "fn f(xs: &mut Vec<i32>) { xs.sort(); }\n"
    ir = normalize_source(src, "rust")
    assert ir.symbol_histogram()["SORT"] == 1
