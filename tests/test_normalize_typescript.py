"""CST -> IR normalisation tests for TypeScript (plan §5, Tier 3 plan §3,
`parsing/normalize.py` driven by `parsing/lang/typescript.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_TS = """\
function linearSearch(arr: number[], n: number, target: number): number {
    for (let i = 0; i < n; i++) {
        if (arr[i] === target) {
            return i;
        }
    }
    return -1;
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_TS, "typescript")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_TS, "typescript")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_TS, "typescript")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    # `i < n` (loop) and `arr[i] === target` (branch): both comparisons.
    assert hist["COMPARE"] == 2


def test_normalize_parameter_type_annotations_do_not_leak_into_the_ir():
    # A parameter's whole subtree is swallowed by [params]'s leaf-type
    # match (see typescript.toml) -- its `: number[]`/`: number` type
    # annotation is never walked into, unlike a *return* type annotation
    # (see the next test), which sits outside the parameter container.
    ir = normalize_source(LINEAR_SEARCH_TS, "typescript")
    assert ir.symbol_histogram()["PARAM"] == 3


def test_normalize_return_type_annotations_grammar_quirk_is_confirmed_present():
    # TypeScript's grammar names its `number` *type* keyword's node the
    # same as a numeric *literal*'s -- a real, documented idiom quirk (see
    # tests/test_golden_cross_language.py), not a bug: this asserts the
    # quirk is still there rather than silently disappearing if the
    # grammar ever changes.
    src = "function f(): number {\n    return 1;\n}\n"
    ir = normalize_source(src, "typescript")
    # One LITERAL for the real `1`, one for the leaked `number` type token.
    assert ir.symbol_histogram()["LITERAL"] == 2


def test_normalize_detects_direct_recursion():
    src = (
        "function fact(n: number): number {\n"
        "    if (n <= 1) { return 1; }\n"
        "    return fact(n - 1);\n"
        "}\n"
    )
    ir = normalize_source(src, "typescript")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "function f(xs: number[]) { return xs.sort(); }\n"
    ir = normalize_source(src, "typescript")
    assert ir.symbol_histogram()["SORT"] == 1
