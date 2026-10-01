"""CST -> IR normalisation tests for JavaScript (plan §5,
`parsing/normalize.py` driven by `parsing/lang/javascript.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_JS = """\
function linearSearch(arr, n, target) {
    for (let i = 0; i < n; i++) {
        if (arr[i] === target) {
            return i;
        }
    }
    return -1;
}
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_JS, "javascript")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_JS, "javascript")
    hist = ir.symbol_histogram()
    assert hist["LOOP_N_BOUND"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_JS, "javascript")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    # `i < n` (loop) and `arr[i] === target` (branch): both comparisons.
    assert hist["COMPARE"] == 2


def test_normalize_disambiguates_binary_expression_by_operator():
    src = "function add(a, b) { return a + b; }\n"
    ir = normalize_source(src, "javascript")
    hist = ir.symbol_histogram()
    assert hist["ARITH"] == 1
    assert hist["COMPARE"] == 0


def test_normalize_detects_direct_recursion():
    src = "function fact(n) { if (n <= 1) { return 1; } return fact(n - 1); }\n"
    ir = normalize_source(src, "javascript")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_detects_recursion_through_a_class_method_qualified_this_call():
    # Two real bugs this regression guards, found together via a live
    # adversarial test: (1) `this.fib(...)` wasn't matched against the bare
    # method name by strict equality: needs the same trailing-segment
    # leniency as `self.`/`this->` in every other served language; (2) a
    # class method's own name is a `property_identifier` node (unlike a
    # plain `function fib(n) {}`, whose name is a plain `identifier`) --
    # `_function_name`'s walk didn't recognise that type at all, so it fell
    # through the method name and silently returned the first PARAMETER's
    # identifier instead, meaning `self._func_name_stack[-1]` was wrong
    # regardless of the qualified-call leniency fix alone.
    src = (
        "class Solution {\n"
        "    fib(n) {\n"
        "        if (n <= 1) { return n; }\n"
        "        return this.fib(n - 1) + this.fib(n - 2);\n"
        "    }\n"
        "}\n"
    )
    ir = normalize_source(src, "javascript")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 2
    assert hist["CALL"] == 0


def test_normalize_maps_member_sort_call():
    src = "function s(a) { a.sort(); }\n"
    ir = normalize_source(src, "javascript")
    assert ir.symbol_histogram()["SORT"] == 1


def test_normalize_maps_arrow_function():
    ir = normalize_source("const add = (a, b) => a + b;\n", "javascript")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["ARITH"] == 1
