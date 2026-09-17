"""CST -> IR normalisation tests for Python (plan §5, `parsing/normalize.py`
driven by `parsing/lang/python.toml`)."""
from __future__ import annotations

from parsing.normalize import normalize_source

LINEAR_SEARCH_PY = """\
def linear_search(arr, n, target):
    for i in range(n):
        if arr[i] == target:
            return i
    return -1
"""


def test_normalize_maps_function_def_and_params():
    ir = normalize_source(LINEAR_SEARCH_PY, "python")
    hist = ir.symbol_histogram()
    assert hist["FUNC_DEF"] == 1
    assert hist["PARAM"] == 3


def test_normalize_maps_loop_branch_and_return():
    ir = normalize_source(LINEAR_SEARCH_PY, "python")
    hist = ir.symbol_histogram()
    assert hist["LOOP_FOR"] == 1
    assert hist["BRANCH"] == 1
    assert hist["RETURN"] == 2
    assert hist["BLOCK"] == 3


def test_normalize_maps_array_index_and_compare():
    ir = normalize_source(LINEAR_SEARCH_PY, "python")
    hist = ir.symbol_histogram()
    assert hist["ARRAY_INDEX"] == 1
    assert hist["COMPARE"] == 1


def test_normalize_elides_range_call_as_loop_bound_sugar():
    ir = normalize_source(LINEAR_SEARCH_PY, "python")
    hist = ir.symbol_histogram()
    # `range` itself is not a CALL, but its argument `n` is still visited.
    assert hist["CALL"] == 0
    assert hist["UNKNOWN"] == 0


def test_normalize_detects_direct_recursion():
    src = "def fact(n):\n    if n <= 1:\n        return 1\n    return fact(n - 1)\n"
    ir = normalize_source(src, "python")
    hist = ir.symbol_histogram()
    assert hist["RECURSE"] == 1
    assert hist["CALL"] == 0


def test_normalize_maps_library_sort_call():
    src = "def f(xs):\n    return sorted(xs)\n"
    ir = normalize_source(src, "python")
    assert ir.symbol_histogram()["SORT"] == 1


def test_normalize_produces_ast_child_edges_between_emitted_nodes():
    ir = normalize_source("def f():\n    return 1\n", "python")
    kinds = {e.kind for e in ir.edges}
    assert "AST_CHILD" in kinds
    func_def = next(n for n in ir.nodes if n.symbol == "FUNC_DEF")
    block = next(n for n in ir.nodes if n.symbol == "BLOCK")
    ret = next(n for n in ir.nodes if n.symbol == "RETURN")
    edge_triples = {(e.src, e.dst, e.kind) for e in ir.edges}
    assert (func_def.id, block.id, "AST_CHILD") in edge_triples
    assert (block.id, ret.id, "AST_CHILD") in edge_triples


def test_normalize_adds_call_edge_from_recursive_call_site_to_its_own_func_def():
    src = "def fact(n):\n    if n <= 1:\n        return 1\n    return fact(n - 1)\n"
    ir = normalize_source(src, "python")
    func_def = next(n for n in ir.nodes if n.symbol == "FUNC_DEF")
    recurse = next(n for n in ir.nodes if n.symbol == "RECURSE")
    edge_triples = {(e.src, e.dst, e.kind) for e in ir.edges}
    assert (recurse.id, func_def.id, "CALL_EDGE") in edge_triples


def test_normalize_adds_call_edge_resolving_a_forward_reference():
    # `main` calls `helper`, defined later in the file -- CALL_EDGE
    # resolution must not depend on emission order.
    src = "def main():\n    return helper()\n\n\ndef helper():\n    return 1\n"
    ir = normalize_source(src, "python")
    helper_def = next(
        n for n in ir.nodes if n.symbol == "FUNC_DEF" and n.text.startswith("def helper")
    )
    call = next(n for n in ir.nodes if n.symbol == "CALL")
    edge_triples = {(e.src, e.dst, e.kind) for e in ir.edges}
    assert (call.id, helper_def.id, "CALL_EDGE") in edge_triples


def test_normalize_has_no_call_edge_for_an_unresolvable_library_call():
    src = "def f(xs):\n    return sorted(xs)\n"
    ir = normalize_source(src, "python")
    assert [e for e in ir.edges if e.kind == "CALL_EDGE"] == []


def test_normalize_adds_loop_carry_and_data_dep_edges_for_a_real_parse():
    ir = normalize_source(LINEAR_SEARCH_PY, "python")
    kinds = {e.kind for e in ir.edges}
    assert "LOOP_CARRY" in kinds
    assert "DATA_DEP" in kinds


def test_normalize_flattens_elided_wrapper_nodes_when_linking_ancestors():
    # `x = 1` is wrapped in an `expression_statement` node with no mapping in
    # python.toml -- ASSIGN must still attach to BLOCK, its nearest emitted
    # ancestor, skipping over the unmapped wrapper.
    ir = normalize_source("def f():\n    x = 1\n    return x\n", "python")
    block = next(n for n in ir.nodes if n.symbol == "BLOCK")
    assign = next(n for n in ir.nodes if n.symbol == "ASSIGN")
    edge_triples = {(e.src, e.dst, e.kind) for e in ir.edges}
    assert (block.id, assign.id, "AST_CHILD") in edge_triples
