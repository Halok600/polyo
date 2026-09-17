"""Tests for the derived DATA_DEP / LOOP_CARRY edge builders in `core/ir.py`
(plan §5's "approximate def-use" and "loop variable -> its uses" edges,
needed for Phase 5's GNN and its graph-edge-types ablation). Built and
tested directly against hand-constructed `IRGraph` fixtures -- these
builders operate purely on graph topology (AST_CHILD edges + node
symbol/text), so no tree-sitter parse is needed to exercise them. See
`tests/test_normalize_python.py` for the same edges exercised through a
real parse, including CALL_EDGE (which needs the parser's callee-text
resolution and so isn't testable from a bare `IRGraph` fixture).
"""
from __future__ import annotations

from core.ir import IREdge, IRGraph, IRNode, data_dependency_edges, loop_carry_edges

_SPAN = (0, 0, 0, 0)


def _node(id_: int, symbol: str, text: str = "") -> IRNode:
    return IRNode(id=id_, symbol=symbol, span=_SPAN, text=text or symbol)


def _ast_edges(*pairs: tuple[int, int]) -> list[IREdge]:
    return [IREdge(src, dst, "AST_CHILD") for src, dst in pairs]


def test_loop_carry_links_loop_node_to_each_repeated_identifier_occurrence():
    # LOOP_FOR(0) -> [IDENT "i"(1), IDENT "i"(2), IDENT "j"(3)] -- "i" repeats,
    # "j" doesn't.
    nodes = [
        _node(0, "LOOP_FOR"),
        _node(1, "IDENT", "i"),
        _node(2, "IDENT", "i"),
        _node(3, "IDENT", "j"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2), (0, 3)))
    carried = {(e.src, e.dst) for e in loop_carry_edges(graph)}
    assert carried == {(0, 1), (0, 2)}
    assert all(e.kind == "LOOP_CARRY" for e in loop_carry_edges(graph))


def test_loop_carry_ignores_identifiers_that_occur_only_once():
    nodes = [_node(0, "LOOP_FOR"), _node(1, "IDENT", "x")]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1)))
    assert loop_carry_edges(graph) == []


def test_loop_carry_ignores_identifiers_outside_any_loop():
    nodes = [_node(0, "FUNC_DEF"), _node(1, "IDENT", "x"), _node(2, "IDENT", "x")]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2)))
    assert loop_carry_edges(graph) == []


def test_loop_carry_reaches_into_nested_loops_and_both_loops_see_the_variable():
    # LOOP_FOR(0) -> [IDENT "total"(1), LOOP_FOR(2) -> IDENT "total"(3)] --
    # the outer loop's own body has only one "total" occurrence directly, but
    # its full subtree (including the nested loop) has two, so it must still
    # get a LOOP_CARRY edge; the inner loop sees only its own one occurrence,
    # so no edge for it (needs >= 2 within its own subtree).
    nodes = [
        _node(0, "LOOP_FOR"),
        _node(1, "IDENT", "total"),
        _node(2, "LOOP_FOR"),
        _node(3, "IDENT", "total"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2), (2, 3)))
    carried = {(e.src, e.dst) for e in loop_carry_edges(graph)}
    assert carried == {(0, 1), (0, 3)}


def test_loop_carry_starves_bounded_reading_would_produce_for_pure_nested_loop():
    # An outer loop whose body is *only* the inner loop -- if LOOP_CARRY were
    # bounded at nested-loop boundaries (excluding the inner loop's own
    # descendants from the outer loop's set), the outer loop would see zero
    # identifiers of its own and get no LOOP_CARRY edges at all, even though
    # its iteration is exactly what makes the inner loop's repeated variable
    # an O(n^2)-relevant signal. This asserts the outer loop does NOT starve.
    nodes = [
        _node(0, "LOOP_FOR"),
        _node(1, "LOOP_FOR"),
        _node(2, "IDENT", "x"),
        _node(3, "IDENT", "x"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (1, 2), (1, 3)))
    carried_srcs = {e.src for e in loop_carry_edges(graph)}
    assert 0 in carried_srcs
    assert 1 in carried_srcs


def test_data_dependency_chains_consecutive_same_text_identifiers_in_document_order():
    # FUNC_DEF(0) -> IDENT "x"(1), IDENT "x"(2), IDENT "x"(3) -- a chain
    # 1->2, 2->3, never 1->3 directly.
    nodes = [
        _node(0, "FUNC_DEF"),
        _node(1, "IDENT", "x"),
        _node(2, "IDENT", "x"),
        _node(3, "IDENT", "x"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2), (0, 3)))
    deps = {(e.src, e.dst) for e in data_dependency_edges(graph)}
    assert deps == {(1, 2), (2, 3)}
    assert all(e.kind == "DATA_DEP" for e in data_dependency_edges(graph))


def test_data_dependency_does_not_cross_function_boundaries():
    # Two sibling FUNC_DEFs, each with their own "x" -- same text, different
    # scopes, must not be chained together.
    nodes = [
        _node(0, "FUNC_DEF"),
        _node(1, "IDENT", "x"),
        _node(2, "FUNC_DEF"),
        _node(3, "IDENT", "x"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (2, 3)))
    assert data_dependency_edges(graph) == []


def test_data_dependency_ignores_non_identifier_symbols_with_matching_text():
    # A PARAM node and a LITERAL node happening to share text with an IDENT
    # must not be pulled into the chain -- only IDENT-symbol nodes qualify.
    nodes = [
        _node(0, "FUNC_DEF"),
        _node(1, "PARAM", "x"),
        _node(2, "IDENT", "x"),
        _node(3, "LITERAL", "x"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2), (0, 3)))
    assert data_dependency_edges(graph) == []


def test_data_dependency_treats_nested_function_defs_independently():
    # FUNC_DEF(0) -> [IDENT "x"(1), FUNC_DEF(2) -> IDENT "x"(3)] -- the inner
    # function's "x" must not chain with the outer function's "x", unlike
    # LOOP_CARRY's deliberately-unbounded nested-loop reach.
    nodes = [
        _node(0, "FUNC_DEF"),
        _node(1, "IDENT", "x"),
        _node(2, "FUNC_DEF"),
        _node(3, "IDENT", "x"),
    ]
    graph = IRGraph(nodes=nodes, edges=_ast_edges((0, 1), (0, 2), (2, 3)))
    assert data_dependency_edges(graph) == []


def test_data_dependency_and_loop_carry_return_empty_for_empty_graph():
    assert loop_carry_edges(IRGraph()) == []
    assert data_dependency_edges(IRGraph()) == []
