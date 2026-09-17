"""The normalised cross-language intermediate representation (IR).

Every supported language is parsed with tree-sitter and mapped into this
~40-symbol vocabulary (see `parsing/lang/*.toml`, added in Phase 1+). All
features, graphs and models operate on the IR -- never on raw source -- so
there is one feature extractor and one model regardless of source language.

See plan SS5. The golden cross-language IR test (added in Phase 1) verifies
that the same algorithm in different languages produces near-identical
symbol histograms under this vocabulary -- that test is what validates the
project's central premise, and it must keep passing as languages are added.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class IRSymbol(str, Enum):
    # Structural
    FUNC_DEF = "FUNC_DEF"
    PARAM = "PARAM"
    RETURN = "RETURN"
    BLOCK = "BLOCK"
    CALL = "CALL"
    RECURSE = "RECURSE"
    # Control
    LOOP_FOR = "LOOP_FOR"
    LOOP_WHILE = "LOOP_WHILE"
    LOOP_CONST_BOUND = "LOOP_CONST_BOUND"
    LOOP_N_BOUND = "LOOP_N_BOUND"
    LOOP_HALVING = "LOOP_HALVING"
    BRANCH = "BRANCH"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    # Data
    ARRAY_ALLOC = "ARRAY_ALLOC"
    ARRAY_INDEX = "ARRAY_INDEX"
    HASH_ALLOC = "HASH_ALLOC"
    HASH_LOOKUP = "HASH_LOOKUP"
    HASH_INSERT = "HASH_INSERT"
    SET_OP = "SET_OP"
    LIST_APPEND = "LIST_APPEND"
    STRING_CONCAT = "STRING_CONCAT"
    SLICE = "SLICE"
    COPY = "COPY"
    # Library
    SORT = "SORT"
    BINARY_SEARCH = "BINARY_SEARCH"
    HEAP_PUSH = "HEAP_PUSH"
    HEAP_POP = "HEAP_POP"
    QUEUE_OP = "QUEUE_OP"
    MATH_OP = "MATH_OP"
    # Memory
    ALLOC_CONST = "ALLOC_CONST"
    ALLOC_N = "ALLOC_N"
    ALLOC_NESTED = "ALLOC_NESTED"
    # Misc
    ASSIGN = "ASSIGN"
    ARITH = "ARITH"
    COMPARE = "COMPARE"
    LITERAL = "LITERAL"
    IDENT = "IDENT"
    UNKNOWN = "UNKNOWN"


IR_SYMBOLS: tuple[str, ...] = tuple(s.value for s in IRSymbol)


class EdgeKind(str, Enum):
    AST_CHILD = "AST_CHILD"
    NEXT_SIBLING = "NEXT_SIBLING"
    DATA_DEP = "DATA_DEP"
    LOOP_CARRY = "LOOP_CARRY"
    CALL_EDGE = "CALL_EDGE"


EDGE_KINDS: frozenset[str] = frozenset(e.value for e in EdgeKind)

# (start_line, start_col, end_line, end_col) -- 0-indexed, tree-sitter convention
Span = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class IRNode:
    id: int
    symbol: str
    span: Span
    text: str


@dataclass(frozen=True, slots=True)
class IREdge:
    src: int
    dst: int
    kind: str


@dataclass(slots=True)
class IRGraph:
    nodes: list[IRNode] = field(default_factory=list)
    edges: list[IREdge] = field(default_factory=list)

    def symbol_histogram(self) -> Counter[str]:
        return Counter(n.symbol for n in self.nodes)


def _ast_children(graph: IRGraph) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for edge in graph.edges:
        if edge.kind == "AST_CHILD":
            children.setdefault(edge.src, []).append(edge.dst)
    return children


def _full_subtree_ids(children: dict[int, list[int]], root_id: int) -> list[int]:
    """`root_id`'s full AST_CHILD descendant set, unbounded -- a nested
    scope's nodes are included too. Used by `loop_carry_edges`, where that is
    the point (see its docstring); `data_dependency_edges` uses the bounded
    variant below instead."""
    ids: list[int] = []
    stack = list(children.get(root_id, []))
    while stack:
        node_id = stack.pop()
        ids.append(node_id)
        stack.extend(children.get(node_id, []))
    return ids


def _bounded_subtree_ids(
    children: dict[int, list[int]],
    nodes_by_id: dict[int, IRNode],
    root_id: int,
    boundary_symbol: str,
) -> list[int]:
    """`root_id`'s AST_CHILD descendants, stopping (but still including) at
    a nested node whose symbol equals `boundary_symbol` -- so a nested scope
    of the same kind (a nested FUNC_DEF inside this one) is excluded from
    the outer scope's own set."""
    ids: list[int] = []
    stack = list(children.get(root_id, []))
    while stack:
        node_id = stack.pop()
        ids.append(node_id)
        if nodes_by_id[node_id].symbol == boundary_symbol:
            continue
        stack.extend(children.get(node_id, []))
    return ids


def _ident_occurrences_by_text(
    nodes_by_id: dict[int, IRNode], node_ids: list[int]
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for node_id in node_ids:
        node = nodes_by_id[node_id]
        if node.symbol == "IDENT":
            groups.setdefault(node.text, []).append(node_id)
    for ids in groups.values():
        ids.sort()  # document order: ids are assigned preorder in parsing/normalize.py
    return groups


def loop_carry_edges(graph: IRGraph) -> list[IREdge]:
    """Approximates "loop variable -> its uses" (plan SS5) without real
    dataflow analysis: for each LOOP_FOR/LOOP_WHILE node, every identifier
    that occurs at least twice anywhere in that loop's full AST subtree gets
    a LOOP_CARRY edge from the loop node to each of those occurrences.

    Deliberately UNBOUNDED at nested loops (unlike `data_dependency_edges`'s
    function-scope boundary): a variable touched again inside a nested loop
    is still carried by the outer loop's own iteration -- that's exactly the
    nested-loop-complexity signal this edge exists to expose. Bounding at a
    nested loop would silently starve an outer loop whose body is just the
    inner loop of any LOOP_CARRY edges at all, which is the common case for
    the O(n^2)-shaped code this signal matters most for."""
    children = _ast_children(graph)
    nodes_by_id = {n.id: n for n in graph.nodes}
    edges: list[IREdge] = []
    for node in graph.nodes:
        if node.symbol not in ("LOOP_FOR", "LOOP_WHILE"):
            continue
        subtree_ids = _full_subtree_ids(children, node.id)
        for ids in _ident_occurrences_by_text(nodes_by_id, subtree_ids).values():
            if len(ids) < 2:
                continue
            edges.extend(IREdge(node.id, ident_id, "LOOP_CARRY") for ident_id in ids)
    return edges


def data_dependency_edges(graph: IRGraph) -> list[IREdge]:
    """Approximates def-use (plan SS5's own wording, "approximate def-use"):
    within each FUNC_DEF's subtree, consecutive occurrences of an
    identically-spelled IDENT are chained pairwise in document order.

    Bounded at nested FUNC_DEFs, unlike `loop_carry_edges`: two different
    variables that happen to share a name in unrelated function scopes must
    not be chained together, and (unlike a nested loop) a nested function's
    body is a genuinely separate scope in real semantics, so there's no
    starvation risk from excluding it here."""
    children = _ast_children(graph)
    nodes_by_id = {n.id: n for n in graph.nodes}
    edges: list[IREdge] = []
    for node in graph.nodes:
        if node.symbol != "FUNC_DEF":
            continue
        subtree_ids = _bounded_subtree_ids(children, nodes_by_id, node.id, "FUNC_DEF")
        for ids in _ident_occurrences_by_text(nodes_by_id, subtree_ids).values():
            edges.extend(IREdge(a, b, "DATA_DEP") for a, b in zip(ids, ids[1:], strict=False))
    return edges
