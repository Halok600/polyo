"""IR -> tabular features for the rung-0 rule baseline (plan §8).

Phase 1 ships exactly the two features the rung-0 rule needs: max loop
nesting depth (time) and max allocation nesting depth (space) -- richer
feature families (loop-bound shape, recursion shape, library calls, ...)
are rung-2 work (plan §8), added once there is a model to feed them to.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.ir import IRGraph

_LOOP_SYMBOLS = frozenset({"LOOP_FOR", "LOOP_WHILE"})
_ALLOC_SYMBOLS = frozenset({"ARRAY_ALLOC", "HASH_ALLOC"})


@dataclass(frozen=True, slots=True)
class TabularFeatures:
    max_loop_nesting_depth: int
    max_alloc_nesting_depth: int


def _children_by_parent(ir: IRGraph) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for edge in ir.edges:
        if edge.kind == "AST_CHILD":
            children.setdefault(edge.src, []).append(edge.dst)
    return children


def _roots(ir: IRGraph) -> list[int]:
    has_parent = {edge.dst for edge in ir.edges if edge.kind == "AST_CHILD"}
    return [n.id for n in ir.nodes if n.id not in has_parent]


def extract_features(ir: IRGraph) -> TabularFeatures:
    children = _children_by_parent(ir)
    symbol_by_id = {n.id: n.symbol for n in ir.nodes}
    max_loop_depth = 0
    max_alloc_depth = 0

    def visit(node_id: int, loop_depth: int) -> None:
        nonlocal max_loop_depth, max_alloc_depth
        symbol = symbol_by_id[node_id]
        if symbol in _LOOP_SYMBOLS:
            loop_depth += 1
            max_loop_depth = max(max_loop_depth, loop_depth)
        elif symbol in _ALLOC_SYMBOLS:
            max_alloc_depth = max(max_alloc_depth, loop_depth)
        for child_id in children.get(node_id, []):
            visit(child_id, loop_depth)

    for root_id in _roots(ir):
        visit(root_id, 0)

    return TabularFeatures(
        max_loop_nesting_depth=max_loop_depth,
        max_alloc_nesting_depth=max_alloc_depth,
    )
