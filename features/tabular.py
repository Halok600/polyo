"""IR -> tabular features (plan §8).

Rung 0 needs exactly two features (loop/alloc nesting depth); rung 2 (plan
§8's "IR features + LightGBM" row) needs the richer set below. Both live in
one dataclass/one extractor so there is a single feature contract read by
every tabular model, not a rung-0 one and a separate rung-2 one. New fields
default to values consistent with "no evidence of X", so `models/rule.py`
and its tests (which only ever construct the first two fields) are
unaffected.

Traversal is an explicit stack, not recursion, on purpose: this is run over
real, arbitrary competitive-programming solutions at training time (plan
§7), and a long chain of binary operators (`1 + 2 + 3 + ... + 500`) is a
real pattern that can exceed Python's default recursion limit if walked
recursively.

**Deliberately NOT implemented here, and why:** loop-bound *shape*
(constant / input-dependent / halving -- plan §8) would need extending
`parsing/normalize.py` beyond Phase 1's stable, golden-tested node-to-symbol
mapping (today one CST node always maps to one fixed IR symbol; bound shape
needs inspecting a loop's *children*, which is a different kind of rule).
Hash/set-lookup features are similarly deferred: Python's `d[k]` and
`arr[i]` are syntactically identical `subscript` nodes, and telling them
apart needs `d`'s type, which this project's static, no-execution design
does not attempt. Both are candidates for a later pass, not silently
approximated here.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.ir import IRGraph

_LOOP_SYMBOLS = frozenset({"LOOP_FOR", "LOOP_WHILE"})
_ALLOC_SYMBOLS = frozenset({"ARRAY_ALLOC", "HASH_ALLOC"})
_LIBRARY_SYMBOL_FIELDS = {
    "SORT": "sort_call_count",
    "BINARY_SEARCH": "binary_search_call_count",
    "HEAP_PUSH": "heap_op_count",
    "HEAP_POP": "heap_op_count",
    "MATH_OP": "math_op_count",
}
_LOOP_DEPTH_BUCKETS = 4  # counts at depth 0, 1, 2, "3+"


@dataclass(frozen=True, slots=True)
class TabularFeatures:
    max_loop_nesting_depth: int
    max_alloc_nesting_depth: int
    loop_count: int = 0
    loop_count_by_depth: tuple[int, int, int, int] = (0, 0, 0, 0)
    alloc_count: int = 0
    alloc_inside_loop_count: int = 0
    recursion_call_count: int = 0
    recursion_shape: str = "none"  # "none" | "single" | "multiple" -- see module note
    branch_count: int = 0
    break_count: int = 0
    continue_count: int = 0
    sort_call_count: int = 0
    binary_search_call_count: int = 0
    heap_op_count: int = 0
    math_op_count: int = 0
    call_count: int = 0
    node_count: int = 0
    edge_count: int = 0

    def as_dict(self) -> dict[str, float]:
        """Flat numeric feature vector for tabular models (rung 1/2) --
        `recursion_shape` is one-hot'd rather than left as a string."""
        d: dict[str, float] = {
            "max_loop_nesting_depth": self.max_loop_nesting_depth,
            "max_alloc_nesting_depth": self.max_alloc_nesting_depth,
            "loop_count": self.loop_count,
            "alloc_count": self.alloc_count,
            "alloc_inside_loop_count": self.alloc_inside_loop_count,
            "recursion_call_count": self.recursion_call_count,
            "branch_count": self.branch_count,
            "break_count": self.break_count,
            "continue_count": self.continue_count,
            "sort_call_count": self.sort_call_count,
            "binary_search_call_count": self.binary_search_call_count,
            "heap_op_count": self.heap_op_count,
            "math_op_count": self.math_op_count,
            "call_count": self.call_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }
        for depth, count in enumerate(self.loop_count_by_depth):
            d[f"loop_count_depth_{depth}"] = count
        for shape in ("none", "single", "multiple"):
            d[f"recursion_shape_{shape}"] = float(self.recursion_shape == shape)
        return d


def _children_by_parent(ir: IRGraph) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for edge in ir.edges:
        if edge.kind == "AST_CHILD":
            children.setdefault(edge.src, []).append(edge.dst)
    return children


def _roots(ir: IRGraph) -> list[int]:
    has_parent = {edge.dst for edge in ir.edges if edge.kind == "AST_CHILD"}
    return [n.id for n in ir.nodes if n.id not in has_parent]


def _recursion_shape(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "single"
    return "multiple"


def extract_features(ir: IRGraph) -> TabularFeatures:
    children = _children_by_parent(ir)
    symbol_by_id = {n.id: n.symbol for n in ir.nodes}

    max_loop_depth = 0
    max_alloc_depth = 0
    loop_count = 0
    loop_count_by_depth = [0, 0, 0, 0]
    alloc_count = 0
    alloc_inside_loop_count = 0
    recursion_call_count = 0
    branch_count = 0
    break_count = 0
    continue_count = 0
    call_count = 0
    library_counts = dict.fromkeys(set(_LIBRARY_SYMBOL_FIELDS.values()), 0)

    stack: list[tuple[int, int]] = [(root_id, 0) for root_id in _roots(ir)]
    while stack:
        node_id, loop_depth = stack.pop()
        symbol = symbol_by_id[node_id]

        if symbol in _LOOP_SYMBOLS:
            loop_count += 1
            loop_count_by_depth[min(loop_depth, _LOOP_DEPTH_BUCKETS - 1)] += 1
            loop_depth += 1
            max_loop_depth = max(max_loop_depth, loop_depth)
        elif symbol in _ALLOC_SYMBOLS:
            alloc_count += 1
            max_alloc_depth = max(max_alloc_depth, loop_depth)
            if loop_depth > 0:
                alloc_inside_loop_count += 1
        elif symbol == "RECURSE":
            recursion_call_count += 1
        elif symbol == "BRANCH":
            branch_count += 1
        elif symbol == "BREAK":
            break_count += 1
        elif symbol == "CONTINUE":
            continue_count += 1
        elif symbol == "CALL":
            call_count += 1
        elif symbol in _LIBRARY_SYMBOL_FIELDS:
            library_counts[_LIBRARY_SYMBOL_FIELDS[symbol]] += 1

        for child_id in children.get(node_id, []):
            stack.append((child_id, loop_depth))

    return TabularFeatures(
        max_loop_nesting_depth=max_loop_depth,
        max_alloc_nesting_depth=max_alloc_depth,
        loop_count=loop_count,
        loop_count_by_depth=(
            loop_count_by_depth[0],
            loop_count_by_depth[1],
            loop_count_by_depth[2],
            loop_count_by_depth[3],
        ),
        alloc_count=alloc_count,
        alloc_inside_loop_count=alloc_inside_loop_count,
        recursion_call_count=recursion_call_count,
        recursion_shape=_recursion_shape(recursion_call_count),
        branch_count=branch_count,
        break_count=break_count,
        continue_count=continue_count,
        sort_call_count=library_counts["sort_call_count"],
        binary_search_call_count=library_counts["binary_search_call_count"],
        heap_op_count=library_counts["heap_op_count"],
        math_op_count=library_counts["math_op_count"],
        call_count=call_count,
        node_count=len(ir.nodes),
        edge_count=len(ir.edges),
    )
