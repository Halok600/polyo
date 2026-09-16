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
