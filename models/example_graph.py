"""A parsed IR as plain numpy arrays: the input of the trained GNN and of its numpy serving twin.

Lives apart from `models/graph_batch.py`, which needs torch to batch graphs for training: the
serving image has no torch (plan section 13), and it must build exactly these arrays to score a
request. `models/graph_batch.py` re-exports everything here, so training code is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ir import EDGE_KINDS, IR_SYMBOLS, IRGraph

SYMBOL_INDEX: dict[str, int] = {s: i for i, s in enumerate(IR_SYMBOLS)}
NUM_SYMBOLS = len(IR_SYMBOLS)
ALL_EDGE_KINDS: tuple[str, ...] = tuple(sorted(EDGE_KINDS))

EMPTY_EDGES = np.zeros((0, 2), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class ExampleGraph:
    """One example's IR pre-converted to plain arrays -- computed once per
    example, not once per epoch, since it never changes across training
    steps. Always has at least one node (see `to_example_graph`'s fallback
    for an IR that produced none), so every example contributes exactly one
    row to any batch of predictions -- the same 1:1 contract
    `models/gbdt.py`/`models/tfidf.py` already guarantee."""

    symbol_ids: np.ndarray  # (num_nodes,) int64
    edges_by_kind: dict[str, np.ndarray]  # kind -> (num_edges, 2) int64

    @property
    def num_nodes(self) -> int:
        return len(self.symbol_ids)


def to_example_graph(ir: IRGraph) -> ExampleGraph:
    if not ir.nodes:
        # An IR with zero nodes is a rare, degenerate parse (e.g. a
        # genuinely empty function body) -- rather than dropping the
        # example (which would break the 1:1 examples-in/predictions-out
        # contract every other rung guarantees), it gets a single UNKNOWN
        # placeholder node with no edges, so pooling has something to average.
        return ExampleGraph(
            symbol_ids=np.array([SYMBOL_INDEX["UNKNOWN"]], dtype=np.int64),
            edges_by_kind={kind: EMPTY_EDGES for kind in ALL_EDGE_KINDS},
        )
    symbol_ids = np.array([SYMBOL_INDEX[n.symbol] for n in ir.nodes], dtype=np.int64)
    by_kind: dict[str, list[tuple[int, int]]] = {kind: [] for kind in ALL_EDGE_KINDS}
    for edge in ir.edges:
        by_kind[edge.kind].append((edge.src, edge.dst))
    edges_by_kind = {
        kind: (np.array(pairs, dtype=np.int64) if pairs else EMPTY_EDGES)
        for kind, pairs in by_kind.items()
    }
    return ExampleGraph(symbol_ids=symbol_ids, edges_by_kind=edges_by_kind)
