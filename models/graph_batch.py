"""IR graph -> batched tensor representation for the GNN (rung 3, plan §8).

No torch-geometric/dgl is installed here (confirmed absent, not an
oversight) -- and that's a deliberate fit with plan §8's "serving without
torch" goal, which reimplements the message pass in ~50 lines of numpy for
Phase 6. A hand-written tensor representation, not a graph-library
abstraction, is what that export actually has to mirror, so this module
builds one directly: the standard block-diagonal batching trick, where every
graph in a mini-batch is concatenated along one long node axis and each
graph's edge indices are offset by the running node count, so one
message-passing step processes an entire batch with no python loop over
individual graphs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from core.ir import EDGE_KINDS, IR_SYMBOLS, IRGraph

SYMBOL_INDEX: dict[str, int] = {s: i for i, s in enumerate(IR_SYMBOLS)}
NUM_SYMBOLS = len(IR_SYMBOLS)
ALL_EDGE_KINDS: tuple[str, ...] = tuple(sorted(EDGE_KINDS))

_EMPTY_EDGES = np.zeros((0, 2), dtype=np.int64)


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
            edges_by_kind={kind: _EMPTY_EDGES for kind in ALL_EDGE_KINDS},
        )
    symbol_ids = np.array([SYMBOL_INDEX[n.symbol] for n in ir.nodes], dtype=np.int64)
    by_kind: dict[str, list[tuple[int, int]]] = {kind: [] for kind in ALL_EDGE_KINDS}
    for edge in ir.edges:
        by_kind[edge.kind].append((edge.src, edge.dst))
    edges_by_kind = {
        kind: (np.array(pairs, dtype=np.int64) if pairs else _EMPTY_EDGES)
        for kind, pairs in by_kind.items()
    }
    return ExampleGraph(symbol_ids=symbol_ids, edges_by_kind=edges_by_kind)


@dataclass(frozen=True, slots=True)
class GraphBatch:
    symbol_ids: torch.Tensor  # (total_nodes,) long
    batch_index: torch.Tensor  # (total_nodes,) long -- which graph each node belongs to
    edge_index: dict[str, torch.Tensor]  # kind -> (2, num_edges) long
    num_graphs: int
    num_nodes: int


def collate(
    graphs: list[ExampleGraph], edge_kinds: tuple[str, ...], device: torch.device
) -> GraphBatch:
    """Every `ExampleGraph` from `to_example_graph` already has >= 1 node;
    the check below is a defensive invariant for hand-built test fixtures,
    not the primary path -- a zero-node graph here would leave that graph's
    pooled row undefined (dividing by a node count of zero)."""
    if not graphs:
        raise ValueError("collate requires at least one graph")
    if any(g.num_nodes == 0 for g in graphs):
        raise ValueError("collate requires every graph to have at least one node")

    symbol_chunks = [g.symbol_ids for g in graphs]
    batch_chunks = [np.full(g.num_nodes, i, dtype=np.int64) for i, g in enumerate(graphs)]

    edge_chunks: dict[str, list[np.ndarray]] = {kind: [] for kind in edge_kinds}
    offset = 0
    for graph in graphs:
        for kind in edge_kinds:
            pairs = graph.edges_by_kind[kind]
            edge_chunks[kind].append(pairs + offset if len(pairs) else pairs)
        offset += graph.num_nodes

    symbol_ids = torch.from_numpy(np.concatenate(symbol_chunks)).to(device)
    batch_index = torch.from_numpy(np.concatenate(batch_chunks)).to(device)
    edge_index: dict[str, torch.Tensor] = {}
    for kind in edge_kinds:
        stacked = np.concatenate(edge_chunks[kind], axis=0) if edge_chunks[kind] else _EMPTY_EDGES
        # (num_edges, 2) -> (2, num_edges): the standard edge_index layout.
        edge_index[kind] = torch.from_numpy(np.ascontiguousarray(stacked.T)).to(device)

    return GraphBatch(
        symbol_ids=symbol_ids,
        batch_index=batch_index,
        edge_index=edge_index,
        num_graphs=len(graphs),
        num_nodes=offset,
    )
