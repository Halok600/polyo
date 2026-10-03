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

from models.example_graph import (
    ALL_EDGE_KINDS,
    EMPTY_EDGES,
    NUM_SYMBOLS,
    SYMBOL_INDEX,
    ExampleGraph,
    to_example_graph,
)

# `ExampleGraph` and friends live in `models/example_graph.py` (numpy only, so the serving image
# can build them without torch); they are re-exported here for the training code.
__all__ = [
    "ALL_EDGE_KINDS",
    "NUM_SYMBOLS",
    "SYMBOL_INDEX",
    "ExampleGraph",
    "GraphBatch",
    "collate",
    "to_example_graph",
]


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
        stacked = np.concatenate(edge_chunks[kind], axis=0) if edge_chunks[kind] else EMPTY_EDGES
        # (num_edges, 2) -> (2, num_edges): the standard edge_index layout.
        edge_index[kind] = torch.from_numpy(np.ascontiguousarray(stacked.T)).to(device)

    return GraphBatch(
        symbol_ids=symbol_ids,
        batch_index=batch_index,
        edge_index=edge_index,
        num_graphs=len(graphs),
        num_nodes=offset,
    )
