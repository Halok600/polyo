"""Tests for IR graph -> batched tensor conversion (plan §8, rung 3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.ir import IREdge, IRGraph, IRNode
from models.graph_batch import ALL_EDGE_KINDS, SYMBOL_INDEX, collate, to_example_graph

_SPAN = (0, 0, 0, 0)


def _node(id_: int, symbol: str) -> IRNode:
    return IRNode(id=id_, symbol=symbol, span=_SPAN, text=symbol)


def test_to_example_graph_maps_symbols_and_groups_edges_by_kind():
    ir = IRGraph(
        nodes=[_node(0, "FUNC_DEF"), _node(1, "LOOP_FOR"), _node(2, "IDENT")],
        edges=[IREdge(0, 1, "AST_CHILD"), IREdge(1, 2, "AST_CHILD"), IREdge(1, 2, "LOOP_CARRY")],
    )
    graph = to_example_graph(ir)
    assert graph.num_nodes == 3
    expected_symbols = [SYMBOL_INDEX["FUNC_DEF"], SYMBOL_INDEX["LOOP_FOR"], SYMBOL_INDEX["IDENT"]]
    assert graph.symbol_ids.tolist() == expected_symbols
    assert graph.edges_by_kind["AST_CHILD"].tolist() == [[0, 1], [1, 2]]
    assert graph.edges_by_kind["LOOP_CARRY"].tolist() == [[1, 2]]
    assert graph.edges_by_kind["CALL_EDGE"].shape == (0, 2)


def test_to_example_graph_falls_back_to_a_single_unknown_node_when_empty():
    graph = to_example_graph(IRGraph())
    assert graph.num_nodes == 1
    assert graph.symbol_ids.tolist() == [SYMBOL_INDEX["UNKNOWN"]]
    assert all(edges.shape == (0, 2) for edges in graph.edges_by_kind.values())


def test_collate_offsets_edges_and_builds_batch_index():
    ir_a = IRGraph(
        nodes=[_node(0, "FUNC_DEF"), _node(1, "IDENT")], edges=[IREdge(0, 1, "AST_CHILD")]
    )
    ir_b = IRGraph(
        nodes=[_node(0, "FUNC_DEF"), _node(1, "IDENT"), _node(2, "IDENT")],
        edges=[IREdge(0, 1, "AST_CHILD"), IREdge(0, 2, "AST_CHILD")],
    )
    graphs = [to_example_graph(ir_a), to_example_graph(ir_b)]
    batch = collate(graphs, ALL_EDGE_KINDS, torch.device("cpu"))

    assert batch.num_graphs == 2
    assert batch.num_nodes == 5
    assert batch.batch_index.tolist() == [0, 0, 1, 1, 1]
    # ir_b's edges are offset by ir_a's 2 nodes: (0,1)->(2,3), (0,2)->(2,4).
    assert batch.edge_index["AST_CHILD"].tolist() == [[0, 2, 2], [1, 3, 4]]


def test_collate_rejects_empty_graph_list():
    with pytest.raises(ValueError):
        collate([], ALL_EDGE_KINDS, torch.device("cpu"))


def test_collate_rejects_a_zero_node_graph():
    from models.graph_batch import ExampleGraph

    zero_node = ExampleGraph(symbol_ids=np.zeros(0, dtype=np.int64), edges_by_kind={})
    with pytest.raises(ValueError):
        collate([zero_node], ALL_EDGE_KINDS, torch.device("cpu"))
