from core.ir import EDGE_KINDS, IR_SYMBOLS, IREdge, IRGraph, IRNode


def test_ir_symbol_vocabulary_has_no_duplicates():
    assert len(IR_SYMBOLS) == len(set(IR_SYMBOLS))


def test_ir_symbol_vocabulary_matches_documented_size():
    # Plan SS5 documents "~40 symbols" across six categories.
    assert 35 <= len(IR_SYMBOLS) <= 45


def test_edge_kinds_match_documented_set():
    assert EDGE_KINDS == {
        "AST_CHILD",
        "NEXT_SIBLING",
        "DATA_DEP",
        "LOOP_CARRY",
        "CALL_EDGE",
    }


def test_ir_graph_symbol_histogram_counts_correctly():
    nodes = [
        IRNode(id=0, symbol="FUNC_DEF", span=(1, 0, 3, 0), text="def f(): ..."),
        IRNode(id=1, symbol="LOOP_FOR", span=(2, 4, 2, 20), text="for x in xs:"),
        IRNode(id=2, symbol="LOOP_FOR", span=(2, 4, 2, 20), text="for y in ys:"),
    ]
    edges = [
        IREdge(src=0, dst=1, kind="AST_CHILD"),
        IREdge(src=1, dst=2, kind="NEXT_SIBLING"),
    ]
    graph = IRGraph(nodes=nodes, edges=edges)
    hist = graph.symbol_histogram()
    assert hist["LOOP_FOR"] == 2
    assert hist["FUNC_DEF"] == 1
    assert hist["ARRAY_ALLOC"] == 0


def test_empty_ir_graph_has_empty_histogram():
    assert IRGraph().symbol_histogram() == {}
