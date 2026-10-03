"""The engine's language-neutral AST (`analysis/nodes.py`).

Why a separate AST: `core/ir.py` flattens programs into ~40 symbols and drops operators and
operands, which is exactly what loop-bound analysis needs (`i < n`, `i *= 2`, `mid = (lo+hi)//2`).
The engine therefore lowers each language to this small tree instead. These tests pin the rendering
(used in human-readable derivations) and the traversal helpers."""
from __future__ import annotations

from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    Call,
    Compare,
    Const,
    ExprStmt,
    FuncDef,
    If,
    Loop,
    Module,
    Name,
    Num,
    Param,
    Return,
    Slice,
    Subscript,
    TypeRef,
    UnOp,
    children,
    render_expr,
    render_module,
    walk,
)


def test_render_expr_is_python_like_and_parenthesises_nested_operations() -> None:
    expr = Compare("<", BinOp("+", Name("lo"), BinOp("//", BinOp("-", Name("hi"), Name("lo")), Num(2))), Name("n"))
    assert render_expr(expr) == "lo + (hi - lo) // 2 < n"


def test_render_expr_covers_calls_members_subscripts_and_slices() -> None:
    call = Call(Attribute(Name("heapq"), "heappush"), (Name("h"), Subscript(Name("a"), Name("i"))))
    assert render_expr(call) == "heapq.heappush(h, a[i])"
    assert render_expr(Slice(Name("s"), Name("i"), None, None)) == "s[i:]"
    assert render_expr(Slice(Name("s"), None, None, Num(-1))) == "s[::-1]"
    assert render_expr(UnOp("not", Name("done"))) == "not done"
    assert render_expr(Const("none")) == "None"


def test_render_expr_shows_floats_and_ints_plainly() -> None:
    assert render_expr(Num(26)) == "26"
    assert render_expr(Num(0.5)) == "0.5"


def _two_sum() -> Module:
    inner = Loop(
        kind="for_each",
        target=Name("j"),
        iter=Call(Name("range"), (BinOp("+", Name("i"), Num(1)), Name("n"))),
        body=(
            If(
                test=Compare("==", BinOp("+", Subscript(Name("nums"), Name("i")), Subscript(Name("nums"), Name("j"))), Name("target")),
                body=(Return(Name("i"), line=5),),
                orelse=(),
                line=4,
            ),
        ),
        line=3,
    )
    outer = Loop(
        kind="for_each",
        target=Name("i"),
        iter=Call(Name("range"), (Name("n"),)),
        body=(inner,),
        line=2,
    )
    func = FuncDef(
        name="two_sum",
        params=(Param("nums"), Param("target")),
        body=(
            Assign((Name("n"),), Call(Name("len"), (Name("nums"),)), line=1),
            outer,
            Return(Const("none"), line=6),
        ),
        line=0,
    )
    return Module(functions=(func,), toplevel=(), main_guard=())


def test_render_module_prints_a_canonical_program() -> None:
    text = render_module(_two_sum())
    assert text.splitlines() == [
        "def two_sum(nums, target):",
        "    n = len(nums)",
        "    for i in range(n):",
        "        for j in range(i + 1, n):",
        "            if nums[i] + nums[j] == target:",
                "                return i",
        "    return None",
    ]


def test_walk_visits_every_node_once_in_preorder() -> None:
    kinds = [type(node).__name__ for node in walk(_two_sum().functions[0])]
    assert kinds[0] == "FuncDef"
    assert kinds.count("Loop") == 2
    assert kinds.count("Return") == 2
    assert "Subscript" in kinds and "Compare" in kinds


def test_children_of_a_leaf_is_empty_and_of_a_loop_includes_header_and_body() -> None:
    assert list(children(Name("x"))) == []
    loop = _two_sum().functions[0].body[1]
    kinds = {type(c).__name__ for c in children(loop)}
    assert {"Name", "Call", "Loop"} <= kinds


def test_typeref_renders_generics_and_array_dimensions() -> None:
    assert TypeRef("int", dims=2).render() == "int[][]"
    vec = TypeRef("vector", (TypeRef("vector", (TypeRef("int"),)),))
    assert vec.render() == "vector<vector<int>>"
    assert TypeRef("map", (TypeRef("int"), TypeRef("int"))).render() == "map<int, int>"


def test_nodes_are_hashable_and_comparable_by_value() -> None:
    a = BinOp("+", Name("x"), Num(1))
    b = BinOp("+", Name("x"), Num(1))
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}


def test_expression_statement_and_assignment_carry_their_line_numbers() -> None:
    stmt = ExprStmt(Call(Name("f"), ()), line=7)
    assert stmt.line == 7
    assert Assign((Name("x"),), Num(1), line=3).line == 3
