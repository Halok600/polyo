"""The complexity engine's language-neutral AST.

`core/ir.py` flattens a program into ~40 symbols and deliberately drops operators and operands so
one GNN can serve every language. Loop-bound analysis needs exactly what that drops (`i < n`,
`i *= 2`, `mid = (lo + hi) // 2`, `range(a, b, step)`), so the engine lowers each language into
this small tree instead (`analysis/lower_*.py`); the IR and the model artifacts stay untouched.

Conventions every lowering follows, so the analysis never branches on the source language:
  * `i++` / `i--` / `x += y` are `Assign` with an augmented `op`; comparison operators are
    `< <= > >= == != in not in is is not` (`===` is `==`); `!` is `not`.
  * Python `range(a, b, c)` stays a `Call`; every iteration form is a `Loop` of kind
    `for_each` (Python, JS for-of/in, Java/C++ range-for, Go range), `for_c` (init; test; update),
    `while` or `do_while`. `while True` and Go's bare `for {}` have `test=None`.
  * A member or module call is `Call(Attribute(obj, name), args)` -- `a.sort()`, `heapq.heappush`,
    `Arrays.sort`, `std::sort` and `it->second` all use `Attribute`.
  * Allocation is `New(type, args, dims)` (`new int[n][m]`, `vector<int> v(n)`, `make([]int, n)`);
    Python's `[0] * n` stays a `BinOp`.
  * Anything a lowering cannot express is `Unknown(text)` (an expression) -- never silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields


class LoweringError(ValueError):
    """The source could not be lowered (a syntax error, or an unsupported language)."""


class Expr:
    __slots__ = ()


class Stmt:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TypeRef:
    """A declared or constructed type: `vector<vector<int>>`, `int[][]`, `map[int]int`."""

    name: str
    args: tuple[TypeRef, ...] = ()
    dims: int = 0
    sizes: tuple[Expr | None, ...] = ()  # declared array sizes: C `int a[26]`, Go `[26]int`

    def render(self) -> str:
        text = self.name
        if self.args:
            text += "<" + ", ".join(a.render() for a in self.args) + ">"
        return text + "[]" * self.dims


# --------------------------------------------------------------------------------------------
# expressions
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Name(Expr):
    id: str


@dataclass(frozen=True, slots=True)
class Num(Expr):
    value: int | float | None


@dataclass(frozen=True, slots=True)
class Str(Expr):
    value: str


@dataclass(frozen=True, slots=True)
class Const(Expr):
    kind: str  # "true" | "false" | "none"


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    op: str  # + - * / // % ** << >> & | ^
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class UnOp(Expr):
    op: str  # - + not ~
    operand: Expr


@dataclass(frozen=True, slots=True)
class Compare(Expr):
    op: str  # < <= > >= == != in "not in" is "is not"
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class BoolOp(Expr):
    op: str  # and | or
    values: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class IfExp(Expr):
    test: Expr
    body: Expr
    orelse: Expr


@dataclass(frozen=True, slots=True)
class Call(Expr):
    func: Expr  # Name for a plain call, Attribute for a member / module / scoped call
    args: tuple[Expr, ...] = ()
    kwargs: tuple[tuple[str, Expr], ...] = ()


@dataclass(frozen=True, slots=True)
class Attribute(Expr):
    obj: Expr
    attr: str


@dataclass(frozen=True, slots=True)
class Subscript(Expr):
    obj: Expr
    index: Expr


@dataclass(frozen=True, slots=True)
class Slice(Expr):
    obj: Expr
    lower: Expr | None = None
    upper: Expr | None = None
    step: Expr | None = None


@dataclass(frozen=True, slots=True)
class ListLit(Expr):
    elts: tuple[Expr, ...] = ()
    kind: str = "list"  # list | tuple | set


@dataclass(frozen=True, slots=True)
class DictLit(Expr):
    items: tuple[tuple[Expr, Expr], ...] = ()


@dataclass(frozen=True, slots=True)
class CompFor:
    target: Expr
    iter: Expr
    conds: tuple[Expr, ...] = ()


@dataclass(frozen=True, slots=True)
class Comp(Expr):
    kind: str  # list | set | dict | gen
    elt: Expr
    generators: tuple[CompFor, ...]
    value: Expr | None = None  # the value of a dict comprehension


@dataclass(frozen=True, slots=True)
class Lambda(Expr):
    params: tuple[str, ...]
    body: Expr | tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class New(Expr):
    type: TypeRef
    args: tuple[Expr, ...] = ()
    dims: tuple[Expr, ...] = ()


@dataclass(frozen=True, slots=True)
class Star(Expr):
    value: Expr


@dataclass(frozen=True, slots=True)
class Unknown(Expr):
    text: str = ""


# --------------------------------------------------------------------------------------------
# statements
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Assign(Stmt):
    targets: tuple[Expr, ...]
    value: Expr | None
    op: str = "="  # "=" or an augmented operator such as "+=" "*=" "//=" "<<="
    decl: TypeRef | None = None  # set when this statement DECLARES the target with a type
    line: int = 0


@dataclass(frozen=True, slots=True)
class ExprStmt(Stmt):
    expr: Expr
    line: int = 0


@dataclass(frozen=True, slots=True)
class Return(Stmt):
    value: Expr | None = None
    line: int = 0


@dataclass(frozen=True, slots=True)
class Break(Stmt):
    line: int = 0


@dataclass(frozen=True, slots=True)
class Continue(Stmt):
    line: int = 0


@dataclass(frozen=True, slots=True)
class If(Stmt):
    test: Expr
    body: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...] = ()
    line: int = 0


@dataclass(frozen=True, slots=True)
class Loop(Stmt):
    kind: str  # for_each | for_c | while | do_while
    body: tuple[Stmt, ...]
    target: Expr | None = None  # for_each: the loop variable(s)
    iter: Expr | None = None  # for_each: what is iterated
    init: tuple[Stmt, ...] = ()  # for_c
    test: Expr | None = None  # for_c / while / do_while; None means "forever"
    update: tuple[Stmt, ...] = ()  # for_c
    line: int = 0


@dataclass(frozen=True, slots=True)
class Switch(Stmt):
    subject: Expr
    arms: tuple[tuple[Stmt, ...], ...]
    line: int = 0


@dataclass(frozen=True, slots=True)
class Try(Stmt):
    body: tuple[Stmt, ...]
    handlers: tuple[tuple[Stmt, ...], ...] = ()
    final: tuple[Stmt, ...] = ()
    line: int = 0


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    type: TypeRef | None = None
    default: Expr | None = None


@dataclass(frozen=True, slots=True)
class FuncDef(Stmt):
    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]
    cls: str | None = None  # the enclosing class, for a method
    decorators: tuple[str, ...] = ()
    returns: TypeRef | None = None
    line: int = 0


@dataclass(frozen=True, slots=True)
class Module:
    functions: tuple[FuncDef, ...]  # every function and method, flattened, in source order
    toplevel: tuple[Stmt, ...] = ()  # module-level code (a "program": reads input, loops, prints)
    main_guard: tuple[Stmt, ...] = ()  # the body of `if __name__ == "__main__":`


# --------------------------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------------------------
_NODE_TYPES = (Expr, Stmt, CompFor)


def _flatten(value: object) -> Iterator[object]:
    if isinstance(value, _NODE_TYPES):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _flatten(item)


def children(node: object) -> Iterator[object]:
    """The direct child nodes (expressions, statements, comprehension clauses) of `node`."""
    if not isinstance(node, _NODE_TYPES):
        return
    for f in fields(node):  # type: ignore[arg-type]
        yield from _flatten(getattr(node, f.name))


def walk(node: object) -> Iterator[object]:
    """`node` and every descendant, in preorder."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(list(children(current))))


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------
_BINARY_PRECEDENCE = {
    "or": 1, "and": 2, "|": 5, "^": 6, "&": 7, "<<": 8, ">>": 8,
    "+": 9, "-": 9, "*": 10, "/": 10, "//": 10, "%": 10, "**": 12,
}  # fmt: skip
_COMPARE_PRECEDENCE = 4
_NOT_PRECEDENCE = 3
_UNARY_PRECEDENCE = 11
_ATOM = 13


def _precedence(expr: Expr) -> int:
    if isinstance(expr, BinOp):
        return _BINARY_PRECEDENCE[expr.op]
    if isinstance(expr, BoolOp):
        return _BINARY_PRECEDENCE[expr.op]
    if isinstance(expr, Compare):
        return _COMPARE_PRECEDENCE
    if isinstance(expr, UnOp):
        return _NOT_PRECEDENCE if expr.op == "not" else _UNARY_PRECEDENCE
    if isinstance(expr, IfExp | Lambda):
        return 0
    return _ATOM


def _paren(expr: Expr, minimum: int) -> str:
    text = render_expr(expr)
    return f"({text})" if _precedence(expr) < minimum else text


def _num(value: int | float | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return repr(value)
    return repr(value)


def render_expr(expr: Expr) -> str:  # noqa: C901 -- one branch per node type is the clearest form
    if isinstance(expr, Name):
        return expr.id
    if isinstance(expr, Num):
        return _num(expr.value)
    if isinstance(expr, Str):
        return '"' + expr.value.replace('"', '\\"') + '"'
    if isinstance(expr, Const):
        return {"true": "True", "false": "False", "none": "None"}.get(expr.kind, expr.kind)
    if isinstance(expr, BinOp):
        prec = _BINARY_PRECEDENCE[expr.op]
        left = _paren(expr.left, prec)
        right = _paren(expr.right, prec + 1)
        return f"{left} {expr.op} {right}"
    if isinstance(expr, UnOp):
        sep = " " if expr.op == "not" else ""
        return f"{expr.op}{sep}{_paren(expr.operand, _precedence(expr))}"
    if isinstance(expr, Compare):
        left = _paren(expr.left, _COMPARE_PRECEDENCE + 1)
        right = _paren(expr.right, _COMPARE_PRECEDENCE + 1)
        return f"{left} {expr.op} {right}"
    if isinstance(expr, BoolOp):
        prec = _BINARY_PRECEDENCE[expr.op]
        return f" {expr.op} ".join(_paren(v, prec + 1) for v in expr.values)
    if isinstance(expr, IfExp):
        return (
            f"{render_expr(expr.body)} if {render_expr(expr.test)} else {render_expr(expr.orelse)}"
        )
    if isinstance(expr, Call):
        parts = [render_expr(a) for a in expr.args] + [
            f"{k}={render_expr(v)}" for k, v in expr.kwargs
        ]
        return f"{_paren(expr.func, _ATOM)}({', '.join(parts)})"
    if isinstance(expr, Attribute):
        return f"{_paren(expr.obj, _ATOM)}.{expr.attr}"
    if isinstance(expr, Subscript):
        return f"{_paren(expr.obj, _ATOM)}[{render_expr(expr.index)}]"
    if isinstance(expr, Slice):
        lower = render_expr(expr.lower) if expr.lower is not None else ""
        upper = render_expr(expr.upper) if expr.upper is not None else ""
        text = f"{lower}:{upper}"
        if expr.step is not None:
            text += f":{render_expr(expr.step)}"
        return f"{_paren(expr.obj, _ATOM)}[{text}]"
    if isinstance(expr, ListLit):
        inner = ", ".join(render_expr(e) for e in expr.elts)
        if expr.kind == "tuple":
            return f"({inner},)" if len(expr.elts) == 1 else f"({inner})"
        if expr.kind == "set":
            return "{" + inner + "}" if expr.elts else "set()"
        return f"[{inner}]"
    if isinstance(expr, DictLit):
        return "{" + ", ".join(f"{render_expr(k)}: {render_expr(v)}" for k, v in expr.items) + "}"
    if isinstance(expr, Comp):
        clauses = "".join(
            f" for {render_expr(g.target)} in {render_expr(g.iter)}"
            + "".join(f" if {render_expr(c)}" for c in g.conds)
            for g in expr.generators
        )
        elt = render_expr(expr.elt)
        if expr.kind == "dict" and expr.value is not None:
            elt = f"{elt}: {render_expr(expr.value)}"
        open_close = {"list": "[]", "set": "{}", "dict": "{}", "gen": "()"}[expr.kind]
        return f"{open_close[0]}{elt}{clauses}{open_close[1]}"
    if isinstance(expr, Lambda):
        params = ", ".join(expr.params)
        if isinstance(expr.body, tuple):
            return f"lambda {params}: <block>"
        return f"lambda {params}: {render_expr(expr.body)}"
    if isinstance(expr, New):
        if expr.dims:
            base = TypeRef(expr.type.name, expr.type.args).render()
            return f"new {base}" + "".join(f"[{render_expr(d)}]" for d in expr.dims)
        return f"new {expr.type.render()}({', '.join(render_expr(a) for a in expr.args)})"
    if isinstance(expr, Star):
        return f"*{render_expr(expr.value)}"
    if isinstance(expr, Unknown):
        return f"<unknown: {expr.text}>" if expr.text else "<unknown>"
    raise TypeError(f"cannot render {type(expr).__name__}")


def _render_param(param: Param) -> str:
    text = param.name
    if param.type is not None:
        text += f": {param.type.render()}"
    if param.default is not None:
        text += f" = {render_expr(param.default)}"
    return text


def _render_stmts(stmts: tuple[Stmt, ...], depth: int, out: list[str]) -> None:
    for stmt in stmts:
        _render_stmt(stmt, depth, out)


def _simple_stmt(stmt: Stmt) -> str:
    """One-line rendering for statements that appear in a `for (init; test; update)` header."""
    lines: list[str] = []
    _render_stmt(stmt, 0, lines)
    return lines[0] if lines else ""


def _render_stmt(stmt: Stmt, depth: int, out: list[str]) -> None:  # noqa: C901
    pad = "    " * depth
    if isinstance(stmt, Assign):
        targets = " = ".join(render_expr(t) for t in stmt.targets)
        annotation = f": {stmt.decl.render()}" if stmt.decl is not None else ""
        if stmt.value is None:
            out.append(f"{pad}{targets}{annotation}")
        else:
            out.append(f"{pad}{targets}{annotation} {stmt.op} {render_expr(stmt.value)}")
    elif isinstance(stmt, ExprStmt):
        out.append(pad + render_expr(stmt.expr))
    elif isinstance(stmt, Return):
        out.append(pad + ("return" if stmt.value is None else f"return {render_expr(stmt.value)}"))
    elif isinstance(stmt, Break):
        out.append(pad + "break")
    elif isinstance(stmt, Continue):
        out.append(pad + "continue")
    elif isinstance(stmt, If):
        out.append(f"{pad}if {render_expr(stmt.test)}:")
        _render_stmts(stmt.body, depth + 1, out)
        orelse = stmt.orelse
        while len(orelse) == 1 and isinstance(orelse[0], If):
            nested = orelse[0]
            out.append(f"{pad}elif {render_expr(nested.test)}:")
            _render_stmts(nested.body, depth + 1, out)
            orelse = nested.orelse
        if orelse:
            out.append(f"{pad}else:")
            _render_stmts(orelse, depth + 1, out)
    elif isinstance(stmt, Loop):
        if stmt.kind == "for_each":
            assert stmt.target is not None and stmt.iter is not None
            out.append(f"{pad}for {render_expr(stmt.target)} in {render_expr(stmt.iter)}:")
        elif stmt.kind == "for_c":
            init = ", ".join(_simple_stmt(s) for s in stmt.init)
            test = render_expr(stmt.test) if stmt.test is not None else ""
            update = ", ".join(_simple_stmt(s) for s in stmt.update)
            out.append(f"{pad}for {init}; {test}; {update}:")
        elif stmt.kind == "do_while":
            out.append(f"{pad}do:")
        else:
            test = render_expr(stmt.test) if stmt.test is not None else "True"
            out.append(f"{pad}while {test}:")
        _render_stmts(stmt.body, depth + 1, out)
        if stmt.kind == "do_while":
            test = render_expr(stmt.test) if stmt.test is not None else "True"
            out.append(f"{pad}while {test}")
    elif isinstance(stmt, Switch):
        out.append(f"{pad}switch {render_expr(stmt.subject)}:")
        for arm in stmt.arms:
            out.append(f"{pad}    case:")
            _render_stmts(arm, depth + 2, out)
    elif isinstance(stmt, Try):
        out.append(f"{pad}try:")
        _render_stmts(stmt.body, depth + 1, out)
        for handler in stmt.handlers:
            out.append(f"{pad}except:")
            _render_stmts(handler, depth + 1, out)
        if stmt.final:
            out.append(f"{pad}finally:")
            _render_stmts(stmt.final, depth + 1, out)
    elif isinstance(stmt, FuncDef):
        for decorator in stmt.decorators:
            out.append(f"{pad}@{decorator}")
        name = f"{stmt.cls}.{stmt.name}" if stmt.cls else stmt.name
        out.append(f"{pad}def {name}({', '.join(_render_param(p) for p in stmt.params)}):")
        _render_stmts(stmt.body, depth + 1, out)
        if not stmt.body:
            out.append(f"{pad}    pass")
    else:
        raise TypeError(f"cannot render {type(stmt).__name__}")


def render_module(module: Module) -> str:
    out: list[str] = []
    for func in module.functions:
        _render_stmt(func, 0, out)
    _render_stmts(module.toplevel, 0, out)
    if module.main_guard:
        out.append('if __name__ == "__main__":')
        _render_stmts(module.main_guard, 1, out)
    return "\n".join(out)
