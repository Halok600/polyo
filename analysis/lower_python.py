"""Python source -> the engine's AST (`analysis/nodes.py`), via the stdlib `ast` module.

The stdlib parser is used here, not tree-sitter: it gives exact Python semantics (chained
comparisons, tuple targets, comprehensions, keyword arguments) without guessing at a grammar. A
program the stdlib cannot parse (Python 2, say) raises `LoweringError`; the caller decides what to
do (the hybrid predictor falls back to the model).
"""

from __future__ import annotations

import ast

from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    BoolOp,
    Break,
    Call,
    Comp,
    Compare,
    CompFor,
    Const,
    Continue,
    DictLit,
    Expr,
    ExprStmt,
    FuncDef,
    If,
    IfExp,
    Lambda,
    ListLit,
    Loop,
    LoweringError,
    Module,
    Name,
    Num,
    Param,
    Return,
    Slice,
    Star,
    Stmt,
    Str,
    Subscript,
    Switch,
    Try,
    TypeRef,
    Unknown,
    UnOp,
)

_BIN_OPS: dict[type[ast.operator], str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
    ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|", ast.BitXor: "^",
    ast.BitAnd: "&", ast.MatMult: "*",
}  # fmt: skip
_CMP_OPS: dict[type[ast.cmpop], str] = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}  # fmt: skip
_SELF_NAMES = frozenset({"self", "cls"})


def lower_python(source: str) -> Module:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise LoweringError(f"cannot parse Python source: {error}") from error
    return _Lowerer().module(tree)


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class _Lowerer:
    # ---------------------------------------------------------------- module level
    def module(self, tree: ast.Module) -> Module:
        functions: list[FuncDef] = []
        toplevel: list[Stmt] = []
        main_guard: list[Stmt] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append(self.func(node, cls=None))
            elif isinstance(node, ast.ClassDef):
                functions.extend(self.class_def(node))
            elif isinstance(node, ast.If) and _is_main_guard(node):
                main_guard.extend(self.stmts(node.body))
            elif isinstance(node, ast.Import | ast.ImportFrom):
                continue
            else:
                toplevel.extend(self.stmt(node))
        return Module(tuple(functions), tuple(toplevel), tuple(main_guard))

    def class_def(self, node: ast.ClassDef) -> list[FuncDef]:
        methods: list[FuncDef] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(self.func(item, cls=node.name))
            elif isinstance(item, ast.ClassDef):
                methods.extend(self.class_def(item))
        return methods

    def func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, cls: str | None) -> FuncDef:
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
        defaults += list(args.defaults)
        params: list[Param] = []
        for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
            if index == 0 and cls is not None and arg.arg in _SELF_NAMES:
                continue
            params.append(self.param(arg, default))
        if args.vararg is not None:
            params.append(self.param(args.vararg, None))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            params.append(self.param(arg, default))
        if args.kwarg is not None:
            params.append(self.param(args.kwarg, None))
        decorators = tuple(ast.unparse(d) for d in node.decorator_list)
        return FuncDef(
            name=node.name,
            params=tuple(params),
            body=self.stmts(node.body),
            cls=cls,
            decorators=decorators,
            returns=self.annotation(node.returns),
            line=node.lineno,
        )

    def param(self, arg: ast.arg, default: ast.expr | None) -> Param:
        return Param(
            name=arg.arg,
            type=self.annotation(arg.annotation),
            default=self.expr(default) if default is not None else None,
        )

    # ---------------------------------------------------------------- types
    def annotation(self, node: ast.expr | None) -> TypeRef | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return TypeRef(node.id)
        if isinstance(node, ast.Attribute):
            return TypeRef(node.attr)
        if isinstance(node, ast.Constant):
            if node.value is None:
                return TypeRef("None")
            if isinstance(node.value, str):
                try:
                    return self.annotation(ast.parse(node.value, mode="eval").body)
                except SyntaxError:
                    return TypeRef(node.value)
            return None
        if isinstance(node, ast.Subscript):
            base = self.annotation(node.value)
            if base is None:
                return None
            if base.name == "Optional":
                inner = node.slice
                return self.annotation(inner)
            inner = node.slice
            elements = inner.elts if isinstance(inner, ast.Tuple) else [inner]
            args = tuple(t for e in elements if (t := self.annotation(e)) is not None)
            return TypeRef(base.name, args)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.annotation(node.left)
        return None

    # ---------------------------------------------------------------- statements
    def stmts(self, body: list[ast.stmt]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for node in body:
            out.extend(self.stmt(node))
        return tuple(out)

    def stmt(self, node: ast.stmt) -> list[Stmt]:  # noqa: C901 -- one branch per statement kind
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Assign):
            targets = tuple(self.expr(t) for t in node.targets)
            return [Assign(targets, self.expr(node.value), line=line)]
        if isinstance(node, ast.AnnAssign):
            value = self.expr(node.value) if node.value is not None else None
            decl = self.annotation(node.annotation)
            return [Assign((self.expr(node.target),), value, decl=decl, line=line)]
        if isinstance(node, ast.AugAssign):
            op = _BIN_OPS[type(node.op)] + "="
            return [Assign((self.expr(node.target),), self.expr(node.value), op=op, line=line)]
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                return []  # a docstring or a bare literal
            return [ExprStmt(self.expr(node.value), line=line)]
        if isinstance(node, ast.Return):
            return [Return(self.expr(node.value) if node.value is not None else None, line=line)]
        if isinstance(node, ast.Break):
            return [Break(line=line)]
        if isinstance(node, ast.Continue):
            return [Continue(line=line)]
        if isinstance(node, ast.Raise):
            return [Return(None, line=line)]
        if isinstance(node, ast.Assert):
            return [ExprStmt(self.expr(node.test), line=line)]
        if isinstance(node, ast.If):
            return [
                If(self.expr(node.test), self.stmts(node.body), self.stmts(node.orelse), line=line)
            ]
        if isinstance(node, ast.For | ast.AsyncFor):
            loop = Loop(
                kind="for_each",
                target=self.expr(node.target),
                iter=self.expr(node.iter),
                body=self.stmts(node.body),
                line=line,
            )
            return [loop, *self.stmts(node.orelse)]
        if isinstance(node, ast.While):
            always = isinstance(node.test, ast.Constant) and bool(node.test.value)
            loop = Loop(
                kind="while",
                test=None if always else self.expr(node.test),
                body=self.stmts(node.body),
                line=line,
            )
            return [loop, *self.stmts(node.orelse)]
        if isinstance(node, ast.Try | ast.TryStar):
            return [
                Try(
                    body=self.stmts(node.body) + self.stmts(node.orelse),
                    handlers=tuple(self.stmts(h.body) for h in node.handlers),
                    final=self.stmts(node.finalbody),
                    line=line,
                )
            ]
        if isinstance(node, ast.With | ast.AsyncWith):
            out: list[Stmt] = []
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    out.append(
                        Assign(
                            (self.expr(item.optional_vars),),
                            self.expr(item.context_expr),
                            line=line,
                        )
                    )
            out.extend(self.stmts(node.body))
            return out
        if isinstance(node, ast.Match):
            arms = tuple(self.stmts(case.body) for case in node.cases)
            return [Switch(self.expr(node.subject), arms, line=line)]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return [self.func(node, cls=None)]
        return []  # imports, pass, global/nonlocal, del, nested classes

    # ---------------------------------------------------------------- expressions
    def expr(self, node: ast.expr | None) -> Expr:  # noqa: C901 -- one branch per expression kind
        if node is None:
            return Unknown("")
        if isinstance(node, ast.Name):
            return Name(node.id)
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None:
                return Const("none")
            if isinstance(value, bool):
                return Const("true" if value else "false")
            if isinstance(value, int | float):
                return Num(value)
            if isinstance(value, str):
                return Str(value)
            if isinstance(value, bytes):
                return Str(value.decode("latin-1"))
            return Unknown("constant")
        if isinstance(node, ast.JoinedStr):
            return Str("<f-string>")
        if isinstance(node, ast.BinOp):
            return BinOp(_BIN_OPS[type(node.op)], self.expr(node.left), self.expr(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = self.expr(node.operand)
            if isinstance(node.op, ast.USub):
                if isinstance(operand, Num) and operand.value is not None:
                    return Num(-operand.value)
                return UnOp("-", operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return UnOp("not", operand)
            return UnOp("~", operand)
        if isinstance(node, ast.BoolOp):
            op = "and" if isinstance(node.op, ast.And) else "or"
            return BoolOp(op, tuple(self.expr(v) for v in node.values))
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.IfExp):
            return IfExp(self.expr(node.test), self.expr(node.body), self.expr(node.orelse))
        if isinstance(node, ast.Call):
            args = tuple(
                Star(self.expr(a.value)) if isinstance(a, ast.Starred) else self.expr(a)
                for a in node.args
            )
            kwargs = tuple((k.arg or "**", self.expr(k.value)) for k in node.keywords)
            return Call(self.expr(node.func), args, kwargs)
        if isinstance(node, ast.Attribute):
            return Attribute(self.expr(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.List):
            return ListLit(tuple(self.expr(e) for e in node.elts), "list")
        if isinstance(node, ast.Tuple):
            return ListLit(tuple(self.expr(e) for e in node.elts), "tuple")
        if isinstance(node, ast.Set):
            return ListLit(tuple(self.expr(e) for e in node.elts), "set")
        if isinstance(node, ast.Dict):
            items = tuple(
                (self.expr(k), self.expr(v))
                for k, v in zip(node.keys, node.values, strict=True)
                if k is not None
            )
            return DictLit(items)
        if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp):
            kind = {ast.ListComp: "list", ast.SetComp: "set", ast.GeneratorExp: "gen"}[type(node)]
            return Comp(kind, self.expr(node.elt), self.generators(node.generators))
        if isinstance(node, ast.DictComp):
            return Comp(
                "dict", self.expr(node.key), self.generators(node.generators), self.expr(node.value)
            )
        if isinstance(node, ast.Lambda):
            params = tuple(a.arg for a in [*node.args.posonlyargs, *node.args.args])
            return Lambda(params, self.expr(node.body))
        if isinstance(node, ast.Starred):
            return Star(self.expr(node.value))
        if isinstance(node, ast.Await):
            return self.expr(node.value)
        if isinstance(node, ast.Yield | ast.YieldFrom):
            return self.expr(node.value)
        if isinstance(node, ast.NamedExpr):
            return Unknown("walrus")
        return Unknown(type(node).__name__)

    def compare(self, node: ast.Compare) -> Expr:
        left = self.expr(node.left)
        pieces: list[Expr] = []
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.expr(comparator)
            pieces.append(Compare(_CMP_OPS[type(op)], left, right))
            left = right
        return pieces[0] if len(pieces) == 1 else BoolOp("and", tuple(pieces))

    def subscript(self, node: ast.Subscript) -> Expr:
        obj = self.expr(node.value)
        index = node.slice
        if isinstance(index, ast.Slice):
            return Slice(
                obj,
                self.expr(index.lower) if index.lower is not None else None,
                self.expr(index.upper) if index.upper is not None else None,
                self.expr(index.step) if index.step is not None else None,
            )
        return Subscript(obj, self.expr(index))

    def generators(self, generators: list[ast.comprehension]) -> tuple[CompFor, ...]:
        return tuple(
            CompFor(
                target=self.expr(g.target),
                iter=self.expr(g.iter),
                conds=tuple(self.expr(c) for c in g.ifs),
            )
            for g in generators
        )
