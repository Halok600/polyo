"""Go -> the engine's AST (`analysis/nodes.py`), via tree-sitter.

Go is shaped unlike the C family: blocks hold a `statement_list`, assignments and returns take
`expression_list`s, there is one `for` keyword for every loop form, and slices, arrays and maps
are built-in types (`make([]int, n)`, `[26]int`, `map[int]int{}`). Slicing (`a[i:j]`) is an O(1)
view in Go, not a copy; that distinction lives in the stdlib cost table, not here.
"""

from __future__ import annotations

from tree_sitter import Node

from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    BoolOp,
    Break,
    Call,
    Compare,
    Const,
    Continue,
    DictLit,
    Expr,
    ExprStmt,
    FuncDef,
    If,
    Lambda,
    ListLit,
    Loop,
    Module,
    Name,
    New,
    Num,
    Param,
    Return,
    Slice,
    Star,
    Stmt,
    Str,
    Subscript,
    Switch,
    TypeRef,
    Unknown,
    UnOp,
)
from analysis.tsbase import TSBase
from parsing.parse import parse_source

_COMPARE_OPS = {"<", "<=", ">", ">=", "==", "!="}
_BINARY_OPS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"}
_RUNE_ESCAPES = {"0": 0, "n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34}


def _number(text: str) -> int | float | None:
    t = text.replace("_", "").lower()
    try:
        if t.startswith(("0x", "0b", "0o")):
            return int(t, 0)
        if t.isdigit():
            return int(t)
        return float(t)
    except ValueError:
        return None


class _Go(TSBase):
    # ------------------------------------------------------------------ types
    def type_ref(self, node: Node | None) -> TypeRef | None:  # noqa: C901
        if node is None:
            return None
        kind = node.type
        if kind == "slice_type":
            inner = self.type_ref(self.field(node, "element"))
            return TypeRef(inner.name, inner.args, inner.dims + 1, inner.sizes) if inner else None
        if kind == "array_type":
            inner = self.type_ref(self.field(node, "element"))
            length = self.field(node, "length")
            size = self.expr(length) if length is not None else None
            if inner is None:
                return None
            return TypeRef(inner.name, inner.args, inner.dims + 1, (*inner.sizes, size))
        if kind == "map_type":
            key = self.type_ref(self.field(node, "key"))
            value = self.type_ref(self.field(node, "value"))
            return TypeRef("map", tuple(t for t in (key, value) if t is not None))
        if kind in ("pointer_type", "parenthesized_type"):
            return self.type_ref(self.first_named(node))
        if kind == "qualified_type":
            package = self.field(node, "package")
            name = self.field(node, "name")
            return TypeRef(
                f"{self.text(package)}.{self.text(name)}" if package and name else self.text(node)
            )
        if kind == "generic_type":
            base = self.type_ref(self.field(node, "type") or self.first_named(node))
            args_node = next((c for c in node.named_children if c.type == "type_arguments"), None)
            args = (
                tuple(t for c in self.named(args_node) if (t := self.type_ref(c)) is not None)
                if args_node
                else ()
            )
            return TypeRef(base.name if base else self.text(node), args)
        if kind in ("channel_type", "function_type", "struct_type", "interface_type"):
            return TypeRef(kind.split("_")[0])
        return TypeRef(self.text(node).strip())

    # ------------------------------------------------------------------ module level
    def module(self, root: Node) -> Module:
        functions: list[FuncDef] = []
        toplevel: list[Stmt] = []
        for child in self.named(root):
            if child.type == "function_declaration":
                functions.append(self.function(child, None))
            elif child.type == "method_declaration":
                functions.append(self.function(child, self.receiver_type(child)))
            elif child.type in ("var_declaration", "const_declaration"):
                toplevel.extend(self.stmt(child))
        return Module(tuple(functions), tuple(toplevel), ())

    def receiver_type(self, node: Node) -> str | None:
        receiver = self.field(node, "receiver")
        if receiver is None:
            return None
        for param in self.named(receiver):
            typ = self.type_ref(self.field(param, "type"))
            if typ is not None:
                return typ.name
        return None

    def params(self, plist: Node | None) -> tuple[Param, ...]:
        if plist is None:
            return ()
        out: list[Param] = []
        for child in self.named(plist):
            if child.type not in ("parameter_declaration", "variadic_parameter_declaration"):
                continue
            typ = self.type_ref(self.field(child, "type"))
            if typ is not None and child.type == "variadic_parameter_declaration":
                typ = TypeRef(typ.name, typ.args, typ.dims + 1, typ.sizes)
            names = self.fields(child, "name")
            if not names:
                out.append(Param("_", typ))
            for name in names:
                out.append(Param(self.text(name), typ))
        return tuple(out)

    def function(self, node: Node, owner: str | None) -> FuncDef:
        name = self.field(node, "name")
        body = self.field(node, "body")
        result = self.field(node, "result")
        return FuncDef(
            name=self.text(name) if name is not None else "<anonymous>",
            params=self.params(self.field(node, "parameters")),
            body=tuple(self.stmts(body)) if body is not None else (),
            cls=owner,
            returns=self.type_ref(result)
            if result is not None and result.type != "parameter_list"
            else None,
            line=self.line(node),
        )

    # ------------------------------------------------------------------ statements
    def stmts(self, node: Node | None) -> list[Stmt]:
        if node is None:
            return []
        if node.type in ("block", "statement_list"):
            out: list[Stmt] = []
            for child in self.named(node):
                out.extend(
                    self.stmts(child)
                    if child.type in ("block", "statement_list")
                    else self.stmt(child)
                )
            return out
        return self.stmt(node)

    def stmt(self, node: Node) -> list[Stmt]:
        handler = getattr(self, f"s_{node.type}", None)
        return handler(node) if handler is not None else []

    def expression_list(self, node: Node | None) -> list[Expr]:
        if node is None:
            return []
        if node.type == "expression_list":
            return [self.expr(c) for c in self.named(node)]
        return [self.expr(node)]

    @staticmethod
    def pack(items: list[Expr]) -> Expr:
        return items[0] if len(items) == 1 else ListLit(tuple(items), "tuple")

    def assign_multi(self, left: Node | None, right: Node | None, op: str, line: int) -> list[Stmt]:
        def build() -> list[Stmt]:
            targets = self.expression_list(left)
            values = self.expression_list(right)
            if not targets:
                return []
            value = self.pack(values) if values else None
            return [Assign((self.pack(targets),), value, op=op, line=line)]

        return self.wrap(build)

    def s_short_var_declaration(self, node: Node) -> list[Stmt]:
        return self.assign_multi(
            self.field(node, "left"), self.field(node, "right"), "=", self.line(node)
        )

    def s_assignment_statement(self, node: Node) -> list[Stmt]:
        op_node = self.field(node, "operator")
        op = self.text(op_node) if op_node is not None else "="
        if op == "&^=":
            op = "&="
        return self.assign_multi(
            self.field(node, "left"), self.field(node, "right"), op, self.line(node)
        )

    def incdec(self, node: Node, op: str) -> list[Stmt]:
        inner = self.first_named(node)
        if inner is None:
            return []
        return self.wrap(lambda: [Assign((self.expr(inner),), Num(1), op=op, line=self.line(node))])

    def s_inc_statement(self, node: Node) -> list[Stmt]:
        return self.incdec(node, "+=")

    def s_dec_statement(self, node: Node) -> list[Stmt]:
        return self.incdec(node, "-=")

    def s_var_declaration(self, node: Node) -> list[Stmt]:
        out: list[Stmt] = []
        for spec in self.named(node):
            specs = self.named(spec) if spec.type == "var_spec_list" else [spec]
            for item in specs:
                if item.type != "var_spec":
                    continue
                typ = self.type_ref(self.field(item, "type"))
                names = self.fields(item, "name")
                value_node = self.field(item, "value")
                values = self.expression_list(value_node) if value_node is not None else []
                for index, name in enumerate(names):
                    value = (
                        values[index]
                        if len(values) == len(names)
                        else (values[0] if len(values) == 1 and len(names) == 1 else None)
                    )
                    out.append(
                        Assign((Name(self.text(name)),), value, decl=typ, line=self.line(item))
                    )
        return out

    s_const_declaration = s_var_declaration

    def s_expression_statement(self, node: Node) -> list[Stmt]:
        inner = self.first_named(node)
        if inner is None:
            return []
        return self.wrap(lambda: [ExprStmt(self.expr(inner), line=self.line(node))])

    def s_return_statement(self, node: Node) -> list[Stmt]:
        inner = self.first_named(node)
        if inner is None:
            return [Return(None, line=self.line(node))]
        value, pre, _ = self.with_effects(lambda: self.pack(self.expression_list(inner)))
        return [*pre, Return(value, line=self.line(node))]

    def s_break_statement(self, node: Node) -> list[Stmt]:
        return [Break(line=self.line(node))]

    def s_continue_statement(self, node: Node) -> list[Stmt]:
        return [Continue(line=self.line(node))]

    def s_go_statement(self, node: Node) -> list[Stmt]:
        inner = self.first_named(node)
        return [ExprStmt(self.expr(inner), line=self.line(node))] if inner is not None else []

    s_defer_statement = s_go_statement

    def s_labeled_statement(self, node: Node) -> list[Stmt]:
        kids = self.named(node)
        return self.stmts(kids[-1]) if kids else []

    def s_block(self, node: Node) -> list[Stmt]:
        return self.stmts(node)

    def s_if_statement(self, node: Node) -> list[Stmt]:
        pre: list[Stmt] = []
        initializer = self.field(node, "initializer")
        if initializer is not None:
            pre.extend(self.stmt(initializer))
        test, p, q = self.with_effects(lambda: self.expr(self.field(node, "condition")))
        consequence = self.stmts(self.field(node, "consequence"))
        alternative = self.field(node, "alternative")
        orelse = self.stmts(alternative) if alternative is not None else []
        return [*pre, *p, *q, If(test, tuple(consequence), tuple(orelse), line=self.line(node))]

    def s_for_statement(self, node: Node) -> list[Stmt]:  # noqa: C901
        line = self.line(node)
        body = tuple(self.stmts(self.field(node, "body")))
        clause = next((c for c in node.named_children if c.type == "for_clause"), None)
        range_clause = next((c for c in node.named_children if c.type == "range_clause"), None)
        if clause is not None:
            init_node = self.field(clause, "initializer")
            update_node = self.field(clause, "update")
            cond_node = self.field(clause, "condition")
            test = self.expr(cond_node) if cond_node is not None else None
            return [
                Loop(
                    kind="for_c",
                    init=tuple(self.stmt(init_node)) if init_node is not None else (),
                    test=test,
                    update=tuple(self.stmt(update_node)) if update_node is not None else (),
                    body=body,
                    line=line,
                )
            ]
        if range_clause is not None:
            left = self.field(range_clause, "left")
            right = self.field(range_clause, "right")
            target = self.pack(self.expression_list(left)) if left is not None else Name("_")
            return [
                Loop(
                    kind="for_each",
                    target=target,
                    iter=self.expr(right) if right is not None else Unknown(""),
                    body=body,
                    line=line,
                )
            ]
        header = [c for c in node.named_children if c.type != "block" and "comment" not in c.type]
        if header:
            return [Loop(kind="while", test=self.expr(header[0]), body=body, line=line)]
        return [Loop(kind="while", test=None, body=body, line=line)]

    def s_expression_switch_statement(self, node: Node) -> list[Stmt]:
        subject = self.field(node, "value")
        arms: list[tuple[Stmt, ...]] = []
        for case in self.named(node):
            if case.type not in (
                "expression_case",
                "default_case",
                "type_case",
                "communication_case",
            ):
                continue
            label = self.field(case, "value")
            statements: list[Stmt] = []
            for part in self.named(case):
                if label is not None and part.id == label.id:
                    continue
                statements.extend(
                    self.stmts(part) if part.type == "statement_list" else self.stmt(part)
                )
            arms.append(tuple(statements))
        test = self.expr(subject) if subject is not None else Const("true")
        return [Switch(test, tuple(arms), line=self.line(node))]

    s_type_switch_statement = s_expression_switch_statement
    s_select_statement = s_expression_switch_statement

    # ------------------------------------------------------------------ expressions
    def expr(self, node: Node | None) -> Expr:
        if node is None:
            return Unknown("")
        handler = getattr(self, f"x_{node.type}", None)
        return handler(node) if handler is not None else Unknown(node.type)

    def args(self, node: Node | None) -> tuple[Expr, ...]:
        return tuple(self.expr(a) for a in self.named(node)) if node is not None else ()

    def x_identifier(self, node: Node) -> Expr:
        return Name(self.text(node))

    x_field_identifier = x_identifier
    x_package_identifier = x_identifier
    x_type_identifier = x_identifier

    def x_true(self, node: Node) -> Expr:
        return Const("true")

    def x_false(self, node: Node) -> Expr:
        return Const("false")

    def x_nil(self, node: Node) -> Expr:
        return Const("none")

    def x_int_literal(self, node: Node) -> Expr:
        return Num(_number(self.text(node)))

    x_float_literal = x_int_literal

    def x_rune_literal(self, node: Node) -> Expr:
        body = self.text(node)[1:-1]
        if len(body) == 1:
            return Num(ord(body))
        if len(body) == 2 and body[0] == "\\" and body[1] in _RUNE_ESCAPES:
            return Num(_RUNE_ESCAPES[body[1]])
        return Str(body)

    def x_interpreted_string_literal(self, node: Node) -> Expr:
        text = self.text(node)
        return Str(text[1:-1] if len(text) >= 2 else text)

    x_raw_string_literal = x_interpreted_string_literal

    def x_parenthesized_expression(self, node: Node) -> Expr:
        inner = self.first_named(node)
        return self.expr(inner) if inner is not None else Unknown("")

    def x_binary_expression(self, node: Node) -> Expr:
        op_node = self.field(node, "operator")
        op = self.text(op_node) if op_node is not None else ""
        left = self.field(node, "left")
        right = self.field(node, "right")
        if op in ("&&", "||"):
            kind = "and" if op == "&&" else "or"
            values: list[Expr] = []
            for part in (left, right):
                value = self.expr(part)
                if isinstance(value, BoolOp) and value.op == kind:
                    values.extend(value.values)
                else:
                    values.append(value)
            return BoolOp(kind, tuple(values))
        if op in _COMPARE_OPS:
            return Compare(op, self.expr(left), self.expr(right))
        if op == "&^":
            op = "&"
        if op in _BINARY_OPS:
            return BinOp(op, self.expr(left), self.expr(right))
        return Unknown(op or "binary")

    def x_unary_expression(self, node: Node) -> Expr:
        op_node = self.field(node, "operator")
        op = self.text(op_node) if op_node is not None else ""
        operand = self.expr(self.field(node, "operand") or self.first_named(node))
        if op == "!":
            return UnOp("not", operand)
        if op == "-":
            if isinstance(operand, Num) and operand.value is not None:
                return Num(-operand.value)
            return UnOp("-", operand)
        if op == "^":
            return UnOp("~", operand)
        if op == "*":
            return Subscript(operand, Num(0))
        return operand  # +x, &x, <-ch

    def x_selector_expression(self, node: Node) -> Expr:
        operand = self.field(node, "operand")
        field = self.field(node, "field")
        return Attribute(self.expr(operand), self.text(field) if field is not None else "")

    def x_index_expression(self, node: Node) -> Expr:
        return Subscript(
            self.expr(self.field(node, "operand")), self.expr(self.field(node, "index"))
        )

    def x_slice_expression(self, node: Node) -> Expr:
        def part(name: str) -> Expr | None:
            child = self.field(node, name)
            return self.expr(child) if child is not None else None

        return Slice(self.expr(self.field(node, "operand")), part("start"), part("end"), None)

    def x_variadic_argument(self, node: Node) -> Expr:
        inner = self.first_named(node)
        return Star(self.expr(inner)) if inner is not None else Unknown("variadic")

    def x_type_assertion_expression(self, node: Node) -> Expr:
        return self.expr(self.field(node, "operand") or self.first_named(node))

    def x_type_conversion_expression(self, node: Node) -> Expr:
        return self.expr(self.field(node, "operand") or self.named(node)[-1])

    def x_call_expression(self, node: Node) -> Expr:
        function = self.field(node, "function")
        arguments = self.field(node, "arguments")
        kids = self.named(arguments) if arguments is not None else []
        if function is not None and function.type == "identifier" and kids:
            callee = self.text(function)
            type_like = (
                "slice_type",
                "map_type",
                "array_type",
                "channel_type",
                "type_identifier",
                "pointer_type",
            )
            if callee in ("make", "new") and kids[0].type in type_like:
                typ = self.type_ref(kids[0]) or TypeRef("?")
                rest = tuple(self.expr(a) for a in kids[1:])
                return New(typ, rest if callee == "make" else ())
        return Call(self.expr(function), tuple(self.expr(a) for a in kids))

    def literal_elements(self, body: Node | None) -> list[Node]:
        return self.named(body) if body is not None else []

    def element_value(self, element: Node) -> Expr:
        if element.type == "literal_element":
            inner = self.first_named(element)
            if inner is None:
                return Unknown("")
            return (
                self.x_literal_value(inner) if inner.type == "literal_value" else self.expr(inner)
            )
        if element.type == "keyed_element":
            kids = self.named(element)
            return self.element_value(kids[-1]) if kids else Unknown("")
        return self.expr(element)

    def x_literal_value(self, node: Node) -> Expr:
        return ListLit(tuple(self.element_value(e) for e in self.named(node)), "list")

    def x_composite_literal(self, node: Node) -> Expr:
        typ_node = self.field(node, "type")
        body = self.field(node, "body")
        typ = self.type_ref(typ_node)
        if typ is not None and typ.name == "map" and typ.dims == 0 and body is not None:
            items: list[tuple[Expr, Expr]] = []
            for element in self.named(body):
                kids = self.named(element)
                if element.type == "keyed_element" and len(kids) == 2:
                    items.append((self.element_value(kids[0]), self.element_value(kids[1])))
            return DictLit(tuple(items))
        elements = tuple(self.element_value(e) for e in self.literal_elements(body))
        kind = "list" if typ is not None and typ.dims > 0 else "tuple"
        return ListLit(elements, kind)

    def x_func_literal(self, node: Node) -> Expr:
        params = tuple(p.name for p in self.params(self.field(node, "parameters")))
        body = self.field(node, "body")
        return Lambda(params, tuple(self.stmts(body)) if body is not None else ())

    def x_slice_type(self, node: Node) -> Expr:
        return Name(self.text(node))

    x_map_type = x_slice_type
    x_array_type = x_slice_type


def lower_go(source: str) -> Module:
    tree = parse_source(source, "go")
    return _Go(source).module(tree.root_node)
