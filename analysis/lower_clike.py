"""C, C++, Java and JavaScript -> the engine's AST (`analysis/nodes.py`), via tree-sitter.

These four grammars share most of their node vocabulary (`binary_expression`, `for_statement`,
`if_statement`, ...) and differ in field names and declaration shapes, so one lowerer with small
per-language branches is clearer than four near-copies. Go is different enough to live in
`lower_go.py`.

Two things here are not mechanical translation:

  * SIDE EFFECTS inside expressions (`tmp[k++] = nums[i++]`, `while ((x = f()) != 0)`) are hoisted
    into explicit statements around the statement that contained them. Loop-bound analysis reads
    "how does the loop variable change", and an increment buried inside a subscript would be
    invisible to it.
  * DECLARATIONS become typed `Assign` nodes whatever their surface shape: C++ `vector<int> v(n)`,
    C `int a[26] = {0}`, Java `int[][] m = new int[r][c]`, JS `const seen = new Map()`.

Anything unsupported degrades to `Unknown` (an expression) or is skipped (a statement); the engine
reports reduced certainty for it instead of crashing.
"""

from __future__ import annotations

import re
from functools import partial

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
    IfExp,
    Lambda,
    ListLit,
    Loop,
    Module,
    Name,
    New,
    Num,
    Param,
    Return,
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
from analysis.roles import (
    DEQUE_TYPES,
    DICT_TYPES,
    HEAP_TYPES,
    LIST_TYPES,
    SET_TYPES,
    STACK_TYPES,
    TREEMAP_TYPES,
    TREESET_TYPES,
)
from analysis.tsbase import TSBase
from parsing.parse import parse_source

_BLOCK_TYPES = frozenset({"compound_statement", "block", "statement_block"})
# templates that can be written as a constructor call: `vector<int>(m, 0)`
_CONSTRUCTIBLE = (
    LIST_TYPES | DEQUE_TYPES | STACK_TYPES | HEAP_TYPES | DICT_TYPES | SET_TYPES
    | TREEMAP_TYPES | TREESET_TYPES
)  # fmt: skip
_PRIMITIVE_TYPES = frozenset(
    {"int", "long", "short", "char", "byte", "float", "double", "bool", "boolean", "unsigned",
     "signed", "size_t", "auto", "string", "String", "var", "void"}
)  # fmt: skip
_COMPARE_OPS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!=",
                "===": "==", "!==": "!="}  # fmt: skip
_BINARY_OPS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "**"}
_DECLARATION_TYPES = frozenset(
    {"declaration", "local_variable_declaration", "lexical_declaration", "variable_declaration"}
)
_FUNCTION_VALUE_TYPES = frozenset({"function_expression", "function", "arrow_function"})
_CHAR_ESCAPES = {"0": 0, "n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34, "a": 7, "b": 8}


def _constructs_object(typ: TypeRef) -> bool:
    """`T x(args)` calls a constructor unless T is a primitive (`int x(5)` is just `int x = 5`)."""
    return typ.dims == 0 and (typ.name not in _PRIMITIVE_TYPES or typ.name in ("string", "String"))


def _clean_number(text: str) -> int | float | None:
    t = text.replace("_", "").replace("'", "")
    lowered = t.lower()
    try:
        if lowered.startswith(("0x", "0b", "0o")):
            return int(lowered.rstrip("ul"), 0)
        stripped = lowered.rstrip("ulfd")
        if re.fullmatch(r"\d+", stripped):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return None


class _CLike(TSBase):
    def __init__(self, source: str, language: str) -> None:
        super().__init__(source)
        self.lang = language

    # ------------------------------------------------------------------ types
    def _type_ref(self, node: Node | None) -> TypeRef | None:  # noqa: C901
        if node is None:
            return None
        kind = node.type
        if kind in ("template_type", "generic_type", "template_function"):
            name_node = self.field(node, "name") or self.first_named(node)
            args_node = self.field(node, "arguments") or next(
                (
                    c
                    for c in node.named_children
                    if c.type in ("type_arguments", "template_argument_list")
                ),
                None,
            )
            name = self.text(name_node) if name_node is not None else self.text(node)
            if name_node is not None and name_node.type in (
                "scoped_type_identifier",
                "qualified_identifier",
            ):
                name = self.text(name_node).replace("::", ".").split(".")[-1]
            args: tuple[TypeRef, ...] = ()
            if args_node is not None:
                args = tuple(
                    t for c in self.named(args_node) if (t := self.type_ref(c)) is not None
                )
            return TypeRef(name, args)
        if kind == "type_descriptor":
            return self.type_ref(self.field(node, "type"))
        if kind in ("qualified_identifier", "scoped_type_identifier", "nested_type_identifier"):
            inner = self.field(node, "name")
            if inner is not None:
                return self.type_ref(inner)
            return TypeRef(self.text(node).replace("::", ".").split(".")[-1])
        if kind in ("struct_specifier", "class_specifier", "union_specifier", "enum_specifier"):
            inner = self.field(node, "name")
            return TypeRef(self.text(inner)) if inner is not None else TypeRef(kind.split("_")[0])
        if kind == "array_type":
            base = self.type_ref(self.field(node, "element"))
            dims_node = self.field(node, "dimensions")
            count = self.text(dims_node).count("[") if dims_node is not None else 1
            if base is None:
                return None
            return TypeRef(base.name, base.args, base.dims + count)
        if kind in ("placeholder_type_specifier", "decltype"):
            return TypeRef("auto")
        return TypeRef(self.text(node).strip())

    def unwrap_declarator(self, node: Node | None, base: TypeRef | None):  # noqa: ANN201, C901
        """(target expression, declared type) for a C/C++ declarator chain, or (None, None) for a
        function prototype."""
        dims = base.dims if base else 0
        sizes = list(base.sizes) if base else []
        is_ref = False
        pointers = 0
        current = node
        while current is not None:
            kind = current.type
            if kind in ("identifier", "field_identifier"):
                typ = (
                    TypeRef(base.name, base.args, dims, tuple(sizes), is_ref, pointers)
                    if base
                    else None
                )
                return Name(self.text(current)), typ
            if kind == "pointer_declarator":
                dims += 1
                pointers += 1
                current = self.field(current, "declarator") or self.first_named(current)
            elif kind in ("reference_declarator", "parenthesized_declarator"):
                is_ref = is_ref or kind == "reference_declarator"
                current = self.first_named(current)
            elif kind == "array_declarator":
                dims += 1
                size = self.field(current, "size")
                sizes.append(self.expr(size) if size is not None else None)
                current = self.field(current, "declarator")
            elif kind == "structured_binding_declarator":
                names = tuple(Name(self.text(c)) for c in self.named(current))
                typ = TypeRef(base.name, base.args, dims, tuple(sizes)) if base else None
                return ListLit(names, "tuple"), typ
            else:
                return None, None
        return None, None

    # ------------------------------------------------------------------ module level
    def module(self, root: Node) -> Module:
        functions: list[FuncDef] = []
        toplevel: list[Stmt] = []
        for child in self.named(root):
            self.top_level(child, functions, toplevel)
        return Module(tuple(functions), tuple(toplevel), ())

    def top_level(self, node: Node, functions: list[FuncDef], toplevel: list[Stmt]) -> None:  # noqa: C901
        kind = node.type
        if kind == "function_definition":
            func = self.function_cpp(node, None)
            if func is not None:
                functions.append(func)
        elif kind in ("class_specifier", "struct_specifier", "union_specifier"):
            self.class_cpp(node, functions)
        elif kind == "template_declaration":
            for child in self.named(node):
                if child.type in ("function_definition", "class_specifier", "struct_specifier"):
                    self.top_level(child, functions, toplevel)
        elif kind in ("namespace_definition", "linkage_specification"):
            body = self.field(node, "body")
            if body is not None:
                for child in self.named(body):
                    self.top_level(child, functions, toplevel)
        elif kind == "declaration":
            toplevel.extend(self.s_declaration(node, local=False))
        elif kind in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            self.class_java_or_js(node, functions)
        elif kind == "function_declaration":
            functions.append(self.function_js(node, None))
        elif kind == "export_statement":
            for child in self.named(node):
                self.top_level(child, functions, toplevel)
        elif kind in ("lexical_declaration", "variable_declaration"):
            self.js_declaration_top(node, functions, toplevel)
        elif kind == "alias_declaration":  # using Grid = vector<vector<int>>;
            name = self.field(node, "name")
            if name is not None:
                self.register_alias(self.text(name), self.type_ref(self.field(node, "type")))
        elif kind == "type_definition":  # typedef vector<int> vi;
            declarator = self.field(node, "declarator")
            if declarator is not None and declarator.type in ("type_identifier", "identifier"):
                self.register_alias(self.text(declarator), self.type_ref(self.field(node, "type")))
        elif kind in (
            "preproc_include", "preproc_def", "preproc_function_def", "preproc_call", "preproc_if",
            "preproc_ifdef", "using_declaration",
            "import_declaration", "package_declaration", "import_statement", "empty_statement",
            "static_assert_declaration", "namespace_alias_definition", "using_directive",
        ):  # fmt: skip
            return
        else:
            toplevel.extend(self.stmt(node))

    def js_declaration_top(
        self, node: Node, functions: list[FuncDef], toplevel: list[Stmt]
    ) -> None:
        for declarator in self.named(node):
            if declarator.type != "variable_declarator":
                continue
            name = self.field(declarator, "name")
            value = self.field(declarator, "value")
            if name is not None and value is not None and value.type in _FUNCTION_VALUE_TYPES:
                functions.append(self.function_js(value, None, name_override=self.text(name)))
            else:
                toplevel.extend(self.declarator_js(declarator, self.line(node)))

    # ------------------------------------------------------------------ classes
    def class_cpp(self, node: Node, functions: list[FuncDef]) -> None:
        name_node = self.field(node, "name")
        owner = self.text(name_node) if name_node is not None else None
        body = self.field(node, "body")
        if body is None:
            return
        for child in self.named(body):
            if child.type == "function_definition":
                func = self.function_cpp(child, owner)
                if func is not None:
                    functions.append(func)
            elif child.type in ("class_specifier", "struct_specifier"):
                self.class_cpp(child, functions)
            elif child.type == "template_declaration":
                for inner in self.named(child):
                    if inner.type == "function_definition":
                        func = self.function_cpp(inner, owner)
                        if func is not None:
                            functions.append(func)

    def class_java_or_js(self, node: Node, functions: list[FuncDef]) -> None:
        name_node = self.field(node, "name")
        owner = self.text(name_node) if name_node is not None else None
        body = self.field(node, "body")
        if body is None:
            return
        members = self.named(body)
        for member in list(members):
            if member.type == "enum_body_declarations":
                members.extend(self.named(member))
        for member in members:
            if member.type in ("method_declaration", "constructor_declaration"):
                functions.append(self.function_java(member, owner))
            elif member.type == "method_definition":
                functions.append(self.function_js(member, owner))
            elif member.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):
                self.class_java_or_js(member, functions)

    # ------------------------------------------------------------------ functions
    def function_cpp(self, node: Node, owner: str | None) -> FuncDef | None:
        declarator = self.field(node, "declarator")
        while declarator is not None and declarator.type in (
            "pointer_declarator", "reference_declarator", "parenthesized_declarator",
        ):  # fmt: skip
            declarator = self.field(declarator, "declarator") or self.first_named(declarator)
        if declarator is None or declarator.type != "function_declarator":
            return None
        name_node = self.field(declarator, "declarator")
        if name_node is None:
            return None
        name = self.text(name_node)
        if name_node.type == "qualified_identifier":
            scope = self.field(name_node, "scope")
            inner = self.field(name_node, "name")
            owner = self.text(scope) if scope is not None else owner
            name = self.text(inner) if inner is not None else name
        params_node = self.field(declarator, "parameters")
        params = self.params_cpp(params_node) if params_node is not None else ()
        body = self.field(node, "body")
        statements = self.stmts(body) if body is not None and body.type in _BLOCK_TYPES else []
        return FuncDef(
            name=name, params=params, body=tuple(statements), cls=owner,
            returns=self.type_ref(self.field(node, "type")), line=self.line(node),
        )  # fmt: skip

    def params_cpp(self, plist: Node) -> tuple[Param, ...]:
        out: list[Param] = []
        for child in self.named(plist):
            if child.type not in ("parameter_declaration", "optional_parameter_declaration"):
                continue
            base = self.type_ref(self.field(child, "type"))
            declarator = self.field(child, "declarator")
            name_expr, typ = (
                self.unwrap_declarator(declarator, base) if declarator else (None, base)
            )
            name = name_expr.id if isinstance(name_expr, Name) else "_"
            default = self.field(child, "default_value")
            out.append(Param(name, typ, self.expr(default) if default is not None else None))
        return tuple(out)

    def function_java(self, node: Node, owner: str | None) -> FuncDef:
        name_node = self.field(node, "name")
        params_node = self.field(node, "parameters")
        params: list[Param] = []
        if params_node is not None:
            for child in self.named(params_node):
                if child.type in ("formal_parameter", "spread_parameter"):
                    typ = self.type_ref(self.field(child, "type") or self.first_named(child))
                    param_name = self.field(child, "name")
                    if param_name is None:
                        declarator = next(
                            (c for c in child.named_children if c.type == "variable_declarator"),
                            None,
                        )
                        param_name = (
                            self.field(declarator, "name") if declarator is not None else None
                        )
                    dims_node = self.field(child, "dimensions")
                    if typ is not None and dims_node is not None:
                        typ = TypeRef(
                            typ.name, typ.args, typ.dims + self.text(dims_node).count("[")
                        )
                    params.append(
                        Param(self.text(param_name) if param_name is not None else "_", typ)
                    )
        body = self.field(node, "body")
        return FuncDef(
            name=self.text(name_node) if name_node is not None else "<init>",
            params=tuple(params),
            body=tuple(self.stmts(body)) if body is not None else (),
            cls=owner,
            returns=self.type_ref(self.field(node, "type")),
            line=self.line(node),
        )

    def params_js(self, params_node: Node | None) -> tuple[Param, ...]:
        if params_node is None:
            return ()
        if params_node.type == "identifier":
            return (Param(self.text(params_node)),)
        out: list[Param] = []
        for child in self.named(params_node):
            if child.type == "identifier":
                out.append(Param(self.text(child)))
            elif child.type == "assignment_pattern":
                left = self.field(child, "left")
                right = self.field(child, "right")
                out.append(
                    Param(
                        self.text(left) if left is not None else "_",
                        None,
                        self.expr(right) if right is not None else None,
                    )
                )
            elif child.type == "rest_pattern":
                inner = self.first_named(child)
                out.append(Param(self.text(inner) if inner is not None else "_"))
            else:
                out.append(Param("_pattern"))
        return tuple(out)

    def function_js(
        self, node: Node, owner: str | None, name_override: str | None = None
    ) -> FuncDef:
        name_node = self.field(node, "name")
        name = name_override or (self.text(name_node) if name_node is not None else "<anonymous>")
        params = self.params_js(self.field(node, "parameters", "parameter"))
        body = self.field(node, "body")
        if body is None:
            statements: tuple[Stmt, ...] = ()
        elif body.type in _BLOCK_TYPES:
            statements = tuple(self.stmts(body))
        else:  # an arrow function with an expression body returns that expression
            value, pre, post = self.with_effects(lambda: self.expr(body))
            statements = (*pre, Return(value, line=self.line(body)), *post)
        return FuncDef(name=name, params=params, body=statements, cls=owner, line=self.line(node))

    # ------------------------------------------------------------------ statements
    def stmts(self, node: Node | None) -> list[Stmt]:
        if node is None:
            return []
        if node.type in _BLOCK_TYPES:
            out: list[Stmt] = []
            for child in self.named(node):
                out.extend(self.stmt(child))
            return out
        return self.stmt(node)

    def stmt(self, node: Node) -> list[Stmt]:
        handler = getattr(self, f"s_{node.type}", None)
        if handler is None:
            return []
        return handler(node)

    def s_compound_statement(self, node: Node) -> list[Stmt]:
        return self.stmts(node)

    s_block = s_compound_statement
    s_statement_block = s_compound_statement

    def s_labeled_statement(self, node: Node) -> list[Stmt]:
        body = self.field(node, "body") or (self.named(node)[-1] if self.named(node) else None)
        return self.stmts(body) if body is not None and body.type != "statement_identifier" else []

    def s_expression_statement(self, node: Node) -> list[Stmt]:
        inner = self.first_named(node)
        return self.expr_stmt(inner, self.line(node)) if inner is not None else []

    def flatten_comma(self, node: Node) -> list[Node]:
        if node.type in ("comma_expression", "sequence_expression"):
            out: list[Node] = []
            for child in self.named(node):
                out.extend(self.flatten_comma(child))
            return out
        return [node]

    def expr_stmt(self, node: Node, line: int) -> list[Stmt]:
        kind = node.type
        if kind in ("comma_expression", "sequence_expression"):
            out: list[Stmt] = []
            for part in self.flatten_comma(node):
                out.extend(self.expr_stmt(part, line))
            return out
        if kind == "parenthesized_expression":
            inner = self.first_named(node)
            return self.expr_stmt(inner, line) if inner is not None else []
        if kind in ("assignment_expression", "augmented_assignment_expression"):
            return self.wrap(lambda: [self.assignment(node, line)])
        if kind == "update_expression":
            return self.wrap(lambda: [self.update_stmt(node, line)])
        return self.wrap(lambda: [ExprStmt(self.expr(node), line=line)])

    def operator_text(self, node: Node) -> str:
        op = self.field(node, "operator")
        if op is not None:
            return self.text(op)
        for child in node.children:
            if not child.is_named and child.type not in ("(", ")", ",", ";"):
                return child.type
        return ""

    def assignment(self, node: Node, line: int) -> Assign:
        left = self.field(node, "left")
        right = self.field(node, "right")
        op = self.operator_text(node) or "="
        if op == ">>>=":
            op = ">>="
        target = self.target(left) if left is not None else Unknown("")
        value = self.expr(right) if right is not None else Unknown("")
        return Assign(
            (target,), value, op=op if op in ("=",) or op.endswith("=") else "=", line=line
        )

    def target(self, node: Node) -> Expr:
        if node.type == "array_pattern":
            return ListLit(tuple(self.target(c) for c in self.named(node)), "tuple")
        if node.type == "object_pattern":
            return ListLit(tuple(Name(self.text(c)) for c in self.named(node)), "tuple")
        return self.expr(node)

    def update_stmt(self, node: Node, line: int) -> Assign:
        op = "+="
        for child in node.children:
            if not child.is_named and child.type in ("++", "--"):
                op = "+=" if child.type == "++" else "-="
        argument = self.field(node, "argument") or self.first_named(node)
        target = self.expr(argument) if argument is not None else Unknown("")
        return Assign((target,), Num(1), op=op, line=line)

    # -- declarations
    def s_declaration(self, node: Node, local: bool = True) -> list[Stmt]:
        base = self.type_ref(self.field(node, "type"))
        line = self.line(node)
        out: list[Stmt] = []
        for declarator in self.fields(node, "declarator"):
            out.extend(self.wrap(partial(self.declare_cpp, base, declarator, line, local)))
        return out

    def vexing_arguments(self, declarator: Node) -> tuple[Name, tuple[Expr, ...]] | None:
        """`vector<int> a(n);` and `vector<int> b(x, y);` read as FUNCTION declarators (the "most
        vexing parse"): a parameter list whose entries are bare type names. Inside a function body
        that is a constructor call, and the bare names are its arguments."""
        name = self.field(declarator, "declarator")
        params = self.field(declarator, "parameters")
        if name is None or name.type != "identifier" or params is None:
            return None
        args: list[Expr] = []
        for param in self.named(params):
            if param.type != "parameter_declaration" or self.field(param, "declarator") is not None:
                return None
            inner = self.named(param)
            if len(inner) != 1 or inner[0].type not in ("type_identifier", "identifier"):
                return None
            args.append(Name(self.text(inner[0])))
        return Name(self.text(name)), tuple(args)

    def declare_cpp(
        self, base: TypeRef | None, declarator: Node, line: int, local: bool = True
    ) -> list[Stmt]:
        if declarator.type == "function_declarator" and local and base is not None:
            found = self.vexing_arguments(declarator)
            if found is not None and _constructs_object(base):
                target, args = found
                return [Assign((target,), New(base, args), decl=base, line=line)]
        if declarator.type == "init_declarator":
            inner = self.field(declarator, "declarator")
            name, typ = self.unwrap_declarator(inner, base)
            if name is None:
                return []
            value = self.init_value(typ, self.field(declarator, "value"))
            return [Assign((name,), value, decl=typ, line=line)]
        name, typ = self.unwrap_declarator(declarator, base)
        if name is None:
            return []
        return [Assign((name,), None, decl=typ, line=line)]

    def init_value(self, typ: TypeRef | None, node: Node | None) -> Expr | None:
        if node is None:
            return None
        if node.type == "argument_list":
            args = tuple(self.expr(a) for a in self.named(node))
            if typ is not None and _constructs_object(typ):
                return New(typ, args)
            return args[0] if args else Num(0)
        return self.expr(node)

    def s_local_variable_declaration(self, node: Node) -> list[Stmt]:
        base = self.type_ref(self.field(node, "type"))
        line = self.line(node)
        out: list[Stmt] = []
        for declarator in self.fields(node, "declarator"):
            out.extend(self.wrap(partial(self.declare_java, base, declarator, line)))
        return out

    def declare_java(self, base: TypeRef | None, declarator: Node, line: int) -> list[Stmt]:
        """One Java declarator. Called through `wrap`, so side effects hoisted out of the
        initialiser (`int x = arr[i++]`) land around THIS statement and are not lost."""
        name_node = self.field(declarator, "name")
        if name_node is None:
            return []
        typ = base
        dims_node = self.field(declarator, "dimensions")
        if typ is not None and dims_node is not None:
            typ = TypeRef(typ.name, typ.args, typ.dims + self.text(dims_node).count("["))
        value_node = self.field(declarator, "value")
        value = self.expr(value_node) if value_node is not None else None
        return [Assign((Name(self.text(name_node)),), value, decl=typ, line=line)]

    def declarator_js(self, declarator: Node, line: int) -> list[Stmt]:
        name_node = self.field(declarator, "name")
        value_node = self.field(declarator, "value")
        if name_node is None:
            return []
        return self.wrap(
            lambda: [
                Assign(
                    (self.target(name_node),),
                    self.expr(value_node) if value_node is not None else None,
                    line=line,
                )
            ]
        )

    def s_lexical_declaration(self, node: Node) -> list[Stmt]:
        out: list[Stmt] = []
        for declarator in self.named(node):
            if declarator.type != "variable_declarator":
                continue
            value = self.field(declarator, "value")
            name = self.field(declarator, "name")
            if name is not None and value is not None and value.type in _FUNCTION_VALUE_TYPES:
                out.append(self.function_js(value, None, name_override=self.text(name)))
            else:
                out.extend(self.declarator_js(declarator, self.line(node)))
        return out

    s_variable_declaration = s_lexical_declaration

    # -- control flow
    def condition(self, node: Node | None):  # noqa: ANN201
        """(test expression, hoisted effects) from an `if` / `while` condition node."""
        if node is None:
            return Const("true"), []
        pre: list[Stmt] = []
        inner: Node | None
        if node.type == "condition_clause":
            for init in self.fields(node, "initializer"):
                pre.extend(self.stmt(init))
            inner = self.field(node, "value") or self.first_named(node)
        else:
            inner = node
        while inner is not None and inner.type == "parenthesized_expression":
            inner = self.first_named(inner)
        if inner is None:
            return Const("true"), pre
        target = inner
        value, p, q = self.with_effects(lambda: self.expr(target))
        return value, [*pre, *p, *q]

    @staticmethod
    def is_forever(test: Expr) -> bool:
        return (isinstance(test, Const) and test.kind == "true") or (
            isinstance(test, Num) and test.value not in (0, None)
        )

    def s_if_statement(self, node: Node) -> list[Stmt]:
        test, effects = self.condition(self.field(node, "condition"))
        consequence = self.stmts(self.field(node, "consequence"))
        alternative = self.field(node, "alternative")
        if alternative is not None and alternative.type == "else_clause":
            alternative = self.first_named(alternative)
        orelse = self.stmts(alternative) if alternative is not None else []
        return [*effects, If(test, tuple(consequence), tuple(orelse), line=self.line(node))]

    def s_while_statement(self, node: Node) -> list[Stmt]:
        test, effects = self.condition(self.field(node, "condition"))
        body = self.stmts(self.field(node, "body"))
        loop = Loop(
            kind="while",
            test=None if self.is_forever(test) else test,
            body=tuple([*body, *effects]),
            line=self.line(node),
        )
        return [*effects, loop]

    def s_do_statement(self, node: Node) -> list[Stmt]:
        test, effects = self.condition(self.field(node, "condition"))
        body = self.stmts(self.field(node, "body"))
        return [
            Loop(
                kind="do_while",
                test=None if self.is_forever(test) else test,
                body=tuple([*body, *effects]),
                line=self.line(node),
            )
        ]

    def s_for_statement(self, node: Node) -> list[Stmt]:
        line = self.line(node)
        init: list[Stmt] = []
        for init_node in self.fields(node, "initializer", "init"):
            if init_node.type in ("empty_statement", ";"):
                continue
            if init_node.type in _DECLARATION_TYPES:
                init.extend(self.stmt(init_node))
            else:
                init.extend(self.expr_stmt(init_node, line))
        cond_node = self.field(node, "condition")
        test: Expr | None = None
        effects: list[Stmt] = []
        if cond_node is not None and cond_node.type not in ("empty_statement", ";"):
            if cond_node.type == "expression_statement":
                cond_node = self.first_named(cond_node)
            if cond_node is not None:
                test, effects = self.condition(cond_node)
        update: list[Stmt] = []
        for update_node in self.fields(node, "update", "increment"):
            update.extend(self.expr_stmt(update_node, line))
        body = self.stmts(self.field(node, "body"))
        return [
            Loop(
                kind="for_c",
                init=tuple(init),
                test=test,
                update=tuple(update),
                body=tuple([*body, *effects]),
                line=line,
            )
        ]

    def s_for_range_loop(self, node: Node) -> list[Stmt]:
        base = self.type_ref(self.field(node, "type"))
        declarator = self.field(node, "declarator")
        target, _ = self.unwrap_declarator(declarator, base)
        iterable = self.field(node, "right")
        body = self.stmts(self.field(node, "body"))
        return [
            Loop(
                kind="for_each",
                target=target or Unknown("target"),
                iter=self.expr(iterable) if iterable is not None else Unknown(""),
                body=tuple(body),
                line=self.line(node),
            )
        ]

    def s_enhanced_for_statement(self, node: Node) -> list[Stmt]:
        name = self.field(node, "name")
        iterable = self.field(node, "value")
        body = self.stmts(self.field(node, "body"))
        return [
            Loop(
                kind="for_each",
                target=Name(self.text(name)) if name is not None else Unknown("target"),
                iter=self.expr(iterable) if iterable is not None else Unknown(""),
                body=tuple(body),
                line=self.line(node),
            )
        ]

    def s_for_in_statement(self, node: Node) -> list[Stmt]:
        left = self.field(node, "left")
        right = self.field(node, "right")
        body = self.stmts(self.field(node, "body"))
        return [
            Loop(
                kind="for_each",
                target=self.target(left) if left is not None else Unknown("target"),
                iter=self.expr(right) if right is not None else Unknown(""),
                body=tuple(body),
                line=self.line(node),
            )
        ]

    def s_return_statement(self, node: Node) -> list[Stmt]:
        inner = self.first_named(node)
        if inner is None:
            return [Return(None, line=self.line(node))]
        value, pre, _ = self.with_effects(lambda: self.expr(inner))
        return [*pre, Return(value, line=self.line(node))]

    def s_break_statement(self, node: Node) -> list[Stmt]:
        return [Break(line=self.line(node))]

    def s_continue_statement(self, node: Node) -> list[Stmt]:
        return [Continue(line=self.line(node))]

    def s_throw_statement(self, node: Node) -> list[Stmt]:
        return [Return(None, line=self.line(node))]

    def s_switch_statement(self, node: Node) -> list[Stmt]:
        subject_node = self.field(node, "condition", "value")
        test, effects = self.condition(subject_node)
        body = self.field(node, "body")
        arms: list[tuple[Stmt, ...]] = []
        if body is not None:
            for child in self.named(body):
                label_ids = {
                    n.id
                    for n in (
                        self.field(child, "value"),
                        self.field(child, "label"),
                        self.field(child, "pattern"),
                    )
                    if n is not None
                }
                statements: list[Stmt] = []
                for part in self.named(child):
                    if part.id in label_ids or part.type in (
                        "switch_label",
                        "case_label",
                        "default_label",
                    ):
                        continue
                    statements.extend(
                        self.stmt(part) if part.type not in _BLOCK_TYPES else self.stmts(part)
                    )
                arms.append(tuple(statements))
        return [*effects, Switch(test, tuple(arms), line=self.line(node))]

    s_switch_expression = s_switch_statement

    def s_try_statement(self, node: Node) -> list[Stmt]:
        body = self.stmts(self.field(node, "body"))
        handlers: list[tuple[Stmt, ...]] = []
        final: list[Stmt] = []
        for child in self.named(node):
            if child.type in ("catch_clause",):
                handlers.append(tuple(self.stmts(self.field(child, "body"))))
            elif child.type in ("finally_clause",):
                final.extend(self.stmts(self.field(child, "body") or self.first_named(child)))
        return [Try(tuple(body), tuple(handlers), tuple(final), line=self.line(node))]

    s_try_with_resources_statement = s_try_statement

    def s_function_declaration(self, node: Node) -> list[Stmt]:  # a nested JS function
        return [self.function_js(node, None)]

    # ------------------------------------------------------------------ expressions
    def args(self, node: Node | None) -> tuple[Expr, ...]:
        if node is None:
            return ()
        return tuple(self.expr(a) for a in self.named(node))

    def expr(self, node: Node | None) -> Expr:
        if node is None:
            return Unknown("")
        handler = getattr(self, f"x_{node.type}", None)
        if handler is None:
            return Unknown(node.type)
        return handler(node)

    def x_identifier(self, node: Node) -> Expr:
        text = self.text(node)
        if text in ("NULL", "undefined", "nullptr"):
            return Const("none")
        return Name(text)

    x_field_identifier = x_identifier
    x_property_identifier = x_identifier
    x_namespace_identifier = x_identifier
    x_type_identifier = x_identifier

    def x_this(self, node: Node) -> Expr:
        return Name("this")

    def x_super(self, node: Node) -> Expr:
        return Name("super")

    def x_true(self, node: Node) -> Expr:
        return Const("true")

    def x_false(self, node: Node) -> Expr:
        return Const("false")

    def x_null(self, node: Node) -> Expr:
        return Const("none")

    x_nullptr = x_null
    x_undefined = x_null
    x_null_literal = x_null

    def x_number_literal(self, node: Node) -> Expr:
        return Num(_clean_number(self.text(node)))

    x_decimal_integer_literal = x_number_literal
    x_hex_integer_literal = x_number_literal
    x_octal_integer_literal = x_number_literal
    x_binary_integer_literal = x_number_literal
    x_decimal_floating_point_literal = x_number_literal
    x_number = x_number_literal

    def x_char_literal(self, node: Node) -> Expr:
        body = self.text(node)[1:-1]
        if len(body) == 1:
            return Num(ord(body))
        if len(body) == 2 and body[0] == "\\" and body[1] in _CHAR_ESCAPES:
            return Num(_CHAR_ESCAPES[body[1]])
        return Str(body)

    x_character_literal = x_char_literal

    def x_string_literal(self, node: Node) -> Expr:
        text = self.text(node)
        return Str(text[1:-1] if len(text) >= 2 else text)

    x_string = x_string_literal

    def x_template_string(self, node: Node) -> Expr:
        return Str("<template>")

    def x_raw_string_literal(self, node: Node) -> Expr:
        return Str("<raw-string>")

    def x_concatenated_string(self, node: Node) -> Expr:
        return Str("<string>")

    def x_parenthesized_expression(self, node: Node) -> Expr:
        inner = self.first_named(node)
        return self.expr(inner) if inner is not None else Unknown("")

    def x_binary_expression(self, node: Node) -> Expr:
        op = self.operator_text(node)
        left = self.field(node, "left")
        right = self.field(node, "right")
        if op in ("&&", "||", "??"):
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
            return Compare(_COMPARE_OPS[op], self.expr(left), self.expr(right))
        if op in ("in", "instanceof"):
            return Compare("in" if op == "in" else "is", self.expr(left), self.expr(right))
        if op == ">>>":
            op = ">>"
        if op in _BINARY_OPS:
            return BinOp(op, self.expr(left), self.expr(right))
        return Unknown(op or "binary")

    def x_unary_expression(self, node: Node) -> Expr:
        op = self.operator_text(node)
        operand_node = self.field(node, "argument", "operand") or self.first_named(node)
        operand = self.expr(operand_node)
        if op == "!":
            return UnOp("not", operand)
        if op == "-":
            if isinstance(operand, Num) and operand.value is not None:
                return Num(-operand.value)
            return UnOp("-", operand)
        if op == "~":
            return UnOp("~", operand)
        if op == "*":
            return Subscript(operand, Num(0))
        return operand  # +x, &x, typeof x, void x, await x

    def x_pointer_expression(self, node: Node) -> Expr:
        operand = self.expr(self.field(node, "argument") or self.first_named(node))
        return Subscript(operand, Num(0)) if self.operator_text(node) == "*" else operand

    def x_update_expression(self, node: Node) -> Expr:
        line = self.line(node)
        prefix = bool(node.children) and not node.children[0].is_named
        stmt = self.update_stmt(node, line)
        (self._pre if prefix else self._post).append(stmt)
        return stmt.targets[0]

    def x_assignment_expression(self, node: Node) -> Expr:
        stmt = self.assignment(node, self.line(node))
        self._pre.append(stmt)
        return stmt.targets[0]

    x_augmented_assignment_expression = x_assignment_expression

    def x_conditional_expression(self, node: Node) -> Expr:
        return IfExp(
            self.expr(self.field(node, "condition")),
            self.expr(self.field(node, "consequence")),
            self.expr(self.field(node, "alternative")),
        )

    x_ternary_expression = x_conditional_expression

    def x_comma_expression(self, node: Node) -> Expr:
        parts = self.flatten_comma(node)
        for part in parts[:-1]:
            self._pre.extend(self.expr_stmt(part, self.line(part)))
        return self.expr(parts[-1])

    x_sequence_expression = x_comma_expression

    def x_sizeof_expression(self, node: Node) -> Expr:
        return Num(1)

    def x_cast_expression(self, node: Node) -> Expr:
        return self.expr(self.field(node, "value") or self.named(node)[-1])

    x_as_expression = x_cast_expression
    x_non_null_expression = x_cast_expression

    def x_call_expression(self, node: Node) -> Expr:
        function = self.field(node, "function")
        args = self.args(self.field(node, "arguments"))
        if function is not None and function.type in ("template_type", "qualified_identifier"):
            typ = self.type_ref(function)
            if typ is not None and typ.args:
                return New(typ, args)
        if function is not None and function.type == "template_function":
            # `vector<int>(m, 0)` builds a container; `max<int>(a, b)` is an ordinary call
            typ = self.type_ref(function)
            if typ is not None and typ.name in _CONSTRUCTIBLE:
                return New(typ, args)
        return Call(self.expr(function), args)

    def x_method_invocation(self, node: Node) -> Expr:
        obj = self.field(node, "object")
        name = self.field(node, "name")
        args = self.args(self.field(node, "arguments"))
        method = self.text(name) if name is not None else ""
        if obj is None:
            return Call(Name(method), args)
        return Call(Attribute(self.expr(obj), method), args)

    def x_field_expression(self, node: Node) -> Expr:
        argument = self.field(node, "argument", "object")
        field = self.field(node, "field", "property")
        return Attribute(self.expr(argument), self.text(field) if field is not None else "")

    x_member_expression = x_field_expression

    def x_field_access(self, node: Node) -> Expr:
        obj = self.field(node, "object")
        field = self.field(node, "field")
        return Attribute(self.expr(obj), self.text(field) if field is not None else "")

    def x_scoped_identifier(self, node: Node) -> Expr:
        parts = self.text(node).split(".")
        expr: Expr = Name(parts[0])
        for part in parts[1:]:
            expr = Attribute(expr, part)
        return expr

    def x_qualified_identifier(self, node: Node) -> Expr:
        scope = self.field(node, "scope")
        name = self.field(node, "name")
        inner = (
            Name(self.text(name))
            if name is not None and name.type == "identifier"
            else self.expr(name)
        )
        if scope is None:
            return inner
        if isinstance(inner, Name):
            return Attribute(Name(self.text(scope)), inner.id)
        return inner

    def x_template_function(self, node: Node) -> Expr:
        name = self.field(node, "name")
        return self.expr(name) if name is not None else Unknown("template_function")

    def x_subscript_expression(self, node: Node) -> Expr:
        obj = self.field(node, "argument", "object")
        indices = self.field(node, "indices")
        if indices is not None:
            kids = self.named(indices)
            index = self.expr(kids[0]) if kids else Unknown("")
        else:
            index = self.expr(self.field(node, "index"))
        return Subscript(self.expr(obj), index)

    def x_array_access(self, node: Node) -> Expr:
        return Subscript(self.expr(self.field(node, "array")), self.expr(self.field(node, "index")))

    def x_initializer_list(self, node: Node) -> Expr:
        return ListLit(tuple(self.expr(c) for c in self.named(node)), "list")

    x_array_initializer = x_initializer_list
    x_array = x_initializer_list

    def x_object(self, node: Node) -> Expr:
        items: list[tuple[Expr, Expr]] = []
        for pair in self.named(node):
            if pair.type != "pair":
                continue
            key = self.field(pair, "key")
            value = self.field(pair, "value")
            key_expr = Str(self.text(key).strip("'\"")) if key is not None else Str("")
            items.append((key_expr, self.expr(value)))
        return DictLit(tuple(items))

    def x_spread_element(self, node: Node) -> Expr:
        inner = self.first_named(node)
        return Star(self.expr(inner)) if inner is not None else Unknown("spread")

    def x_await_expression(self, node: Node) -> Expr:
        inner = self.first_named(node)
        return self.expr(inner) if inner is not None else Unknown("await")

    def x_new_expression(self, node: Node) -> Expr:
        if self.lang == "javascript":
            constructor = self.field(node, "constructor")
            name = self.text(constructor).split(".")[-1] if constructor is not None else "Object"
            return New(TypeRef(name), self.args(self.field(node, "arguments")))
        typ = self.type_ref(self.field(node, "type")) or TypeRef("?")
        dims: list[Expr] = []
        declarator = self.field(node, "declarator")
        while declarator is not None and declarator.type == "new_declarator":
            length = self.field(declarator, "length")
            if length is not None:
                dims.append(self.expr(length))
            declarator = next(
                (c for c in declarator.named_children if c.type == "new_declarator"), None
            )
        args = self.args(self.field(node, "arguments"))
        if dims:
            return New(TypeRef(typ.name, typ.args, len(dims)), args, tuple(dims))
        return New(typ, args)

    def x_object_creation_expression(self, node: Node) -> Expr:
        typ = self.type_ref(self.field(node, "type")) or TypeRef("?")
        return New(typ, self.args(self.field(node, "arguments")))

    def x_array_creation_expression(self, node: Node) -> Expr:
        element = self.type_ref(self.field(node, "type")) or TypeRef("?")
        dim_exprs: list[Expr] = []
        total = 0
        for child in node.named_children:
            if child.type == "dimensions_expr":
                inner = self.first_named(child)
                dim_exprs.append(self.expr(inner) if inner is not None else Unknown(""))
                total += 1
            elif child.type == "dimensions":
                total += self.text(child).count("[")
        value = self.field(node, "value")
        if value is not None:
            return self.x_initializer_list(value)
        return New(TypeRef(element.name, element.args, total), (), tuple(dim_exprs))

    def x_method_reference(self, node: Node) -> Expr:
        """`Integer::sum`, `String::length`: a function value, opaque to the cost model."""
        return Name(self.text(node))

    def x_lambda_expression(self, node: Node) -> Expr:
        params_node = self.field(node, "parameters")
        names: list[str] = []
        if params_node is not None:
            if params_node.type == "identifier":
                names.append(self.text(params_node))
            else:
                for p in self.named(params_node):
                    if p.type == "identifier":
                        names.append(self.text(p))
                    elif p.type in ("formal_parameter", "parameter_declaration"):
                        nm = self.field(p, "name")
                        names.append(self.text(nm) if nm is not None else "_")
        declarator = self.field(node, "declarator")  # C++: abstract_function_declarator
        if declarator is not None:
            cpp_params = self.field(declarator, "parameters")
            if cpp_params is not None:
                names = [p.name for p in self.params_cpp(cpp_params)]
        body = self.field(node, "body")
        if body is None:
            return Lambda(tuple(names), Unknown(""))
        if body.type in _BLOCK_TYPES:
            return Lambda(tuple(names), tuple(self.stmts(body)))
        return Lambda(tuple(names), self.expr(body))

    def x_arrow_function(self, node: Node) -> Expr:
        params = [p.name for p in self.params_js(self.field(node, "parameters", "parameter"))]
        body = self.field(node, "body")
        if body is None:
            return Lambda(tuple(params), Unknown(""))
        if body.type in _BLOCK_TYPES:
            return Lambda(tuple(params), tuple(self.stmts(body)))
        return Lambda(tuple(params), self.expr(body))

    def x_function_expression(self, node: Node) -> Expr:
        func = self.function_js(node, None)
        return Lambda(tuple(p.name for p in func.params), func.body)

    x_function = x_function_expression


def lower_clike(source: str, language: str) -> Module:
    if language not in ("c", "cpp", "java", "javascript"):
        raise ValueError(f"lower_clike does not handle {language!r}")
    tree = parse_source(source, language)
    return _CLike(source, language).module(tree.root_node)
