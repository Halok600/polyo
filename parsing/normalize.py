"""CST -> normalised cross-language IR (plan §5).

Each language's semantics are declared in `parsing/lang/<lang>.toml`
(node-type mapping, call-name classification, binary-operator
disambiguation, parameter-container leaves) -- this module is the one
generic walker all languages share, so adding a language is a new toml file
plus a grammar dependency in `parsing/parse.py`, not new walker code.

Design notes on what is deliberately NOT emitted as an IR node (elided, but
still walked into for its children):
  - A call whose callee matches a `[calls]` pattern mapped to `""` (e.g.
    Python's `range(...)`) -- it is loop-bound syntax, not an operation.
  - Any CST node with no entry in `[nodes]` (punctuation, keywords, wrapper
    nodes like `argument_list`/`parameters`) -- it is flattened so its
    meaningful descendants still attach to the nearest emitted ancestor.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from tree_sitter import Node as TSNode

from core.ir import IREdge, IRGraph, IRNode, data_dependency_edges, loop_carry_edges
from parsing.parse import UnsupportedLanguageError, parse_source

_LANG_DIR = Path(__file__).resolve().parent / "lang"

# Node types that represent a call across all supported grammars -- shared
# because recognising "this is call-shaped" is not a per-language semantic
# decision, unlike everything in `parsing/lang/*.toml`. C#'s is
# `invocation_expression` (verified against a real parse, not guessed --
# same discipline as the other five grammars' `call`/`call_expression`).
_CALL_NODE_TYPES = frozenset(
    {"call", "call_expression", "method_invocation", "invocation_expression"}
)
# Node types that wrap a function's body -- skipped by `_function_name`'s
# walk for the same reason (see its docstring). Shared for the same reason
# as `_CALL_NODE_TYPES` above.
_BODY_NODE_TYPES = frozenset({"block", "compound_statement", "statement_block"})


class _LangConfig:
    def __init__(self, data: dict[str, Any]):
        self.nodes: dict[str, str] = data.get("nodes", {})
        # Derived, not a separate toml section: whichever node type(s) a
        # language maps to FUNC_DEF are the ones that carry a function name
        # onto `_func_name_stack` for recursion detection. This used to be a
        # single hardcoded node type ("function_definition"), which only
        # worked for Python/C++ by coincidence -- Java's is
        # `method_declaration`, Go's is `function_declaration`.
        self.func_def_types: frozenset[str] = frozenset(
            node_type for node_type, symbol in self.nodes.items() if symbol == "FUNC_DEF"
        )
        self.calls: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pattern), symbol) for pattern, symbol in data.get("calls", {}).items()
        ]
        self.calls_by_arity: list[tuple[re.Pattern[str], dict[str, str]]] = [
            (re.compile(pattern), branches)
            for pattern, branches in data.get("calls_by_arity", {}).items()
        ]
        self.binary_operator_nodes: frozenset[str] = frozenset(
            data.get("binary_operator_nodes", {}).get("types", [])
        )
        self.binary_operator_map: dict[str, str] = data.get("binary_operator_map", {})
        params = data.get("params", {})
        self.param_container: str | None = params.get("container")
        self.param_leaf_types: frozenset[str] = frozenset(params.get("leaf_types", []))
        # Optional: a callee pattern (Python's `range`) whose call denotes a
        # literal iteration COUNT rather than "iterate this container" --
        # only languages with that idiom declare it (see `_classify_loop`'s
        # docstring). Absent for languages without it, which is safe: those
        # iterator-style loops just always classify as LOOP_N_BOUND, never
        # falsely as LOOP_CONST_BOUND.
        loop_shape = data.get("loop_shape", {})
        range_call = loop_shape.get("range_call")
        self.range_call: re.Pattern[str] | None = re.compile(range_call) if range_call else None


_CONFIG_CACHE: dict[str, _LangConfig] = {}


def _load_config(language: str) -> _LangConfig:
    if language not in _CONFIG_CACHE:
        try:
            with (_LANG_DIR / f"{language}.toml").open("rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError as e:
            raise UnsupportedLanguageError(f"no IR mapping for language: {language!r}") from e
        _CONFIG_CACHE[language] = _LangConfig(data)
    return _CONFIG_CACHE[language]


def _trailing_call_segment(callee_text: str) -> str:
    """The text after a qualified call's last member-access operator --
    `self.fib` -> `fib`, `this->fib` -> `fib` (C++'s pointer-member arrow,
    distinct from every other served language's plain `.`) -- or the whole
    text if unqualified. Shared by RECURSE detection (`_visit_call`) and
    CALL_EDGE resolution (`resolve_call_edges`) below, since a qualified
    call's receiver carries no name-resolution information this static IR
    tracks anyway -- both places only ever care whether the CALLED name
    matches a known function name."""
    return callee_text.replace("->", ".").rsplit(".", 1)[-1]


class _Normalizer:
    def __init__(self, config: _LangConfig, source_bytes: bytes):
        self.config = config
        self.source_bytes = source_bytes
        self.nodes: list[IRNode] = []
        self.edges: list[IREdge] = []
        self._next_id = 0
        self._func_name_stack: list[str] = []
        # CALL_EDGE (plan §5) needs a whole-file view -- a call may reference
        # a function defined later in the file -- so candidates are recorded
        # here during `walk()` and resolved once in `finalize()`.
        self._pending_calls: list[tuple[IRNode, str]] = []
        self._func_defs: list[tuple[str, IRNode]] = []

    def _text(self, node: TSNode) -> str:
        return self.source_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def _emit(self, symbol: str, node: TSNode) -> IRNode:
        ir_node = IRNode(
            id=self._next_id,
            symbol=symbol,
            span=(node.start_point[0], node.start_point[1], node.end_point[0], node.end_point[1]),
            text=self._text(node),
        )
        self._next_id += 1
        self.nodes.append(ir_node)
        return ir_node

    def _callee_text(self, call_node: TSNode) -> str:
        # Most grammars (Python, C/C++, JavaScript, Go) expose the callee as
        # a single "function" field, whether or not it's qualified --
        # `a.sort` already comes back as one field's text. Java's grammar is
        # the odd one out: `method_invocation` splits a qualified call into
        # separate "object" and "name" fields with no single node spanning
        # both, so that shape is reassembled here instead of in a
        # per-language config knob, since it's a call-shape quirk rather
        # than a semantic choice.
        name = call_node.child_by_field_name("name")
        if name is not None:
            obj = call_node.child_by_field_name("object")
            return f"{self._text(obj)}.{self._text(name)}" if obj is not None else self._text(name)
        func = call_node.child_by_field_name("function")
        if func is not None:
            return self._text(func)
        # Kotlin's `call_expression` exposes no field at all for the
        # callee -- neither "name"/"object" (Java's shape) nor "function"
        # (every other grammar's) -- verified against a real parse. The
        # callee is positionally always the node's first child, with the
        # argument list (`value_arguments`) as a later sibling, so this
        # falls back to that rather than the call's full text (which would
        # include the arguments and break both `[calls]` regex matching
        # and recursion self-detection).
        if call_node.children:
            return self._text(call_node.children[0])
        return self._text(call_node)

    def _call_arg_count(self, call_node: TSNode) -> int:
        args = call_node.child_by_field_name("arguments")
        return len(args.named_children) if args is not None else 0

    def _classify_call(self, call_node: TSNode) -> str | None:
        """Returns the IR symbol for a call, or None if it should be elided.

        `calls_by_arity` is checked first: a handful of callees (`min`/`max`
        across every supported language) have a genuinely different static
        complexity depending on how many arguments they're given -- one
        argument means "scan an iterable" (O(n)), two or more means "compare
        a fixed handful of scalars" (O(1)) -- a real disambiguation, not a
        type-inference workaround, since argument COUNT is directly readable
        from the call node without knowing anything about argument types."""
        callee = self._callee_text(call_node)
        for pattern, branches in self.config.calls_by_arity:
            if pattern.search(callee):
                key = "one" if self._call_arg_count(call_node) <= 1 else "many"
                return branches.get(key) or None
        for pattern, symbol in self.config.calls:
            if pattern.search(callee):
                return symbol or None
        return "CALL"

    def _function_name(self, func_def_node: TSNode) -> str | None:
        """First identifier in `func_def_node`, not descending into its body.

        Works across grammars without per-language special-casing because in
        every supported language the function's own name always precedes its
        body in document order, and a preorder walk finds it before ever
        descending into the parameter list (also an identifier source) --
        EXCEPT a JS/TS class method (`method_definition`), whose own name is
        a `property_identifier` node, not `identifier`/`field_identifier`
        (also matched below), and Go's `method_declaration`, whose RECEIVER
        (`func (s *Solution) fib(...)`) is itself a parameter list that
        precedes the method's own name in document order, breaking the
        "walk finds the name before any parameter list" assumption outright
        -- a receiver isn't a name-bearing parameter list the walk should
        skip by type, it's a parameter list that happens to come first. Try
        the grammar's own `name` field first (Go's `method_declaration`
        exposes exactly this, distinct from its `receiver` field); most
        grammars have no such field at all, in which case this is a no-op
        and the preorder walk below runs exactly as it always did. Both
        gaps were caught together via a live adversarial test showing
        qualified recursive calls (`self.fib`/`this.fib`/a Go receiver's
        `s.fib`) misclassified as non-recursive, because the "function
        name" compared against was actually a parameter/receiver name.
        """
        named_field = func_def_node.child_by_field_name("name")
        if named_field is not None:
            return self._text(named_field)

        def walk(node: TSNode) -> TSNode | None:
            for child in node.children:
                if child.type in _BODY_NODE_TYPES:
                    continue
                if child.type in ("identifier", "field_identifier", "property_identifier"):
                    return child
                found = walk(child)
                if found is not None:
                    return found
            return None

        name_node = walk(func_def_node)
        return self._text(name_node) if name_node is not None else None

    def _operator_text(self, node: TSNode) -> str | None:
        op = node.child_by_field_name("operator")
        return op.type if op is not None else None

    def _clause_container(self, node: TSNode) -> TSNode:
        """Go wraps a 3-clause for-loop's init/condition/update in a
        `for_clause` child with no field name of its own (unlike every
        other grammar here, which exposes `condition`/`update` directly on
        the for-loop node) -- unwrap it so the rest of loop-shape
        classification below is uniform across languages."""
        for child in node.children:
            if child.type == "for_clause":
                return child
        return node

    def _update_node(self, container: TSNode) -> TSNode | None:
        # Every C-family grammar here field-names a for-loop's update
        # clause "update" except JavaScript/TypeScript, which calls it
        # "increment" -- verified against real parses, not guessed.
        for field_name in ("update", "increment"):
            found = container.child_by_field_name(field_name)
            if found is not None:
                return found
        return None

    def _is_halving_update(self, update: TSNode) -> bool:
        op = update.child_by_field_name("operator")
        return op is not None and self._text(op) in {"*=", "/=", "<<=", ">>="}

    def _condition_node(self, container: TSNode) -> TSNode | None:
        found = container.child_by_field_name("condition")
        if found is not None:
            return found
        # Go's bare `for cond { ... }` (no init/update clauses) exposes the
        # condition as an unnamed direct child instead of a "condition"
        # field -- the one shape among these grammars with no field name
        # for it at all.
        body = container.child_by_field_name("body")
        for child in container.children:
            if child.type == "for" or child is body:
                continue
            return child
        return None

    def _bound_operand(self, condition: TSNode) -> TSNode | None:
        """The condition's own comparison right-operand -- conventionally
        the loop's bound, since the left operand is conventionally the
        loop/comparison variable. Searches the condition's subtree (which
        several grammars wrap in an extra node -- C++'s `condition_clause`,
        Java/JS's `parenthesized_expression`) for the first node exposing
        both `left` and `right` fields, rather than naming every wrapper
        type explicitly. Falls back to the condition's own last child for a
        grammar with no left/right field names on a comparison at all
        (Python's `comparison_operator`, which only field-names the
        operator itself, not its operands)."""
        stack = [condition]
        while stack:
            current = stack.pop()
            left = current.child_by_field_name("left")
            right = current.child_by_field_name("right")
            if left is not None and right is not None:
                return right
            stack.extend(current.children)
        return condition.children[-1] if condition.child_count >= 2 else None

    def _is_constant_subtree(self, node: TSNode) -> bool:
        """True if no `identifier` node appears anywhere in `node`'s
        subtree -- built purely from literals and operators (`10`, `26`,
        `100 - 1`), as opposed to referencing a variable (`n`, or
        `arr.length`, whose base `arr` is itself an `identifier`)."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "identifier":
                return False
            stack.extend(current.children)
        return True

    def _classify_iterator_loop(self, iterable: TSNode) -> str:
        if (
            self.config.range_call is not None
            and iterable.type in _CALL_NODE_TYPES
            and self.config.range_call.search(self._callee_text(iterable))
        ):
            args = iterable.child_by_field_name("arguments")
            if args is not None and self._is_constant_subtree(args):
                return "LOOP_CONST_BOUND"
        return "LOOP_N_BOUND"

    def _classify_condition_loop(self, node: TSNode, base_symbol: str) -> str:
        container = self._clause_container(node)
        update = self._update_node(container)
        if update is not None and self._is_halving_update(update):
            return "LOOP_HALVING"
        if base_symbol != "LOOP_FOR":
            # A `while`/`loop` node has no init clause at all -- the
            # compared variable's starting value is set somewhere outside
            # this node entirely, so a constant bound here (`count > 0`,
            # `node is not None`) does NOT mean a constant trip count: the
            # loop runs once per unit of whatever `count`/`node` actually
            # started as, which is invisible without dataflow this static
            # IR doesn't attempt. Only a C-style for-loop's own init
            # clause -- right there in the same node -- gives a
            # (still heuristic, not proven) reason to trust a literal
            # bound; see below.
            return "LOOP_N_BOUND"
        condition = self._condition_node(container)
        if condition is None:
            return "LOOP_N_BOUND"
        bound = self._bound_operand(condition)
        if bound is None:
            return "LOOP_N_BOUND"
        return "LOOP_CONST_BOUND" if self._is_constant_subtree(bound) else "LOOP_N_BOUND"

    def _classify_loop(self, node: TSNode, base_symbol: str) -> str:
        """Refines a plain LOOP_FOR/LOOP_WHILE mapping into a bound-shape
        symbol (LOOP_CONST_BOUND / LOOP_N_BOUND / LOOP_HALVING -- reserved
        in `core/ir.py`'s vocabulary since Phase 0 but never actually
        emitted until now).

        Two loop shapes exist across these grammars: Python's (and JS's
        for-of/for-in) "iterate this expression" style, which exposes the
        iterable directly via a `right` field, and everyone else's C-style
        init/condition/update style, dispatched on whether `right` is
        present rather than on `base_symbol` (Go has no separate
        `while_statement` node type at all -- a bare `for cond {}` is still
        mapped LOOP_FOR, and correctly falls through to the condition-style
        path below since it has no `right` field either).

        HALVING is intentionally narrow: only the syntactically explicit
        case (a for-loop's own update clause multiplies/divides, e.g.
        `i *= 2`) is detected. A `while`-loop halving idiom (`while hi >
        lo: mid = (hi+lo)//2; ...`) would need body dataflow analysis this
        static IR doesn't attempt, so it falls back to LOOP_N_BOUND instead
        of guessing -- named here rather than silently missed, same as
        `features/tabular.py`'s own documented scope limits."""
        if base_symbol == "LOOP_FOR":
            # "right" (Python's for_statement, JS's for_in_statement) or
            # "value" (Java's enhanced_for_statement) -- verified against
            # real parses, not guessed.
            iterable = node.child_by_field_name("right") or node.child_by_field_name("value")
            if iterable is not None:
                return self._classify_iterator_loop(iterable)
        return self._classify_condition_loop(node, base_symbol)

    def walk(self, node: TSNode, parent_ir: IRNode | None) -> None:
        prev: IRNode | None = None
        for child in node.children:
            for emitted in self.visit(child, parent_ir):
                if parent_ir is not None:
                    self.edges.append(IREdge(parent_ir.id, emitted.id, "AST_CHILD"))
                if prev is not None:
                    self.edges.append(IREdge(prev.id, emitted.id, "NEXT_SIBLING"))
                prev = emitted

    def visit(self, node: TSNode, parent_ir: IRNode | None) -> list[IRNode]:
        if node.type == self.config.param_container:
            return [
                self._emit("PARAM", child)
                for child in node.children
                if child.type in self.config.param_leaf_types
            ]

        if node.type in _CALL_NODE_TYPES:
            return self._visit_call(node, parent_ir)

        if node.type in self.config.binary_operator_nodes:
            op_text = self._operator_text(node)
            symbol = self.config.binary_operator_map.get(op_text or "", "UNKNOWN")
            emitted = self._emit(symbol, node)
            self.walk(node, emitted)
            return [emitted]

        if node.type in self.config.func_def_types:
            return self._visit_function_def(node, parent_ir)

        mapped_symbol = self.config.nodes.get(node.type)
        if mapped_symbol in ("LOOP_FOR", "LOOP_WHILE"):
            emitted = self._emit(self._classify_loop(node, mapped_symbol), node)
            self.walk(node, emitted)
            return [emitted]
        if mapped_symbol is not None:
            emitted = self._emit(mapped_symbol, node)
            self.walk(node, emitted)
            return [emitted]

        self.walk(node, parent_ir)
        return []

    def _visit_call(self, node: TSNode, parent_ir: IRNode | None) -> list[IRNode]:
        symbol = self._classify_call(node)
        if symbol == "CALL" and self._func_name_stack:
            callee = self._callee_text(node)
            current = self._func_name_stack[-1]
            # A qualified self-call (`self.fib(...)`, `this.helper(...)`,
            # C++'s `this->fib(...)`) is still direct recursion -- match on
            # the trailing segment, same leniency `resolve_call_edges` below
            # already applies when resolving a qualified CALL_EDGE to its
            # FUNC_DEF. Without this, every OOP-style recursive method
            # (calling itself via `self.`/`this.`/`this->` rather than its
            # bare name) silently fails the strict equality check and gets
            # classified as a plain CALL, never RECURSE -- the same
            # recursion signal downstream features and the served model
            # rely on for O(2^n)/O(n) recursion shapes.
            if callee == current or _trailing_call_segment(callee) == current:
                symbol = "RECURSE"
        args = node.child_by_field_name("arguments")
        if symbol is None:
            if args is not None:
                self.walk(args, parent_ir)
            return []
        emitted = self._emit(symbol, node)
        if symbol in ("CALL", "RECURSE"):
            self._pending_calls.append((emitted, self._callee_text(node)))
        if args is not None:
            self.walk(args, emitted)
        return [emitted]

    def _visit_function_def(self, node: TSNode, parent_ir: IRNode | None) -> list[IRNode]:
        symbol = self.config.nodes.get(node.type)
        emitted = self._emit(symbol, node) if symbol is not None else None
        name = self._function_name(node)
        if name is not None:
            self._func_name_stack.append(name)
            if emitted is not None:
                self._func_defs.append((name, emitted))
        try:
            self.walk(node, emitted if emitted is not None else parent_ir)
        finally:
            if name is not None:
                self._func_name_stack.pop()
        return [emitted] if emitted is not None else []

    def resolve_call_edges(self) -> None:
        """CALL_EDGE: call site -> the FUNC_DEF it resolves to, by in-file
        name match (first definition wins on a duplicate name, for
        determinism). Only CALL/RECURSE sites are candidates -- a library
        call (SORT, HEAP_PUSH, ...) isn't a call into this file's own code.
        A qualified callee (`self.helper`, `obj.method`, C++'s
        `this->helper`) also matches on its trailing segment (see
        `_trailing_call_segment`), since the static IR has no receiver-type
        information to resolve the qualifier itself. Run once after `walk()`
        completes -- see `_pending_calls`'s docstring above."""
        by_name: dict[str, IRNode] = {}
        for name, func_node in self._func_defs:
            by_name.setdefault(name, func_node)
        for call_node, callee_text in self._pending_calls:
            target = by_name.get(callee_text) or by_name.get(_trailing_call_segment(callee_text))
            if target is not None:
                self.edges.append(IREdge(call_node.id, target.id, "CALL_EDGE"))


def normalize_source(source: str, language: str) -> IRGraph:
    config = _load_config(language)
    tree = parse_source(source, language)
    source_bytes = source.encode("utf-8")
    normalizer = _Normalizer(config, source_bytes)
    normalizer.walk(tree.root_node, None)
    normalizer.resolve_call_edges()
    graph = IRGraph(nodes=normalizer.nodes, edges=normalizer.edges)
    graph.edges.extend(data_dependency_edges(graph))
    graph.edges.extend(loop_carry_edges(graph))
    return graph
