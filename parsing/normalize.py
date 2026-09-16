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

from core.ir import IREdge, IRGraph, IRNode
from parsing.parse import parse_source

_LANG_DIR = Path(__file__).resolve().parent / "lang"

_CALL_NODE_TYPES = frozenset({"call", "call_expression"})
_FUNC_DEF_NODE_TYPE = "function_definition"


class _LangConfig:
    def __init__(self, data: dict[str, Any]):
        self.nodes: dict[str, str] = data.get("nodes", {})
        self.calls: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pattern), symbol) for pattern, symbol in data.get("calls", {}).items()
        ]
        self.binary_operator_nodes: frozenset[str] = frozenset(
            data.get("binary_operator_nodes", {}).get("types", [])
        )
        self.binary_operator_map: dict[str, str] = data.get("binary_operator_map", {})
        params = data.get("params", {})
        self.param_container: str | None = params.get("container")
        self.param_leaf_types: frozenset[str] = frozenset(params.get("leaf_types", []))


_CONFIG_CACHE: dict[str, _LangConfig] = {}


def _load_config(language: str) -> _LangConfig:
    if language not in _CONFIG_CACHE:
        with (_LANG_DIR / f"{language}.toml").open("rb") as f:
            data = tomllib.load(f)
        _CONFIG_CACHE[language] = _LangConfig(data)
    return _CONFIG_CACHE[language]


class _Normalizer:
    def __init__(self, config: _LangConfig, source_bytes: bytes):
        self.config = config
        self.source_bytes = source_bytes
        self.nodes: list[IRNode] = []
        self.edges: list[IREdge] = []
        self._next_id = 0
        self._func_name_stack: list[str] = []

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
        func = call_node.child_by_field_name("function")
        return self._text(func) if func is not None else self._text(call_node)

    def _classify_call(self, call_node: TSNode) -> str | None:
        """Returns the IR symbol for a call, or None if it should be elided."""
        callee = self._callee_text(call_node)
        for pattern, symbol in self.config.calls:
            if pattern.search(callee):
                return symbol or None
        return "CALL"

    def _function_name(self, func_def_node: TSNode) -> str | None:
        """First identifier in `func_def_node`, not descending into its body.

        Works across grammars without per-language special-casing because in
        both Python and C++ the function's own name always precedes its body
        in document order, and a preorder walk finds it before ever
        descending into the parameter list (also an identifier source).
        """
        body_types = {"block", "compound_statement"}

        def walk(node: TSNode) -> TSNode | None:
            for child in node.children:
                if child.type in body_types:
                    continue
                if child.type in ("identifier", "field_identifier"):
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

        if node.type == _FUNC_DEF_NODE_TYPE:
            return self._visit_function_def(node, parent_ir)

        mapped_symbol = self.config.nodes.get(node.type)
        if mapped_symbol is not None:
            emitted = self._emit(mapped_symbol, node)
            self.walk(node, emitted)
            return [emitted]

        self.walk(node, parent_ir)
        return []

    def _visit_call(self, node: TSNode, parent_ir: IRNode | None) -> list[IRNode]:
        symbol = self._classify_call(node)
        if symbol == "CALL" and self._func_name_stack:
            if self._callee_text(node) == self._func_name_stack[-1]:
                symbol = "RECURSE"
        args = node.child_by_field_name("arguments")
        if symbol is None:
            if args is not None:
                self.walk(args, parent_ir)
            return []
        emitted = self._emit(symbol, node)
        if args is not None:
            self.walk(args, emitted)
        return [emitted]

    def _visit_function_def(self, node: TSNode, parent_ir: IRNode | None) -> list[IRNode]:
        symbol = self.config.nodes.get(node.type)
        emitted = self._emit(symbol, node) if symbol is not None else None
        name = self._function_name(node)
        if name is not None:
            self._func_name_stack.append(name)
        try:
            self.walk(node, emitted if emitted is not None else parent_ir)
        finally:
            if name is not None:
                self._func_name_stack.pop()
        return [emitted] if emitted is not None else []


def normalize_source(source: str, language: str) -> IRGraph:
    config = _load_config(language)
    tree = parse_source(source, language)
    source_bytes = source.encode("utf-8")
    normalizer = _Normalizer(config, source_bytes)
    normalizer.walk(tree.root_node, None)
    return IRGraph(nodes=normalizer.nodes, edges=normalizer.edges)
