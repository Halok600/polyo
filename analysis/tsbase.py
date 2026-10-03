"""Helpers shared by the tree-sitter lowerers (`lower_clike.py`, `lower_go.py`).

The main non-trivial piece is side-effect hoisting. Lowering an expression such as `a[i++]` or
`x = y = 0` cannot express the increment or the inner assignment as an expression in the engine's
AST, so the lowerer appends them to `_pre` / `_post`; `wrap` then emits them as explicit statements
immediately before / after the statement that contained them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tree_sitter import Node

from analysis.nodes import Stmt, TypeRef

T = TypeVar("T")


class TSBase:
    def __init__(self, source: str) -> None:
        self.src = source.encode("utf-8")
        self._pre: list[Stmt] = []
        self._post: list[Stmt] = []
        self.aliases: dict[str, TypeRef] = {}

    # ------------------------------------------------------------------ type aliases
    def _type_ref(self, node: Node | None) -> TypeRef | None:
        raise NotImplementedError

    def type_ref(self, node: Node | None) -> TypeRef | None:
        """The type a node names, with a name that stands for another type (`typedef vector<int>
        vi;`, `using Grid = vector<vector<int>>;`, Go's `type stack []int`) replaced by it."""
        return self.expand_alias(self._type_ref(node))

    def expand_alias(self, ref: TypeRef | None) -> TypeRef | None:
        if ref is None or ref.args:
            return ref
        alias = self.aliases.get(ref.name)
        if alias is None:
            return ref
        return TypeRef(
            alias.name,
            alias.args,
            alias.dims + ref.dims,
            (*alias.sizes, *ref.sizes),
            ref.ref or alias.ref,
            ref.ptr + alias.ptr,
        )

    def register_alias(self, name: str, target: TypeRef | None) -> None:
        """Remember `name` as another type, but only a collection: a numeric typedef changes
        nothing and a struct's name is what marks its pointers as list or tree nodes."""
        from analysis.roles import _container_kind

        if target is not None and (_container_kind(target.name) is not None or target.dims > 0):
            self.aliases[name] = target

    def text(self, node: Node) -> str:
        return self.src[node.start_byte : node.end_byte].decode("utf-8", "replace")

    @staticmethod
    def line(node: Node) -> int:
        return node.start_point[0] + 1

    @staticmethod
    def named(node: Node) -> list[Node]:
        return [c for c in node.named_children if "comment" not in c.type]

    @staticmethod
    def field(node: Node, *names: str) -> Node | None:
        for name in names:
            child = node.child_by_field_name(name)
            if child is not None:
                return child
        return None

    @staticmethod
    def fields(node: Node, *names: str) -> list[Node]:
        out: list[Node] = []
        for name in names:
            out.extend(node.children_by_field_name(name))
        return out

    def first_named(self, node: Node) -> Node | None:
        kids = self.named(node)
        return kids[0] if kids else None

    def wrap(self, build: Callable[[], list[Stmt]]) -> list[Stmt]:
        """Run `build`, collecting side effects hoisted out of expressions, and surround its
        statements with them."""
        saved = (self._pre, self._post)
        self._pre, self._post = [], []
        try:
            built = build()
            return [*self._pre, *built, *self._post]
        finally:
            self._pre, self._post = saved

    def with_effects(self, build: Callable[[], T]) -> tuple[T, list[Stmt], list[Stmt]]:
        """Like `wrap`, but returns (value, pre, post); the caller decides where effects go."""
        saved = (self._pre, self._post)
        self._pre, self._post = [], []
        try:
            value = build()
            return value, self._pre, self._post
        finally:
            self._pre, self._post = saved
