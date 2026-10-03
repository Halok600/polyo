"""Abstract values for the complexity interpreter.

The interpreter does not run code; it tracks, for each variable, just enough about its value to
bound loops and costs: an integer's magnitude as a `Poly` (so `n // 2`, `hi - lo`, `i + 1` are
exact), a container's element count, how many nodes a linked structure reaches. Everything is an
upper bound on a size, never a concrete value.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace

from analysis.nodes import FuncDef, Lambda
from analysis.poly import Poly, Var

_container_ids = itertools.count(1)

# Containers whose element lookup / insert / delete are O(1) on average.
HASH_KINDS = frozenset({"dict", "set"})
# Ordered trees: O(log n) per operation.
TREE_KINDS = frozenset({"treemap", "treeset"})
# Double-ended queues: O(1) at both ends.
DEQUE_KINDS = frozenset({"deque", "queue"})
SEQUENCE_KINDS = frozenset({"list", "array", "str"})


def new_container_id() -> int:
    return next(_container_ids)


@dataclass(frozen=True, slots=True)
class IntV:
    """An integer. `mag` is its magnitude as a polynomial in size variables; None means unknown."""

    mag: Poly | None = None
    of: int | None = None  # set when this integer is `len(c)` of the container with this uid


@dataclass(frozen=True, slots=True)
class ScalarV:
    """An opaque O(1) value: a bool, a char, a float, an element of unknown type."""


@dataclass(frozen=True, slots=True)
class NoneV:
    """`None` / `null` / `nil` / `nullptr` -- the end of a linked structure."""


@dataclass(frozen=True, slots=True)
class ContV:
    kind: str  # list array str dict set treemap treeset deque queue stack heap
    length: Poly  # an upper bound on the number of elements
    elem: Value | None = None  # element (a row of a 2-D structure, or a dict's value)
    uid: int = 0  # identity, so index domains and growth can name the container
    cap: Var | None = None  # symbolic capacity of a growable local container
    view: bool = False  # a slice VIEW sharing storage with its source (Go)
    owned: bool = False  # allocated by the code under analysis (an input parameter is not)
    ragged: bool = False  # rows differ in length (an adjacency list); False: a rectangular grid


@dataclass(frozen=True, slots=True)
class NodeV:
    """A pointer into a linked structure (list, tree) that reaches `size` nodes."""

    size: Poly


@dataclass(frozen=True, slots=True)
class TupleV:
    items: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class FuncV:
    func: FuncDef | Lambda
    closure: object | None = None  # the defining Env (for nested functions)
    name: str = ""


@dataclass(frozen=True, slots=True)
class UnknownV:
    reason: str = ""


Value = IntV | ScalarV | NoneV | ContV | NodeV | TupleV | FuncV | UnknownV

SCALAR = ScalarV()
NONE = NoneV()
UNKNOWN = UnknownV()


def const_int(value: int | float | None) -> IntV:
    if value is None:
        return IntV(None)
    if isinstance(value, float):
        value = int(value) if value.is_integer() else 1
    return IntV(Poly.const(value))  # the SIGN matters: range(n, -1, -1) counts down


def make_container(
    kind: str,
    length: Poly,
    elem: Value | None = None,
    *,
    cap: Var | None = None,
    view: bool = False,
    owned: bool = False,
    ragged: bool = False,
    uid: int | None = None,
) -> ContV:
    return ContV(
        kind=kind,
        length=length,
        elem=elem,
        uid=uid if uid is not None else new_container_id(),
        cap=cap,
        view=view,
        owned=owned,
        ragged=ragged,
    )


def mem(value: Value | None) -> Poly:
    """Memory units a value occupies: 1 for a scalar, `length * mem(elem)` for a container."""
    if isinstance(value, ContV):
        per = mem(value.elem) if value.elem is not None else Poly.const(1)
        return (value.length * per).order()
    if isinstance(value, TupleV):
        total = Poly.zero()
        for item in value.items:
            total += mem(item)
        return total.order() if not total.is_zero() else Poly.const(1)
    return Poly.const(1)


def magnitude(value: Value | None) -> Poly | None:
    """The integer magnitude of a value, or None if it is not a usable integer."""
    if isinstance(value, IntV):
        return value.mag
    return None


def length_of(value: Value | None) -> Poly | None:
    if isinstance(value, ContV):
        return value.length
    if isinstance(value, TupleV):
        return Poly.const(len(value.items))
    return None


def substitute(value: Value | None, mapping: Mapping[Var, Poly]) -> Value | None:
    """Rewrite the size variables inside a value (used to instantiate a callee's result)."""
    if value is None or not mapping:
        return value
    # sizes stay exact (n - 2, not n): a recursion is measured by how much its argument shrinks
    if isinstance(value, IntV):
        mag = value.mag.substitute(mapping) if value.mag is not None else None
        return IntV(mag, value.of)
    if isinstance(value, ContV):
        elem = substitute(value.elem, mapping)
        return replace(value, length=value.length.substitute(mapping), elem=elem)
    if isinstance(value, NodeV):
        return NodeV(value.size.substitute(mapping))
    if isinstance(value, TupleV):
        return TupleV(tuple(substitute(i, mapping) or UNKNOWN for i in value.items))
    return value


def join(a: Value | None, b: Value | None) -> Value:
    """The value after control flow merges: the larger of two size bounds."""
    if a is None:
        return b if b is not None else UNKNOWN
    if b is None or a == b:
        return a
    if isinstance(a, IntV) and isinstance(b, IntV):
        if a.mag is None or b.mag is None:
            return IntV(None)
        return IntV(a.mag.max(b.mag), a.of if a.of == b.of else None)
    if isinstance(a, ContV) and isinstance(b, ContV) and a.kind == b.kind:
        elem = (
            join(a.elem, b.elem) if a.elem is not None and b.elem is not None else a.elem or b.elem
        )
        keep = a if a.uid == b.uid else None
        return ContV(
            kind=a.kind,
            length=a.length.max(b.length),
            elem=elem,
            uid=keep.uid if keep else new_container_id(),
            cap=a.cap if a.cap == b.cap else None,
            view=a.view and b.view,
            owned=a.owned and b.owned,
            ragged=a.ragged or b.ragged,
        )
    if isinstance(a, NodeV) and isinstance(b, NodeV):
        return NodeV(a.size.max(b.size))
    if isinstance(a, TupleV) and isinstance(b, TupleV) and len(a.items) == len(b.items):
        return TupleV(tuple(join(x, y) for x, y in zip(a.items, b.items, strict=True)))
    if isinstance(a, NoneV) and isinstance(b, NodeV):
        return b
    if isinstance(a, NodeV) and isinstance(b, NoneV):
        return a
    if isinstance(a, ScalarV) and isinstance(b, ScalarV):
        return a
    return UNKNOWN
