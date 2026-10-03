"""Parameter roles: which parameters are input sizes, collections or linked nodes.

Typed languages (C, C++, Java, Go, annotated Python) declare this, so the declared type is read
directly. Python and JavaScript declare nothing, so the role is inferred from how each parameter
is USED. Inference follows the usual evidence: indexed / iterated / `len()`ed / given a collection
method means a collection; `.next` / `.left` / `.right` means a linked node; anything else is an
integer. The role only has to be right enough to decide what a loop over the parameter costs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from analysis.nodes import (
    Attribute,
    Call,
    Comp,
    Compare,
    Expr,
    FuncDef,
    Lambda,
    Loop,
    Name,
    New,
    Slice,
    Star,
    Str,
    Subscript,
    TypeRef,
    walk,
)
from analysis.poly import Poly, new_var
from analysis.values import (
    SCALAR,
    UNKNOWN,
    ContV,
    IntV,
    NodeV,
    Value,
    make_container,
)

NODE_ATTRS = frozenset({"next", "left", "right", "prev", "parent", "children", "child"})
CONTAINER_METHODS = frozenset(
    {
        "append", "pop", "extend", "insert", "remove", "sort", "reverse", "keys", "values",
        "items", "get", "add", "push", "popleft", "appendleft", "count", "index", "clear",
        "update", "copy", "discard", "setdefault", "push_back", "pop_back", "size", "length",
        "empty", "front", "back", "begin", "end", "contains", "containsKey", "put", "offer",
        "poll", "peek", "includes", "indexOf", "slice", "splice", "shift", "unshift", "map",
        "filter", "forEach", "reduce", "at", "has", "set", "delete",
    }
)  # fmt: skip
STRING_METHODS = frozenset(
    {
        "lower", "upper", "strip", "lstrip", "rstrip", "split", "startswith", "endswith",
        "isdigit", "isalpha", "isalnum", "replace", "find", "rfind", "format", "encode",
        "charAt", "substring", "substr", "toLowerCase", "toUpperCase", "trim", "isupper",
        "islower", "title", "zfill",
    }
)  # fmt: skip
CONTAINER_CONSUMERS = frozenset(
    {
        "len", "sorted", "sum", "max", "min", "list", "set", "tuple", "dict", "enumerate",
        "reversed", "zip", "any", "all", "map", "filter", "iter", "heapify", "deque",
        "frozenset", "Counter",
    }
)  # fmt: skip

_ITERABLE_CONSTRUCTORS = frozenset({"Set", "Map", "WeakSet", "WeakMap"})
_CALLBACK_METHODS = frozenset(
    {"map", "filter", "forEach", "reduce", "reduceRight", "some", "every", "find", "findIndex",
     "findLast", "findLastIndex", "flatMap", "anyMatch", "allMatch", "noneMatch", "removeIf"}
)  # fmt: skip
_MODULE_CONSUMERS = frozenset(
    {
        "bisect_left", "bisect_right", "bisect", "insort", "heappush", "heappop", "heapify",
        "heappushpop", "heapreplace", "nlargest", "nsmallest", "permutations", "combinations",
        "accumulate", "chain", "groupby", "deepcopy", "copy", "shuffle", "choice", "sample",
        "fsum", "mean", "median", "reduce", "sort", "sorted", "sum", "max", "min", "reverse",
        "fill", "binarySearch", "asList", "copyOf", "copyOfRange", "stream", "join", "Join",
        "Ints", "Strings", "Sort", "Slice", "SliceStable", "Contains", "Index", "Reverse",
    }
)  # fmt: skip
# a 2-D parameter with one of these names is an adjacency structure whose rows differ in length;
# any other (grid, matrix, board, dp ...) is read as rectangular
_GRAPH_NAME = re.compile(
    r"graph|adj|neighbo|connect|route|road|link|network|friend|prereq|depend|child|tree"
    r"|room|node|bucket|group|^g$",
    re.IGNORECASE,
)
# a list of small tuples: `intervals[i]` is [start, end], `edges[i]` is [u, v]
_PAIR_NAME = re.compile(
    r"interval|pair|point|edge|quer|meeting|event|segment|range|job|task|coordinate|coord"
    r"|rectangle|trip|flight|ticket|relation|domino|offer|booking|request|operation|update",
    re.IGNORECASE,
)


def graph_like(name: str) -> bool:
    return _GRAPH_NAME.search(name.rstrip("[]")) is not None


def pair_rows(name: str) -> bool:
    """True for a 2-D parameter whose rows are pairs/triples rather than variable-length lists."""
    return _PAIR_NAME.search(name.rstrip("[]")) is not None


def _row_length(name: str) -> Poly:
    """Length of the container called `name`: a fresh size, except the rows (`name[]`) of a list of
    pairs, which have a fixed handful of entries."""
    if name.endswith("[]") and pair_rows(name):
        return Poly.const(2)
    return Poly.var(new_var(name, "len"))


INT_TYPES = frozenset(
    {
        "int", "long", "short", "byte", "size_t", "uint", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "long long", "unsigned", "unsigned int",
        "unsigned long", "unsigned long long", "Integer", "Long", "Short", "Byte", "BigInteger",
        "number", "ssize_t", "ptrdiff_t", "uintptr",
    }
)  # fmt: skip
SCALAR_TYPES = frozenset(
    {
        "bool", "boolean", "Boolean", "float", "double", "float32", "float64", "Float",
        "Double", "char", "Character", "rune", "void", "None", "any", "interface",
        "auto", "var", "object", "Object", "T", "pair", "tuple", "Pair", "Entry",
    }
)  # fmt: skip
STRING_TYPES = frozenset(
    {"string", "String", "str", "StringBuilder", "StringBuffer", "CharSequence"}
)
LIST_TYPES = frozenset(
    {"vector", "List", "ArrayList", "list", "Vector", "array", "Array", "Collection", "Iterable"}
)
DEQUE_TYPES = frozenset({"deque", "Deque", "ArrayDeque", "LinkedList", "queue", "Queue"})
STACK_TYPES = frozenset({"stack", "Stack"})
HEAP_TYPES = frozenset(
    {"priority_queue", "PriorityQueue", "MinPriorityQueue", "MaxPriorityQueue", "heap"}
)
DICT_TYPES = frozenset(
    {
        "unordered_map", "HashMap", "Map", "dict", "defaultdict", "Counter", "OrderedDict",
        "LinkedHashMap", "map_hash", "Object_dict", "Hashtable", "ConcurrentHashMap",
    }
)  # fmt: skip
SET_TYPES = frozenset({"unordered_set", "HashSet", "Set", "LinkedHashSet", "frozenset"})
TREEMAP_TYPES = frozenset({"map", "TreeMap", "multimap", "SortedDict"})
TREESET_TYPES = frozenset({"set", "TreeSet", "multiset", "SortedSet"})


@dataclass(frozen=True, slots=True)
class RoleInfo:
    role: str  # container | node | int
    depth: int = 1  # container nesting depth (a grid is 2)
    kind: str = "list"  # list | str


# --------------------------------------------------------------------------------- inference
def _root_of(expr: Expr, aliases: dict[str, tuple[str, int]]) -> tuple[str, int] | None:
    """(parameter name, subscript depth) when `expr` is `p`, `p[i]`, `p[i][j]`, `p[i:j]`, or a
    loop alias of one of those."""
    depth = 0
    current = expr
    while isinstance(current, Subscript | Slice):
        depth += 1
        current = current.obj
    if isinstance(current, Name):
        if current.id in aliases:
            root, base = aliases[current.id]
            return root, base + depth
        return current.id, depth
    return None


def infer_roles(
    func: FuncDef,
    callees: Mapping[str, FuncDef] | None = None,
    _active: frozenset[str] = frozenset(),
) -> dict[str, RoleInfo]:
    """Roles of func's parameters from how they are used. `callees` (the other functions of the
    file, by name) lets a parameter passed on to a function inherit what that function does with
    it: `row_sum(grid, i)` where `row_sum` iterates `grid[i]` makes `grid` a grid."""
    names = [p.name for p in func.params]
    node_evidence: set[str] = set()
    container_depth: dict[str, int] = {}
    string_evidence: set[str] = set()
    aliases: dict[str, tuple[str, int]] = {}

    def container(root: tuple[str, int] | None, extra: int = 1) -> None:
        if root is None or root[0] not in names:
            return
        container_depth[root[0]] = max(container_depth.get(root[0], 0), root[1] + extra)

    # aliases first: `for row in grid` makes `row` one level below `grid`
    for node in walk(func):
        if isinstance(node, Loop) and node.kind == "for_each" and isinstance(node.target, Name):
            root = _root_of(node.iter, aliases) if node.iter is not None else None
            if root is not None and root[0] in names:
                aliases[node.target.id] = (root[0], root[1] + 1)
        elif isinstance(node, Comp):
            for generator in node.generators:
                if isinstance(generator.target, Name):
                    root = _root_of(generator.iter, aliases)
                    if root is not None and root[0] in names:
                        aliases[generator.target.id] = (root[0], root[1] + 1)
        elif (
            isinstance(node, Call)
            and isinstance(node.func, Attribute)
            and node.func.attr in _CALLBACK_METHODS
        ):  # `grid.map(row => ...)`: the callback's item is one level below `grid`
            root = _root_of(node.func.obj, aliases)
            if root is not None and root[0] in names:
                for arg in node.args:
                    if isinstance(arg, Lambda) and arg.params:
                        folds = node.func.attr in ("reduce", "reduceRight") and len(arg.params) >= 2
                        aliases[arg.params[1 if folds else 0]] = (root[0], root[1] + 1)

    for node in walk(func):
        if isinstance(node, Attribute):
            root = _root_of(node.obj, aliases)
            if root is not None and root[0] in names and node.attr in NODE_ATTRS:
                node_evidence.add(root[0])
            elif root is not None and node.attr in ("length", "size"):
                container(root)
        elif isinstance(node, Subscript | Slice):
            container(_root_of(node.obj, aliases))
        elif isinstance(node, Star):  # `[...nums]`, `f(*nums)`
            container(_root_of(node.value, aliases))
        elif isinstance(node, New) and node.type.name in _ITERABLE_CONSTRUCTORS:
            for arg in node.args:  # `new Set(nums)`
                container(_root_of(arg, aliases))
        elif isinstance(node, Loop) and node.kind == "for_each" and node.iter is not None:
            container(_root_of(node.iter, aliases))
        elif isinstance(node, Comp):
            for generator in node.generators:
                container(_root_of(generator.iter, aliases))
        elif isinstance(node, Compare) and node.op in ("in", "not in"):
            container(_root_of(node.right, aliases))
            if isinstance(node.right, Name | Subscript) and isinstance(node.left, Str):
                root = _root_of(node.right, aliases)
                if root is not None:
                    string_evidence.add(root[0])
        elif isinstance(node, Call):
            func_expr = node.func
            if isinstance(func_expr, Attribute):
                root = _root_of(func_expr.obj, aliases)
                if root is not None and root[0] in names:
                    if func_expr.attr in STRING_METHODS:
                        string_evidence.add(root[0])
                        container(root)
                    elif func_expr.attr in CONTAINER_METHODS:
                        container(root)
                elif (
                    isinstance(func_expr.obj, Name)
                    and func_expr.obj.id not in names
                    and func_expr.attr in _MODULE_CONSUMERS
                ):  # `copy.deepcopy(grid)`, `heapq.heapify(h)`, `Arrays.sort(a)`
                    if node.args and not (
                        func_expr.attr in ("max", "min", "sum") and len(node.args) >= 2
                    ):  # `Math.max(a, b)` compares values; the first argument is the collection
                        container(_root_of(node.args[0], aliases))
            elif isinstance(func_expr, Name) and func_expr.id in CONTAINER_CONSUMERS:
                if func_expr.id in ("max", "min") and len(node.args) >= 2:
                    continue  # `max(a, b)` compares two values; only `max(xs)` iterates
                for arg in node.args:
                    container(_root_of(arg, aliases))
        if isinstance(node, Compare) and node.op in ("==", "!="):
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if isinstance(other, Str) and isinstance(side, Subscript):
                    root = _root_of(side, aliases)
                    if root is not None:
                        string_evidence.add(root[0])

    if callees:
        active = _active | {func.name}
        _callee_evidence(func, callees, active, aliases, names, container, node_evidence)
    infos: dict[str, RoleInfo] = {}
    for name in names:
        if name in node_evidence:
            infos[name] = RoleInfo("node")
        elif name in container_depth:
            kind = "str" if name in string_evidence else "list"
            infos[name] = RoleInfo("container", max(1, container_depth[name]), kind)
        else:
            infos[name] = RoleInfo("int")
    return infos


def _callee_evidence(  # noqa: PLR0913
    func: FuncDef,
    callees: Mapping[str, FuncDef],
    active: frozenset[str],
    aliases: dict[str, tuple[str, int]],
    names: list[str],
    container: Callable[[tuple[str, int] | None, int], None],
    node_evidence: set[str],
) -> None:
    for node in walk(func):
        if not isinstance(node, Call):
            continue
        callee_expr = node.func
        if isinstance(callee_expr, Name):
            target = callees.get(callee_expr.id)
            method_call = False
        elif isinstance(callee_expr, Attribute):
            target = callees.get(callee_expr.attr)
            method_call = True
        else:
            continue
        if target is None or target.name in active:
            continue
        formals = [p.name for p in target.params]
        if method_call and formals and formals[0] in ("self", "cls", "this"):
            formals = formals[1:]
        callee_roles = infer_roles(target, callees, active)
        for formal, arg in zip(formals, node.args, strict=False):
            root = _root_of(arg, aliases)
            info = callee_roles.get(formal)
            if root is None or root[0] not in names or info is None:
                continue
            if info.role == "container":
                container((root[0], root[1] + info.depth - 1), 1)
            elif info.role == "node":
                node_evidence.add(root[0])


# -------------------------------------------------------------------------------------- values
def _container_kind(name: str) -> str | None:
    if name in DICT_TYPES:
        return "dict"
    if name in TREEMAP_TYPES:
        return "treemap"
    if name in SET_TYPES:
        return "set"
    if name in TREESET_TYPES:
        return "treeset"
    if name in HEAP_TYPES:
        return "heap"
    if name in DEQUE_TYPES:
        return "deque"
    if name in STACK_TYPES:
        return "stack"
    if name in LIST_TYPES:
        return "list"
    if name in STRING_TYPES:
        return "str"
    return None


def _element_value(args: tuple[TypeRef, ...], index: int, name: str) -> Value:
    if len(args) <= index:
        return SCALAR
    inner = value_for_type(args[index], f"{name}[]")
    return inner if isinstance(inner, ContV | NodeV) else SCALAR


def value_for_type(t: TypeRef, name: str) -> Value:
    """The abstract value of a parameter of declared type `t`, with fresh size variables."""
    base = t.name
    if t.dims > 0:
        if base == "char" and t.dims == 1:
            return make_container("str", Poly.var(new_var(name, "len")))
        inner: Value = SCALAR
        if t.dims > 1:
            inner = value_for_type(TypeRef(base, t.args, t.dims - 1), f"{name}[]")
        elif base not in INT_TYPES and base not in SCALAR_TYPES and base[:1].isupper():
            inner = NodeV(Poly.var(new_var(f"{name}[]", "len")))
        return make_container(
            "array",
            _row_length(name),
            inner,
            ragged=isinstance(inner, ContV) and graph_like(name),
        )
    kind = _container_kind(base)
    if kind is not None:
        length = _row_length(name)
        if kind in ("dict", "treemap"):
            return make_container(kind, length, _element_value(t.args, 1, name))
        if kind == "str":
            return make_container("str", length)
        element = _element_value(t.args, 0, name)
        return make_container(
            kind, length, element, ragged=isinstance(element, ContV) and graph_like(name)
        )
    if base in INT_TYPES:
        return IntV(Poly.var(new_var(name, "val")))
    if base in SCALAR_TYPES:
        return SCALAR
    if base[:1].isupper() or base in ("struct", "node", "tree"):
        return NodeV(Poly.var(new_var(name, "len")))
    return UNKNOWN


def _value_for_role(info: RoleInfo, name: str) -> Value:
    if info.role == "node":
        return NodeV(Poly.var(new_var(name, "len")))
    if info.role == "container":
        value: Value | None = None
        for level in range(info.depth, 0, -1):
            label = name if level == 1 else f"{name}[{level - 1}]"
            kind = info.kind if level == info.depth else "list"
            value = make_container(
                kind,
                _row_length(label),
                value,
                ragged=isinstance(value, ContV) and graph_like(name),
            )
        assert value is not None
        return value
    return IntV(Poly.var(new_var(name, "val")))


def params_to_values(
    func: FuncDef, language: str, callees: Mapping[str, FuncDef] | None = None
) -> dict[str, Value]:
    """The initial environment of a function: one abstract value per parameter."""
    inferred = infer_roles(func, callees)
    values: dict[str, Value] = {}
    for param in func.params:
        declared = param.type
        if declared is not None and declared.name not in ("auto", "var", "any", "Any", "object"):
            value = value_for_type(declared, param.name)
            if value is not UNKNOWN:
                values[param.name] = value
                continue
        values[param.name] = _value_for_role(inferred[param.name], param.name)
    return values
