"""Library call costs: what `x in list`, `s.sort()`, `heappush`, `list.pop(0)` really cost.

Hidden library cost was the shipped model's single largest failure bucket (43% of its time
errors). The conventions here are the standard worst-case ones used by editorials:

  * hash-map / hash-set operations are O(1) average; ordered (tree) containers are O(log n);
  * `x in list`, `list.index/count/remove`, `list.insert(i, ..)` and `list.pop(i)` are O(n);
    a deque is O(1) at both ends;
  * a sort is O(n log n); a heap push/pop is O(log n) and `heapify` is O(n);
  * copies, slices of lists/strings, `split`, `join` cost their length; a Go slice is a free view;
  * auxiliary space of an in-place sort differs by language (Timsort in Python/JavaScript needs
    O(n); introsort / dual-pivot quicksort / pdqsort in C++/Java/Go need O(log n)).

`call_library` takes the already-evaluated receiver and arguments and returns the cost, the
result value, any memory allocated and any growth applied to the receiver, or None if the call is
not in the table (the interpreter then assumes O(1) and records the assumption).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction

from analysis.nodes import Attribute, Expr, Name, Star, Subscript
from analysis.poly import Poly
from analysis.values import (
    SCALAR,
    ContV,
    FuncV,
    IntV,
    NodeV,
    TupleV,
    UnknownV,
    Value,
    join,
    make_container,
    mem,
)

ONE = Poly.const(1)
ZERO = Poly.zero()

SEQ = frozenset({"list", "array"})
STR = frozenset({"str"})
HASH = frozenset({"dict", "set"})
TREE = frozenset({"treemap", "treeset"})
DEQ = frozenset({"deque", "queue"})
STACK = frozenset({"stack"})
HEAP = frozenset({"heap"})


@dataclass
class LibCall:
    lang: str
    name: str
    qualifier: str | None
    receiver: Value | None
    args: list[Value]
    arg_exprs: list[Expr]
    kwargs: dict[str, Value]
    default_size: Poly
    input_size: Poly | None = None
    receiver_expr: Expr | None = None

    def subject(self) -> ContV | None:
        if isinstance(self.receiver, ContV):
            return self.receiver
        if self.args and isinstance(self.args[0], ContV):
            return self.args[0]
        return None

    def rest(self) -> list[Value]:
        """Arguments after the subject container (the whole list for a method call)."""
        if isinstance(self.receiver, ContV):
            return list(self.args)
        if self.args and isinstance(self.args[0], ContV):
            return list(self.args[1:])
        return list(self.args)

    def kind(self) -> str | None:
        subject = self.subject()
        return subject.kind if subject is not None else None

    def length(self) -> Poly:
        subject = self.subject()
        return subject.length if subject is not None else self.default_size

    def input(self) -> Poly:
        return self.input_size if self.input_size is not None else self.default_size


@dataclass
class LibResult:
    value: Value
    time: Poly = ONE
    alloc: Poly | None = None
    grow: Poly | None = None
    binds_args: dict[int, Value] = field(default_factory=dict)
    assumed: str | None = None
    set_receiver: ContV | None = None  # the receiver after the call (its element shape changed)
    reset: bool = False  # the receiver was emptied


Handler = Callable[[LibCall], LibResult | None]
_REGISTRY: dict[str, list[Handler]] = defaultdict(list)


def lib(*names: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        for name in names:
            _REGISTRY[name].append(fn)
        return fn

    return register


def call_library(c: LibCall) -> LibResult | None:
    for handler in _REGISTRY.get(c.name, ()):
        result = handler(c)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------------- helpers
def _fresh(kind: str, length: Poly, elem: Value | None = None) -> ContV:
    return make_container(kind, length, elem, owned=True)


def _int(c: LibCall, index: int) -> Poly | None:
    rest = c.rest()
    if index < len(rest) and isinstance(rest[index], IntV):
        return rest[index].mag  # type: ignore[union-attr]
    return None


def _const(poly: Poly | None) -> Fraction | None:
    return poly.const_value() if poly is not None else None


def _scalar_or_elem(c: LibCall) -> Value:
    subject = c.subject()
    if subject is not None and subject.elem is not None:
        return subject.elem
    return SCALAR


def _arg_mem(c: LibCall) -> Poly:
    rest = c.rest()
    return mem(rest[-1]) if rest else ONE


# ------------------------------------------------------------------------------------ sizes
@lib("len", "size", "length", "cap", "Len", "strlen", "count_elements")
def _length(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        if c.name == "length" and not c.args and c.receiver is None:
            return None
        if c.name == "strlen":  # no string known: a scan of an input-sized one
            return LibResult(IntV(c.default_size), c.default_size)
        return LibResult(IntV(None), ONE)
    if c.name == "strlen":  # C strings carry no length: strlen walks to the terminator
        return LibResult(IntV(subject.length, subject.uid), subject.length)
    return LibResult(IntV(subject.length, subject.uid), ONE)


@lib("empty", "isEmpty", "is_empty", "hasNext", "isdigit", "isalpha", "isalnum", "isupper",
     "islower", "isspace", "isLetter", "isDigit", "isLetterOrDigit", "isUpperCase",
     "isLowerCase", "isWhitespace", "isnan", "isinf")  # fmt: skip
def _predicate(c: LibCall) -> LibResult | None:
    return LibResult(SCALAR, ONE)


# ------------------------------------------------------------------------------ membership
@lib("contains", "includes", "has", "containsKey", "count", "find", "indexOf", "index",
     "lastIndexOf", "containsValue", "rfind", "findIndex", "includes_key", "Contains",
     "Index", "ContainsRune", "ContainsAny", "contains_key")  # fmt: skip
def _membership(c: LibCall) -> LibResult | None:
    kind = c.kind()
    if kind is None:
        return None
    rest = c.rest()
    if kind in HASH:
        if c.name == "containsValue":
            return LibResult(SCALAR, c.length())
        subject = c.subject()
        if c.name == "find" and subject is not None:  # C++: an iterator into the container
            return LibResult(
                make_container(subject.kind, subject.length, subject.elem, view=True), ONE
            )
        return LibResult(SCALAR, ONE)
    if kind in TREE:
        return LibResult(SCALAR, c.length().log())
    if kind in STR:
        argument = rest[0] if rest else None
        # searching a short literal in a string is still a scan of the string
        _ = argument
        return LibResult(IntV(None), c.length())
    if kind in SEQ or kind in DEQ or kind in HEAP or kind in STACK:
        if c.name == "count" and not rest:
            return None
        return LibResult(IntV(None), c.length())
    return None


@lib("startswith", "endswith", "startsWith", "endsWith", "HasPrefix", "HasSuffix", "equals",
     "equalsIgnoreCase", "compareTo", "compare", "strcmp", "localeCompare")  # fmt: skip
def _prefix_test(c: LibCall) -> LibResult | None:
    rest = c.rest() or c.args
    longest = Poly.const(1)
    for value in rest:
        if isinstance(value, ContV):
            longest = value.length
    return LibResult(SCALAR, longest)


# ------------------------------------------------------------------------ positional access
@lib("get", "at", "charAt", "elementAt", "getFirst", "getLast", "charCodeAt", "codePointAt",
     "getOrDefault", "getOrElse", "getAt")  # fmt: skip
def _get(c: LibCall) -> LibResult | None:
    kind = c.kind()
    if kind is None:
        return None
    if kind in SEQ or kind in STR:
        return LibResult(_scalar_or_elem(c), ONE)
    if kind in HASH:
        return LibResult(_scalar_or_elem(c), ONE)
    if kind in TREE:
        return LibResult(_scalar_or_elem(c), c.length().log())
    if kind in DEQ:
        return LibResult(_scalar_or_elem(c), c.length() if c.name in ("get", "at") else ONE)
    return LibResult(_scalar_or_elem(c), ONE)


# ----------------------------------------------------------------------------- appending
@lib("append", "push", "push_back", "emplace_back", "addLast", "offerLast", "enqueue",
     "add", "offer", "emplace", "put", "set", "insert", "putIfAbsent", "computeIfAbsent",
     "merge", "compute", "setdefault", "Push", "Add", "Set", "push_front", "appendleft",
     "addFirst", "offerFirst", "unshift", "WriteByte", "WriteRune", "WriteString",
     "appendChild", "put_if_absent")  # fmt: skip
def _insert_like(c: LibCall) -> LibResult | None:
    if c.lang == "go" and c.qualifier == "heap":
        return None  # container/heap: log n per push, handled by `_go_heap`
    result = _insert_cost(c)
    subject = c.subject()
    if result is None or subject is None:
        return result
    rest = c.rest()
    added = rest[-1] if rest else None
    amount = ONE
    if c.arg_exprs and isinstance(c.arg_exprs[-1], Star) and isinstance(added, ContV):
        # `append(a, b...)` / `a.push(...b)` stores every element of b
        amount = added.length
        result = replace(result, time=amount, grow=amount)
    elif (
        c.lang == "cpp"
        and isinstance(added, ContV)
        and c.arg_exprs
        and isinstance(c.arg_exprs[-1], Name | Subscript | Attribute)
        and subject.kind in SEQ | DEQ | STACK
    ):
        # C++ copies an lvalue it is given (`words.push_back(cur)` copies the whole string)
        result = replace(result, time=(result.time + mem(added)).order())
    stores_rows = subject.kind in SEQ | DEQ | STACK or (
        subject.kind in ("dict", "treemap") and c.name in ("setdefault", "putIfAbsent")
    )
    if (
        stores_rows
        and isinstance(added, ContV)
        and (subject.elem is None or isinstance(subject.elem, ContV))
    ):
        elem = added if subject.elem is None else join(subject.elem, added)
        # the container now holds this kind of row; rows added empty are grown later, one by one
        subject = replace(subject, elem=elem, ragged=subject.ragged or added.length.is_zero())
        result = replace(result, set_receiver=subject)
    if c.name == "append" and c.receiver is None:
        if subject.cap is None and not subject.view:
            # appending to a fresh slice (`append([]int(nil), xs...)`) is the grown copy
            subject = replace(subject, length=(subject.length + amount).order(), owned=True)
            result = replace(result, alloc=amount)
        return replace(result, value=subject)  # Go: `s = append(s, x)` returns the grown slice
    return result


def _insert_cost(c: LibCall) -> LibResult | None:
    kind = c.kind()
    if kind is None:
        return None
    if c.name in ("set", "list", "tuple") and not isinstance(c.receiver, ContV):
        return None  # `set(xs)` is a conversion (see _copy), not `JSMap.set(k, v)`
    rest = c.rest()
    amount = ONE  # one slot: a stored container is a reference (counted where it was built)
    front = c.name in ("push_front", "appendleft", "addFirst", "offerFirst", "unshift")
    indexed = c.name in ("insert", "add") and len(rest) >= 2 and isinstance(rest[0], IntV)
    position = rest[0] if c.name == "insert" and len(rest) >= 2 else None
    if isinstance(position, ContV) and position.view and kind in SEQ:
        # v.insert(it, x) shifts everything after the iterator: begin() is O(n), end() is O(1)
        shifted = position.length.order() if not position.length.is_zero() else ONE
        return LibResult(SCALAR, shifted, grow=amount)
    if kind in HEAP:
        return LibResult(SCALAR, c.length().log(), grow=amount)
    if kind in TREE:
        return LibResult(SCALAR, c.length().log(), grow=amount)
    if kind in HASH:
        if c.name in ("put", "insert", "add", "set", "emplace") and len(rest) >= 1:
            return LibResult(SCALAR, ONE, grow=amount)
        return LibResult(SCALAR, ONE, grow=amount)
    if kind in SEQ and c.name == "set" and len(rest) >= 2:
        return LibResult(SCALAR, ONE)  # list.set(i, x): overwrite, no growth
    if kind in SEQ and (front or indexed or (c.name == "insert" and len(rest) >= 2)):
        index = _const(_int(c, 0))
        cost = (
            ONE
            if (index is not None and index != 0 and not front and c.name != "insert")
            else c.length()
        )
        return LibResult(SCALAR, cost, grow=amount)
    if kind in STR:
        return LibResult(SCALAR, ONE, grow=amount)
    return LibResult(SCALAR, ONE, grow=amount)


@lib("extend", "addAll", "putAll", "update", "append_all", "extendleft", "concat_in_place")
def _extend(c: LibCall) -> LibResult | None:
    if c.subject() is None:
        return None
    rest = c.rest()
    added = rest[0] if rest else None
    length = added.length if isinstance(added, ContV) else c.default_size
    return LibResult(SCALAR, length, grow=length)


# -------------------------------------------------------------------------------- removing
@lib("pop", "pop_back", "removeLast", "pollLast", "popLast", "Pop", "pop_front", "popleft",
     "pollFirst", "removeFirst", "dequeue", "shift", "poll", "remove", "erase", "delete",
     "discard", "popitem", "Delete", "poll_first")  # fmt: skip
def _remove_like(c: LibCall) -> LibResult | None:
    if c.lang == "go" and c.qualifier == "heap":
        return None  # container/heap: log n per pop, handled by `_go_heap`
    kind = c.kind()
    if kind is None:
        return None
    rest = c.rest()
    L = c.length()
    name = c.name
    from_front = name in (
        "pop_front",
        "popleft",
        "pollFirst",
        "removeFirst",
        "dequeue",
        "shift",
        "poll_first",
    )
    if kind in HEAP:
        return LibResult(
            _scalar_or_elem(c),
            L.log() if name in ("pop", "poll", "dequeue", "Pop", "remove") and not rest else L,
        )
    if kind in TREE:
        return LibResult(_scalar_or_elem(c), L.log())
    if kind in HASH:
        return LibResult(_scalar_or_elem(c), ONE)
    if kind in DEQ or kind in STACK:
        if name in ("remove", "erase", "delete") and rest:
            return LibResult(_scalar_or_elem(c), L)
        return LibResult(_scalar_or_elem(c), ONE)
    if kind in SEQ or kind in STR:
        if from_front:
            return LibResult(_scalar_or_elem(c), L)
        if name in ("remove", "erase", "delete", "discard"):
            return LibResult(_scalar_or_elem(c), L)
        if name == "pop" and rest:
            index = _const(_int(c, 0))
            if index is not None and index < 0:
                return LibResult(_scalar_or_elem(c), ONE)
            return LibResult(_scalar_or_elem(c), L)
        return LibResult(_scalar_or_elem(c), ONE)
    return None


@lib("clear", "reset", "Reset", "Clear")
def _clear(c: LibCall) -> LibResult | None:
    return LibResult(SCALAR, ONE, reset=True) if c.subject() is not None else None


@lib("setLength")
def _set_length(c: LibCall) -> LibResult | None:
    """sb.setLength(0) empties a StringBuilder; any other length just resizes it."""
    if c.subject() is None:
        return None
    rest = c.rest()
    first = rest[0] if rest else None
    emptied = isinstance(first, IntV) and first.mag is not None and first.mag.is_zero()
    return LibResult(SCALAR, ONE, reset=emptied)


# -------------------------------------------------------------------------------- peeking
@lib("top", "peek", "front", "back", "first", "last", "peekFirst", "peekLast", "element",
     "getMin", "getMax", "firstKey", "lastKey", "firstEntry", "lastEntry", "peek_front",
     "peek_back", "Front", "Back", "Peek", "Top")  # fmt: skip
def _peek(c: LibCall) -> LibResult | None:
    kind = c.kind()
    if kind is None:
        return None
    if kind in TREE:
        return LibResult(_scalar_or_elem(c), c.length().log())
    return LibResult(_scalar_or_elem(c), ONE)


_END_ITERATORS = frozenset({"end", "cend", "rend", "crend"})


@lib("begin", "end", "rbegin", "rend", "cbegin", "cend", "crend", "iterator", "listIterator",
     "descendingIterator", "stream", "values", "keySet", "entrySet", "keys", "items",
     "entries", "asList", "boxed", "mapToInt", "chars", "iter", "lazy")  # fmt: skip
def _view(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return None
    if c.name in _END_ITERATORS and not c.args:
        # an iterator is the number of elements from it to the end: end() has none left
        return LibResult(make_container(subject.kind, ZERO, subject.elem, view=True), ONE)
    if c.qualifier == "Object" and c.lang == "javascript":
        return LibResult(
            _fresh("list", subject.length, subject.elem), subject.length, alloc=subject.length
        )
    if c.name in ("keys", "keySet") and subject.kind == "dict":
        return LibResult(make_container("list", subject.length, SCALAR, view=True), ONE)
    if c.name == "values" and subject.kind in ("dict", "treemap") and subject.elem is not None:
        # the rows of a dict of lists, in one pass: they are the dict's own storage (same identity),
        # so the entries they hold add up to the entries the dict was filled with
        rows = isinstance(subject.elem, ContV)
        return LibResult(
            make_container(
                "list", subject.length, subject.elem, view=True, uid=subject.uid, ragged=rows
            ),
            ONE,
        )
    return LibResult(make_container(subject.kind, subject.length, subject.elem, view=True), ONE)


# -------------------------------------------------------------------------------- copying
@lib("copy", "clone", "toArray", "toList", "copyOf", "Array.from", "from",
     "deepcopy", "list", "tuple", "set", "frozenset", "dict", "deque", "Counter", "OrderedDict",
     "defaultdict", "sorted_copy", "toCharArray", "split_chars", "of", "values_copy",
     "ToSlice", "Collect", "collect")  # fmt: skip
def _copy(c: LibCall) -> LibResult | None:
    subject = c.subject()
    name = c.name
    if name == "copy" and c.lang == "go":
        # Go builtin copy(dst, src): O(len(src)) work, allocates nothing
        rest = c.args[1] if len(c.args) > 1 else None
        length = rest.length if isinstance(rest, ContV) else c.default_size
        return LibResult(IntV(length), length)
    if subject is None:
        if name in ("list", "tuple", "set", "frozenset", "dict", "deque", "Counter"):
            kind = {"list": "list", "tuple": "list", "set": "set", "frozenset": "set",
                    "dict": "dict", "deque": "deque", "Counter": "dict"}[name]  # fmt: skip
            return LibResult(_fresh(kind, ONE), ONE, alloc=ONE)
        if name == "defaultdict":
            # defaultdict(list): every missing key starts an empty list of its own
            factory = c.arg_exprs[0] if c.arg_exprs else None
            kinds = {"list": "list", "set": "set", "deque": "deque", "dict": "dict"}
            row_kind = kinds.get(factory.id) if isinstance(factory, Name) else None
            row = _fresh(row_kind, ZERO) if row_kind is not None else None
            return LibResult(_fresh("dict", ZERO, row), ONE, alloc=ONE)
        return None
    kind = {"list": "list", "tuple": "list", "set": "set", "frozenset": "set", "dict": "dict",
            "deque": "deque", "Counter": "dict", "OrderedDict": "dict",
            "defaultdict": "dict"}.get(name, subject.kind)  # fmt: skip
    if name in ("set", "frozenset") and subject.kind in ("dict",):
        kind = "set"
    length = subject.length
    if name == "toCharArray":
        return LibResult(_fresh("array", length), length, alloc=length)
    if name == "deepcopy":  # every nested element is copied too
        return LibResult(_fresh(kind, length, subject.elem), mem(subject), alloc=mem(subject))
    # a shallow copy duplicates the slots (row references stay shared)
    return LibResult(_fresh(kind, length, subject.elem), length, alloc=length)


# -------------------------------------------------------------------------------- sorting
_IN_PLACE_SORT_SPACE_LOG = {
    # (language, qualifier or None) -> sorts with an O(log n) auxiliary stack
    ("cpp", "std"), ("cpp", None), ("c", None), ("go", "sort"), ("go", "slices"),
    ("java", "Arrays"),
}  # fmt: skip


def _sort_space(c: LibCall, length: Poly) -> Poly:
    name = c.name
    if (
        c.lang == "java"
        and c.qualifier == "Arrays"
        and not any(isinstance(a, FuncV) for a in c.args)
    ):
        subject = c.subject()
        if subject is not None and not isinstance(subject.elem, ContV | NodeV):
            return length.log()  # dual-pivot quicksort on a primitive array
        return length  # an array of arrays / objects uses TimSort
    if name == "stable_sort":
        return length
    if (c.lang, c.qualifier) in _IN_PLACE_SORT_SPACE_LOG:
        return length.log()
    if c.lang == "go" and c.qualifier in ("sort", "slices"):
        return length.log()
    return length  # TimSort: Python list.sort, JavaScript, Java Collections.sort


@lib("sort", "stable_sort", "Sort", "Ints", "Strings", "Float64s", "Slice", "SliceStable",
     "qsort", "sort_unstable", "sort_by", "sort_by_key", "SortFunc", "SortStableFunc",
     "sort_heap")  # fmt: skip
def _sort(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if c.name == "qsort":
        n = _int(c, 0)
        length = n if n is not None else c.length()
    elif subject is None:
        if c.name == "sort" and c.receiver is None and not c.args:
            return None
        length = c.default_size
    else:
        length = subject.length
    time = (length * length.log()).order() if not length.is_zero() else ONE
    return LibResult(subject if subject is not None else SCALAR, time, alloc=_sort_space(c, length))


@lib("sorted", "nlargest", "nsmallest", "sorted_by")
def _sorted_copy(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return None
    length = subject.length
    if c.name in ("nlargest", "nsmallest"):
        return LibResult(_fresh("list", length, subject.elem), length * length.log(), alloc=length)
    return LibResult(
        _fresh("list", length, subject.elem), (length * length.log()).order(), alloc=length
    )


@lib("reverse", "Reverse", "reverse_in_place", "shuffle", "Shuffle", "rotate", "fill", "Fill",
     "iota", "memset", "unique", "next_permutation", "prev_permutation", "swap_ranges",
     "partial_sort", "nth_element", "random_shuffle", "replace_all")  # fmt: skip
def _in_place_pass(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return LibResult(SCALAR, ONE) if c.name in ("fill", "memset") else None
    return LibResult(subject, subject.length)


@lib("malloc", "calloc", "realloc", "alloca", "reallocarray", "aligned_alloc")
def _c_allocate(c: LibCall) -> LibResult | None:
    """C heap allocation. `sizeof` lowers to 1, so a byte count reads as an element count."""
    sizes = c.args[1:] if c.name in ("realloc", "reallocarray") else c.args
    first = sizes[0] if sizes else None
    length = first.mag if isinstance(first, IntV) and first.mag is not None else c.default_size
    if c.name in ("calloc", "reallocarray") and len(sizes) > 1:
        second = sizes[1]
        if isinstance(second, IntV) and second.mag is not None:  # calloc(count, size)
            length = (length * second.mag).order()
    return LibResult(_fresh("array", length, SCALAR), length, alloc=length)


# ------------------------------------------------------------------------------ reductions
@lib("sum", "max", "min", "any", "all", "accumulate", "max_element", "min_element", "Max",
     "Min", "reduce", "fold", "product", "prod", "average", "mean", "median", "statistics",
     "getAsInt", "frequency", "Frequency", "Sum", "count_if", "all_of", "any_of", "none_of",
     "equal", "mismatch", "lexicographical_compare", "is_sorted", "adjacent_find")  # fmt: skip
def _reduce(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is not None and (
        c.name not in ("max", "min", "Max", "Min")
        or len(c.rest()) == 0
        or isinstance(c.receiver, ContV)
    ):
        value = _scalar_or_elem(c)
        return LibResult(IntV(None) if value is SCALAR else value, subject.length)
    if c.name in ("max", "min", "Max", "Min") and len(c.args) >= 2:
        return _scalar_extreme(c)
    if c.name in ("max", "min") and not c.args:
        return LibResult(SCALAR, ONE)
    return None


def _scalar_extreme(c: LibCall) -> LibResult:
    mags = [a.mag for a in c.args if isinstance(a, IntV)]
    if len(mags) != len(c.args) or any(m is None for m in mags):
        return LibResult(IntV(None), ONE)
    polys: list[Poly] = [m for m in mags if m is not None]
    if c.name in ("max", "Max"):
        best = polys[0]
        for other in polys[1:]:
            best = best.max(other)
        return LibResult(IntV(best), ONE)
    # an upper bound for min(a, b): the smaller-order argument when one dominates, else the first
    best = polys[0]
    for other in polys[1:]:
        if (best + other).order() == best.order() and (best + other).order() != other.order():
            best = other
    return LibResult(IntV(best), ONE)


# ---------------------------------------------------------------------------------- heaps
@lib("heappush", "heappop", "heapreplace", "heappushpop", "heapify", "make_heap", "push_heap",
     "pop_heap", "Init", "Fix", "Remove", "heap_push", "heap_pop")  # fmt: skip
def _heap_module(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if (
        c.lang == "go"
        and c.qualifier == "heap"
        and c.name in ("Push", "Pop", "Init", "Fix", "Remove")
    ):
        return None  # handled by the generic Go handler below
    if subject is None:
        return None
    L = subject.length
    if c.name in ("heapify", "make_heap", "Init"):
        return LibResult(SCALAR, L)
    grow = _arg_mem(c) if c.name in ("heappush", "push_heap") else None
    return LibResult(_scalar_or_elem(c), L.log(), grow=grow)


@lib("Push", "Pop", "Init", "Fix", "Remove")
def _go_heap(c: LibCall) -> LibResult | None:
    if not (c.lang == "go" and c.qualifier == "heap"):
        return None
    subject = c.subject()
    length = subject.length if subject is not None else c.default_size
    assumed = (
        None if subject is not None else "heap size unknown; assumed proportional to the input"
    )
    if c.name == "Init":
        return LibResult(SCALAR, length, assumed=assumed)
    grow = _arg_mem(c) if c.name == "Push" else None
    return LibResult(SCALAR, length.log(), grow=grow, assumed=assumed)


# ------------------------------------------------------------------------- binary search
@lib("bisect_left", "bisect_right", "bisect", "lower_bound", "upper_bound", "binary_search",
     "binarySearch", "SearchInts", "SearchStrings", "SearchFloat64s", "Search", "equal_range",
     "partition_point", "searchsorted")  # fmt: skip
def _binary_search(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if c.name == "Search" and subject is None:
        first = c.args[0] if c.args else None
        n = first.mag if isinstance(first, IntV) else None
        return LibResult(IntV(None), (n or c.default_size).log())
    if subject is None:
        return None
    return LibResult(IntV(None), subject.length.log())


@lib("insort", "insort_left", "insort_right")
def _insort(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return None
    return LibResult(SCALAR, subject.length, grow=ONE)


# -------------------------------------------------------------------------------- strings
@lib("split", "Split", "Fields", "splitlines", "SplitN", "splitWhitespace", "rsplit",
     "partition", "tokenize")  # fmt: skip
def _split(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None or subject.kind != "str":
        return None
    return LibResult(
        _fresh("list", subject.length, make_container("str", ONE, owned=True)),
        subject.length,
        alloc=subject.length,
    )


@lib("join", "Join", "concat", "merge_lists")
def _join(c: LibCall) -> LibResult | None:
    parts: ContV | None = None
    for value in c.rest() if isinstance(c.receiver, ContV) else c.args:
        if isinstance(value, ContV):
            parts = value
    if c.name == "concat":
        subject = c.subject()
        if subject is None:
            return None
        total = subject.length + (parts.length if parts is not None else Poly.const(1))
        return LibResult(_fresh(subject.kind, total.order(), subject.elem), total.order(),
                         alloc=total.order())  # fmt: skip
    if parts is None:
        return None
    return LibResult(_fresh("str", parts.length), parts.length, alloc=parts.length)


@lib("replace", "strip", "lstrip", "rstrip", "lower", "upper", "toLowerCase", "toUpperCase",
     "trim", "trimStart", "trimEnd", "title", "capitalize", "swapcase", "casefold", "zfill",
     "ljust", "rjust", "center", "format", "ToLower", "ToUpper", "TrimSpace", "Replace",
     "ReplaceAll", "Trim", "TrimLeft", "TrimRight", "replaceAll", "padStart", "padEnd",
     "encode", "decode", "translate", "expandtabs", "normalize")  # fmt: skip
def _string_transform(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None or subject.kind != "str":
        return None
    return LibResult(_fresh("str", subject.length), subject.length, alloc=subject.length)


@lib("substring", "substr", "slice", "Substring", "subSequence", "copyOfRange", "subList",
     "subarray", "sliceRange")  # fmt: skip
def _substring(c: LibCall) -> LibResult | None:
    """A sub-range of a string or array: s.substring(i, j), a.slice(1), Arrays.copyOfRange(a, i,
    j), list.subList(i, j). The length is kept exact (j - i, n - 1) because a recursion on a
    sub-range is measured by how much the range shrinks. A subList is a view: nothing is copied."""
    subject = c.subject()
    if subject is None:
        return None
    if subject.kind not in ("str", "list", "array", "deque"):
        return None
    rest = c.rest()
    low = rest[0].mag if rest and isinstance(rest[0], IntV) else None
    high = rest[1].mag if len(rest) > 1 and isinstance(rest[1], IntV) else None
    length = subject.length
    if c.name == "substr" and low is not None and high is not None:
        length = high  # substr(start, length)
    elif low is not None and high is not None:
        length = high - low
    elif low is not None and len(rest) == 1:
        start = low.const_value()
        length = -low if start is not None and start < 0 else subject.length - low  # slice(-k)
    if not any(coef > 0 for _, coef in length.terms):
        length = ONE
    if c.name == "subList":
        return LibResult(make_container(subject.kind, length, subject.elem, view=True), ONE)
    return LibResult(_fresh(subject.kind, length, subject.elem), length, alloc=length)


@lib("distance")
def _distance(c: LibCall) -> LibResult | None:
    """std::distance(first, last): the elements between two iterators."""
    views = [a for a in c.args if isinstance(a, ContV)]
    if len(views) >= 2:
        return LibResult(IntV(views[0].length - views[1].length), ONE)
    return None


@lib("repeat", "multiply")
def _repeat(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return None
    count = _int(c, 0)
    total = (subject.length * (count if count is not None else ONE)).order()
    return LibResult(_fresh(subject.kind, total), total, alloc=total)


@lib("toString", "String", "str", "String.valueOf", "valueOf", "to_string", "Itoa", "string",
     "tostring", "repr", "StringBuilder", "StringBuffer", "to_str")  # fmt: skip
def _stringify(c: LibCall) -> LibResult | None:
    arg = c.args[0] if c.args else None
    if (
        isinstance(arg, IntV)
        and arg.mag is not None
        and any(v.kind == "val" for v in arg.mag.vars())
    ):
        # the text of a number has as many characters as it has digits: log of its value. (Only a
        # parameter counts as a size; converting a loop index is a machine-word operation.)
        digits = arg.mag.log()
        return LibResult(_fresh("str", digits), digits, alloc=digits)
    subject = c.subject()
    if subject is not None and subject.kind != "str":
        return LibResult(_fresh("str", subject.length), subject.length, alloc=subject.length)
    if subject is not None:
        return LibResult(_fresh("str", subject.length), subject.length, alloc=subject.length)
    return LibResult(make_container("str", ONE, owned=True), ONE)


# ------------------------------------------------------------------------ lazy iterators
@lib("reversed", "enumerate", "zip", "map", "filter", "iter", "Array.keys", "islice",
     "chain", "permutations", "combinations", "product_iter", "cycle", "takewhile",
     "dropwhile", "accumulate_iter", "groupby", "pairwise", "batched")  # fmt: skip
def _lazy(c: LibCall) -> LibResult | None:
    containers = [a for a in c.args if isinstance(a, ContV)]
    if (
        c.name in ("map", "filter")
        and containers
        and c.lang == "javascript"
        and isinstance(c.receiver, ContV)
    ):
        subject = c.receiver
        return LibResult(
            _fresh("list", subject.length, subject.elem), subject.length, alloc=subject.length
        )
    if not containers:
        return None
    if c.name == "zip":
        shortest = containers[0].length
        for other in containers[1:]:
            if (other.length + shortest).order() == shortest.order():
                shortest = other.length
        return LibResult(make_container("list", shortest, SCALAR, view=True), ONE)
    longest = containers[-1] if c.name in ("map", "filter") else containers[0]
    return LibResult(make_container("list", longest.length, longest.elem, view=True), ONE)


@lib("forEach", "some", "every", "find_js", "findLast", "flatMap", "flat", "from_entries")
def _js_array_pass(c: LibCall) -> LibResult | None:
    subject = c.subject()
    if subject is None:
        return None
    if c.name in ("flatMap", "flat"):
        return LibResult(
            _fresh("list", subject.length, subject.elem), subject.length, alloc=subject.length
        )
    return LibResult(SCALAR, subject.length)


# ---------------------------------------------------------------------------- scalar maths
_PRESERVE = frozenset(
    {"int", "abs", "floor", "ceil", "round", "trunc", "float", "long", "double", "int64",
     "Abs", "Floor", "Ceil", "Round", "Trunc", "fabs", "valueOf_int", "Integer", "Number",
     "float64", "int32", "uint", "unsigned", "float32", "parseFloat", "negate", "ord", "chr",
     "toInt", "toLong", "intValue", "longValue", "doubleValue", "Float64", "Int", "Int64",
     "bit_length_of"}
)  # fmt: skip


@lib(*sorted(_PRESERVE))
def _preserve(c: LibCall) -> LibResult | None:
    arg = c.args[0] if c.args else c.receiver
    if isinstance(arg, ContV) and arg.kind == "str":
        return LibResult(IntV(arg.length), ONE)  # int(input()): an integer as large as the input
    if isinstance(arg, IntV):
        return LibResult(IntV(arg.mag, arg.of if c.name in ("int", "long") else None), ONE)
    return LibResult(IntV(None) if c.name in ("int", "long", "abs", "ord") else SCALAR, ONE)


@lib("sqrt", "isqrt", "cbrt", "Sqrt", "sqrtf", "Cbrt")
def _sqrt(c: LibCall) -> LibResult | None:
    arg = c.args[0] if c.args else None
    if isinstance(arg, IntV) and arg.mag is not None:
        return LibResult(IntV(arg.mag.sqrt().order()), ONE)
    return LibResult(IntV(None), ONE)


@lib("log", "log2", "log10", "Log", "Log2", "Log10", "logf", "log1p", "bit_length", "clz",
     "numberOfLeadingZeros", "highestOneBit", "ilog2", "ilog")  # fmt: skip
def _log(c: LibCall) -> LibResult | None:
    arg = c.args[0] if c.args else c.receiver
    if isinstance(arg, IntV) and arg.mag is not None:
        return LibResult(IntV(arg.mag.log().order()), ONE)
    return LibResult(IntV(None), ONE)


@lib("pow", "Pow", "powf", "power", "exp", "Exp", "expf")
def _pow(c: LibCall) -> LibResult | None:
    if len(c.args) >= 2 and isinstance(c.args[0], IntV) and isinstance(c.args[1], IntV):
        base, exponent = c.args[0].mag, c.args[1].mag
        k = _const(exponent)
        if base is not None and k is not None:
            return LibResult(IntV(base.power(k).order()), ONE)
    return LibResult(IntV(None), ONE)


@lib("gcd", "lcm", "__gcd", "Gcd", "hypot", "sin", "cos", "tan", "atan", "atan2", "asin", "acos",
     "sinh", "cosh", "tanh", "radians", "degrees", "signum", "sign", "random", "rand",
     "nextInt_bounded",
     "randint", "uniform", "Random", "randomInt", "Rand", "Intn", "rint", "isfinite", "copysign",
     "fmod", "mod", "modf", "frexp", "ldexp", "comb", "perm", "popcount", "popCount", "bitCount",
     "__builtin_popcount", "__builtin_clz", "__builtin_ctz", "TrailingZeros", "OnesCount",
     "swap", "Swap", "hash", "id", "type", "isinstance", "callable", "print", "println",
     "printf", "puts", "putchar", "cout", "log_info", "Println", "Printf", "Print", "Sprintf",
     "Fprintf", "Fprintln", "write", "flush", "assert", "assert_eq", "panic", "Errorf", "error",
     "console_log", "toFixed", "toPrecision", "isNaN", "isFinite", "isInteger", "Boolean",
     "bool", "is_integer", "compareTo_int", "exit", "free", "close", "Close", "Sleep", "sleep",
     "time", "now", "clock", "Now", "currentTimeMillis", "nanoTime", "NewReader", "NewWriter",
     "NewScanner", "Buffer", "bufio", "Flush", "strconv", "setdefault_scalar", "isalpha_c",
     "toupper", "tolower", "isspace_c", "atof", "atol", "stoi", "stol", "stoll", "stod", "stoul",
     "parseInt", "Atoi", "Integer_parseInt", "parseLong", "parseDouble", "Parse", "ParseInt",
     "ParseFloat", "Float", "Double", "decode_int", "make_pair", "make_tuple", "pair", "tie",
     "move", "forward", "Math", "max_int", "numeric_limits", "min_int", "ref", "cref",
     "static_cast", "reinterpret_cast", "dynamic_cast", "sizeof", "alignof", "typeof")  # fmt: skip
def _o1_scalar(c: LibCall) -> LibResult | None:
    if c.name in ("parseInt", "Atoi", "stoi", "stol", "stoll", "stoul", "atol", "parseLong",
                  "Parse", "ParseInt"):  # fmt: skip
        arg = c.args[0] if c.args else None
        if isinstance(arg, ContV) and arg.kind == "str":
            if c.name == "Atoi":
                return LibResult(TupleV((IntV(arg.length), SCALAR)), ONE)
            return LibResult(IntV(arg.length), ONE)
        return LibResult(IntV(None), ONE)
    if c.name in ("swap", "Swap") and c.args:
        return LibResult(SCALAR, ONE)
    if c.name in ("make_pair", "make_tuple", "pair", "tie"):
        return LibResult(TupleV(tuple(c.args)) if c.args else SCALAR, ONE)
    return LibResult(SCALAR, ONE)


# ------------------------------------------------------------------------------ program input
_INPUT_STR = frozenset(
    {"input", "raw_input", "readline", "readLine", "nextLine", "next", "readFileSync", "read",
     "getline", "gets", "fgets", "Text", "ReadString", "ReadLine", "prompt", "readlines",
     "getchar_line", "stdin_read", "next_token", "Bytes", "ReadAll"}
)  # fmt: skip
_INPUT_INT = frozenset(
    {"nextInt", "nextLong", "nextDouble", "nextFloat", "nextBigInteger", "readInt", "readLong",
     "nextShort", "nextByte", "getchar", "readNumber", "read_int", "Int"}
)  # fmt: skip
_INPUT_SCAN = frozenset({"Fscan", "Scan", "Fscanln", "Scanln", "Fscanf", "Scanf", "Sscan", "Sscanf",
                         "scanf", "fscanf", "sscanf", "scan"})  # fmt: skip


@lib(*sorted(_INPUT_STR))
def _read_str(c: LibCall) -> LibResult | None:
    if c.name == "next" and c.subject() is not None:
        return None  # an iterator's next(), not reading input
    if c.name == "read" and c.subject() is not None:
        return None
    size = c.input()
    value = make_container("str", size, owned=True)
    if c.name == "readlines":
        return LibResult(
            _fresh("list", size, make_container("str", ONE, owned=True)), size, alloc=size
        )
    if c.name in ("read", "readFileSync", "ReadAll", "Bytes", "stdin_read"):
        return LibResult(value, ONE, alloc=size)  # the whole input is materialised
    # one line: it only costs memory if the program keeps it (binding it to a name counts it),
    # `int(input())` never holds more than a number
    return LibResult(value, ONE)


@lib(*sorted(_INPUT_INT))
def _read_int(c: LibCall) -> LibResult | None:
    if c.name == "getchar":
        return LibResult(SCALAR, ONE)
    return LibResult(IntV(c.input()), ONE)


@lib(*sorted(_INPUT_SCAN))
def _scan(c: LibCall) -> LibResult | None:
    first = (
        1
        if c.name.startswith(("F", "S", "f", "s"))
        and c.name not in ("Scan", "Scanln", "scanf", "scan")
        else 0
    )
    binds: dict[int, Value] = {i: IntV(c.input()) for i in range(first, len(c.args))}
    if c.name in ("scanf", "scan"):
        binds = {i: IntV(c.input()) for i in range(1 if c.name == "scanf" else 0, len(c.args))}
    return LibResult(SCALAR, ONE, binds_args=binds)


@lib("range", "xrange", "Range", "arange")
def _range_value(c: LibCall) -> LibResult | None:
    mags = [a.mag for a in c.args if isinstance(a, IntV)]
    if not mags or any(m is None for m in mags):
        return LibResult(make_container("list", c.default_size, SCALAR, view=True), ONE)
    polys = [m for m in mags if m is not None]
    if len(polys) == 1:
        length = polys[0]
    else:
        length = (polys[1] - polys[0]).order()
    return LibResult(
        make_container(
            "list", length.order(), IntV(polys[-1] if len(polys) > 1 else polys[0]), view=True
        ),
        ONE,
    )


@lib("charAt_unused", "new_node", "nullptr_check", "NodeV_unused")
def _never(c: LibCall) -> LibResult | None:
    _ = (NodeV, UnknownV)
    return None
