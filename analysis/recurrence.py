"""Recursion: turn the recursive calls recorded while interpreting a function into a closed form.

The interpreter analyses a recursive function once, with every parameter a fresh symbolic size,
and records each call it makes to itself instead of costing it: the arguments (as sizes), how many
times the call runs per invocation (enclosing loop counts) and the comparisons that held on the way
to it. This module reads those records as a recurrence

    T(size) = work(size) + sum over calls of  T(smaller size)

and solves it, in the order a person would:

  1. The recursion is a LOOP whose test is the path condition of the call and whose body replaces
     the parameters by the arguments. The same potential argument that bounds a `while` loop then
     gives the depth: a quantity (the summed distance between compared values) that falls by a
     constant (subtractive recursion, depth = size / step) or by a constant factor (divisive
     recursion, depth = log size).
  2. One call per level: time is depth * work. Several calls per level subtractively: a ** depth
     (a calls that each take a constant off the size). Several calls dividing the size: the master
     theorem, a = number of calls, b = the factor, work = n^d (log n)^e. A loop of calls whose
     arguments do not depend on the loop variable gives a factorial; one whose arguments do
     (`for i in range(start, n): f(i + 1)`) gives 2 ** depth.
  3. Recursion on a structure (a tree, a linked list) visits every node once: size * work.
  4. Recursion guarded by a visited / memo container that it also writes executes its body at most
     once per entry of the container: domain * work, shared by every call that reaches it.

Every path through the function is a recurrence of its own and the answer is the worst of them.
Anything that cannot be proven this way is reported as unsolved, and the engine says "unknown"
rather than guess: a recursion that grows its argument on one branch (collatz) is never answered
with the cost of the branch that halves it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

from analysis.bundle import Bundle, Cond, RecCall
from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    Call,
    Compare,
    Const,
    Expr,
    FuncDef,
    If,
    ListLit,
    Name,
    Num,
    Subscript,
    walk,
)
from analysis.poly import Mono, Poly, Var, new_var
from analysis.values import ContV, IntV, NodeV, Value, mem

if TYPE_CHECKING:
    from analysis.interp import FuncSummary, Interp

ONE = Poly.const(1)
ZERO = Poly.zero()
_MARKING_CALLS = frozenset(
    {"add", "put", "insert", "append", "push", "set", "setdefault", "update", "emplace", "offer"}
)
_REMOVING_CALLS = frozenset({"remove", "discard", "pop", "delete", "clear", "erase", "popitem"})
_GUARD_CALLS = frozenset(
    {"contains", "containsKey", "has", "get", "count", "find", "includes", "indexOf", "exists"}
)


@dataclass
class Chain:
    """How the size shrinks along the chain of recursive calls."""

    kind: str  # "arith": falls by a constant step;  "geom": falls by a constant factor
    depth: Poly  # how many nested calls
    measure: Poly  # the size at entry (order level)
    step: Fraction = Fraction(1)  # arith: amount taken off per call
    ratio: Fraction = Fraction(1, 2)  # geom: new size = ratio * size
    # positions the size is measured FROM (`lo` in hi - lo): the cost of a call depends on how far
    # the range extends, not on where it starts, so they count as 0 in the work
    offsets: frozenset[Var] = frozenset()


# ======================================================================== updates and chains
def _updates(call: RecCall, params: dict[str, Value]) -> dict[Var, Poly]:
    """How each size variable of the function changes from one call to the next."""
    mapping: dict[Var, Poly] = {}
    for (_, formal), actual in zip(params.items(), call.args, strict=False):
        _bind(formal, actual, mapping)
    return mapping


def _fresh(name: str) -> Poly:
    return Poly.var(new_var(name, "free"))


def _bind(formal: Value, actual: object | None, mapping: dict[Var, Poly]) -> None:
    if isinstance(formal, IntV) and formal.mag is not None:
        var = _only_var(formal.mag)
        if var is not None:
            mag = actual.mag if isinstance(actual, IntV) else None
            mapping[var] = mag if mag is not None else _fresh(var.name)
    elif isinstance(formal, ContV):
        var = _only_var(formal.length)
        if var is not None:
            mapping[var] = actual.length if isinstance(actual, ContV) else _fresh(var.name)
        if isinstance(formal.elem, ContV):
            _bind(formal.elem, actual.elem if isinstance(actual, ContV) else None, mapping)
    elif isinstance(formal, NodeV):
        var = _only_var(formal.size)
        if var is not None:
            mapping[var] = actual.size if isinstance(actual, NodeV) else _fresh("n")


def _only_var(poly: Poly) -> Var | None:
    if len(poly.terms) != 1:
        return None
    mono, coef = poly.terms[0]
    if coef != 1 or len(mono.pows) != 1 or mono.exps or mono.facts:
        return None
    var, power, log = mono.pows[0]
    return var if power == 1 and log == 0 else None


def _potentials(conds: tuple[Cond, ...]) -> list[Poly]:
    """Candidate measures from the path condition: the summed distance between the compared
    values. A `!=` has no direction, so both are offered."""
    base = ZERO
    undirected: list[Poly] = []
    for op, left, right in conds:
        if op in ("<", "<="):
            base = base + (right - left)
        elif op in (">", ">="):
            base = base + (left - right)
        elif op == "!=":
            undirected.append(left - right)
    options = [base]
    for diff in undirected:
        options = [o + diff for o in options] + [o - diff for o in options]
    return [o for o in options if not o.is_zero()]


def _falls(phi: Poly, updates: dict[Var, Poly]) -> Chain | None:
    """Does the quantity `phi` fall by a constant, or by a constant factor, when the parameters are
    replaced by the arguments of a call?"""
    from analysis.bounds import _geometric_ratio

    phi_next = phi.substitute(updates)
    measure = phi.order() if not phi.is_zero() else ONE
    step = (phi - phi_next).const_value()
    if step is not None and step > 0:
        return Chain("arith", (measure / step).order(), measure, step=step, offsets=_offsets(phi))
    ratio = _geometric_ratio(phi, phi_next)
    if ratio is not None:
        return Chain("geom", measure.log(), measure, ratio=ratio, offsets=_offsets(phi))
    return None


def _chain(call: RecCall, params: dict[str, Value]) -> Chain | None:
    """The potential argument for one recursive call: does a quantity fall on every call?

    A `!=` test has no direction (`lo != hi` may mean either is larger), so both orientations are
    tried and, when both fall, the size is their union: |hi - lo| is at most lo + hi."""
    updates = _updates(call, params)
    found = [c for phi in _potentials(call.conds) if (c := _falls(phi, updates)) is not None]
    if not found:
        return _euclid(call, params)
    first = found[0]
    if len(found) == 1 or any(c.kind != first.kind for c in found):
        return first
    measure = _widest(c.measure for c in found)
    offsets = frozenset.intersection(*(c.offsets for c in found))
    if first.kind == "arith":
        step = min(c.step for c in found)
        return Chain("arith", (measure / step).order(), measure, step=step, offsets=offsets)
    ratio = max(c.ratio for c in found)
    return Chain("geom", measure.log(), measure, ratio=ratio, offsets=offsets)


def _offsets(phi: Poly) -> frozenset[Var]:
    """Variables that only ever lower the size (the `lo` of hi - lo)."""
    positive: set[Var] = set()
    negative: set[Var] = set()
    for mono, coef in phi.terms:
        (positive if coef > 0 else negative).update(mono.vars())
    return frozenset(negative - positive)


def _euclid(call: RecCall, params: dict[str, Value]) -> Chain | None:
    """`gcd(b, a % b)`: the old divisor becomes the dividend and the remainder is smaller than the
    divisor, so the larger value at least halves every two calls: log(b) calls."""
    names = list(params)
    if len(names) < 2 or len(call.arg_exprs) < 2:
        return None
    first, second = call.arg_exprs[0], call.arg_exprs[1]
    a, b = names[0], names[1]
    is_euclid = (
        isinstance(first, Name)
        and first.id == b
        and isinstance(second, BinOp)
        and second.op == "%"
        and isinstance(second.left, Name)
        and second.left.id == a
        and isinstance(second.right, Name)
        and second.right.id == b
    )
    divisor = params[b]
    if is_euclid and isinstance(divisor, IntV) and divisor.mag is not None:
        return Chain("geom", divisor.mag.log(), divisor.mag.order())
    return None


# ========================================================================== the closed forms
def _exp(base: int, exponent: Poly) -> Poly:
    t = new_var("t", "free")
    return Poly.exp(t, base).substitute({t: exponent})


def _fact(of: Poly) -> Poly:
    t = new_var("t", "free")
    return Poly.fact(t).substitute({t: of})


def _log_base(a: Fraction, b: Fraction) -> Fraction:
    """log_b(a) as an exact rational when a is a power of b, else rounded up a little."""
    if a == 1:
        return Fraction(0)
    exact = math.log(float(a)) / math.log(float(b))
    nearest = round(exact)
    if abs(exact - nearest) < 1e-9:
        return Fraction(nearest)
    return Fraction(math.ceil(exact * 1000), 1000)


def _master(a: Fraction, b: Fraction, work: Poly, measure: Poly) -> Poly:
    """T(n) = a T(n / b) + work(n), where n is `measure`.

    `work` is a sum of terms and the recurrence is linear, so each term is solved on its own: a
    term c * n^d (log n)^e times a factor `other` that does not involve n gives other * n^log_b(a)
    when d is below log_b(a), other * n^d (log n)^(e+1) when equal, and itself when above."""
    variables = measure.vars()
    crit = _log_base(a, b)
    answer = ZERO
    for mono, _ in work.terms:
        inside = Mono(
            tuple((v, p, lg) for v, p, lg in mono.pows if v in variables),
        )
        other = Poly._make(
            {
                Mono(
                    tuple((v, p, lg) for v, p, lg in mono.pows if v not in variables),
                    mono.exps,
                    mono.facts,
                ): Fraction(1)
            }
        )
        d = sum((p for _, p, _ in inside.pows), Fraction(0))
        e = sum((lg for _, _, lg in inside.pows), 0)
        if d < crit:
            piece = measure.power(crit)
        elif d == crit:
            piece = measure.power(d) * measure.log().power(e + 1)
        else:
            piece = Poly._make({inside: Fraction(1)})
        term = (piece * other).order()
        answer = term if answer.is_zero() else answer.max(term)
    return answer if not answer.is_zero() else ONE


# ============================================================================== the solver
def solve_recurrence(it: Interp, summary: FuncSummary, body: Bundle) -> None:
    func = summary.func
    alternatives = [
        tuple(c for c in alt if c.callee == func.name)
        for alt in body.rec
        if any(c.callee == func.name for c in alt)
    ]
    if not alternatives:
        summary.solved = False
        return
    every_call = tuple(c for alt in alternatives for c in alt)
    work, frame = summary.time, summary.space

    domain = _domain_recursion(it, summary, every_call)
    if domain is not None:
        if domain.size is None:
            summary.solved = False
            it.note(
                "unknown",
                f"recursion in {func.name}() is memoised through {domain.container}, "
                "whose size could not be bounded",
            )
            return
        _apply_domain(it, summary, domain, work, frame)
        return
    if _structural(summary, every_call):
        _apply_structural(it, summary, work, frame)
        return
    try:
        paths = list(dict.fromkeys(alternatives))
    except TypeError:
        paths = alternatives
    outcomes = [_solve_path(it, summary, path, work, frame) for path in paths]
    if any(outcome is None for outcome in outcomes):
        summary.solved = False
        it.note("unknown", f"recursion in {func.name}() could not be solved")
        return
    time, space = ZERO, ZERO
    for outcome in outcomes:
        assert outcome is not None
        time = outcome[0] if time.is_zero() else time.max(outcome[0])
        space = outcome[1] if space.is_zero() else space.max(outcome[1])
    summary.time, summary.space, summary.solved = time, space, True


def _solve_path(
    it: Interp, summary: FuncSummary, calls: tuple[RecCall, ...], work: Poly, frame: Poly
) -> tuple[Poly, Poly] | None:
    """The time and stack space of the recurrence along ONE path through the function."""
    chains: list[tuple[RecCall, Chain]] = []
    for call in calls:
        chain = _chain(call, summary.params) or _chain_from_loop(it, call, summary.params)
        if chain is None:
            return None
        chains.append((call, chain))
    kinds = {chain.kind for _, chain in chains}
    if len(kinds) != 1:
        return None
    total_calls = ZERO
    for call, _ in chains:
        total_calls = total_calls + call.mult
    offsets = frozenset.intersection(*(c.offsets for _, c in chains))
    if offsets:
        zero = dict.fromkeys(offsets, ZERO)
        work = work.substitute(zero) if not work.substitute(zero).is_zero() else ONE
        frame = frame.substitute(zero)
    depth = _widest(c.depth for _, c in chains)
    measure = _widest(c.measure for _, c in chains)
    a_const = total_calls.const_value()

    if "arith" in kinds:
        step = min(c.step for _, c in chains)
        depth = (measure / step).order()
        if any(_depends_on_loop_index(call) for call, _ in chains):
            time = _exp(2, depth)  # for i in range(start, n): f(i + 1): T(m) = sum of T(k), k < m
        elif a_const is not None and a_const <= 1:
            time = (depth * work).order()
        elif a_const is not None:
            time = _exp(int(math.ceil(a_const)), depth)
        else:  # a loop of calls that each take a constant off: a factorial
            time = _fact(depth)
    else:
        if a_const is None:
            return None
        ratio = max(c.ratio for _, c in chains)
        time = _master(a_const, 1 / ratio, work, measure)
    return (
        time if not time.is_zero() else ONE,
        _stack_space(depth, frame, measure, kinds == {"geom"}),
    )


def _widest(polys: Iterable[Poly]) -> Poly:
    out = ZERO
    for p in polys:
        out = p if out.is_zero() else out.max(p)
    return out if not out.is_zero() else ONE


def _stack_space(depth: Poly, frame: Poly, measure: Poly, divisive: bool) -> Poly:
    """Frames alive at once: depth of them, each holding its own allocations. When the size
    halves, frame allocations that grow with the size form a geometric series: one frame."""
    per_frame = frame if not frame.is_zero() else ONE
    if divisive and any(v in measure.vars() for v in per_frame.vars()):
        return per_frame.order()
    return (depth * per_frame).order()


def _iter_vars(arg: object) -> set[Var]:
    poly = arg.mag if isinstance(arg, IntV) else None
    return {v for v in poly.vars() if v.kind == "iter"} if poly is not None else set()


def _depends_on_loop_index(call: RecCall) -> bool:
    return any(_iter_vars(arg) for arg in call.args)


def _chain_from_loop(it: Interp, call: RecCall, params: dict[str, Value]) -> Chain | None:
    """Recursion inside a loop has no `if` guard: the loop's own trip count is the measure
    (`for i in range(start, n): f(i + 1)` runs while start < n)."""
    if not call.mult.vars():
        return None
    # the arguments depend on the loop index; take its first value (the worst call)
    first: dict[Var, Poly] = {}
    for arg in call.args:
        for v in _iter_vars(arg):
            info = it.iters.get(v)
            first[v] = info.lower if info is not None and info.lower is not None else ZERO
    shifted = {var: poly.substitute(first) for var, poly in _updates(call, params).items()}
    phi = call.mult
    step = (phi - phi.substitute(shifted)).const_value()
    measure = phi.order() if not phi.is_zero() else ONE
    if step is not None and step > 0:
        return Chain("arith", (measure / step).order(), measure, step=step)
    return None


# ===================================================================== structural recursion
def _structural(summary: FuncSummary, calls: tuple[RecCall, ...]) -> bool:
    """Every call passes a node reached from a node parameter and nothing numeric shrinks."""
    if not any(isinstance(v, NodeV) for v in summary.params.values()):
        return False
    for call in calls:
        if not any(isinstance(a, NodeV) for a in call.args):
            return False
        if _chain(call, summary.params) is not None:
            return False  # a number shrinks too: the general rule applies
    return True


def _apply_structural(it: Interp, summary: FuncSummary, work: Poly, frame: Poly) -> None:
    size = _widest(v.size for v in summary.params.values() if isinstance(v, NodeV))
    # the children of all nodes together number size - 1: one per node on average
    work = work.substitute({kids: ONE for kids in it.kids_vars.values()})
    summary.time = (size * work).order()
    summary.space = (size * (frame if not frame.is_zero() else ONE)).order()
    summary.solved = True


# ================================================================ memoised / visited recursion
@dataclass
class Domain:
    container: str  # the container that is marked and guarded ("<cache>" for @lru_cache)
    size: Poly | None  # how many distinct entries it can hold; None: memoised but not sizeable
    shared: bool = False  # a parameter: every call that is handed the same container shares it
    uid: int | None = None  # a captured container (a closure variable): shared by its identity


@dataclass
class Usage:
    """How a function uses the containers it can reach (parameters, captured variables, globals,
    fields): which it writes (marks), which it reads to skip work (guards), which it empties."""

    marked: set[str] = field(default_factory=set)
    guarded: set[str] = field(default_factory=set)
    unmarked: set[str] = field(default_factory=set)
    stored: dict[str, set[str]] = field(default_factory=dict)  # what kinds of value are stored
    keys: dict[str, list[Expr]] = field(default_factory=dict)

    def mark(self, container: str, key: Expr, stored: str | None = None) -> None:
        self.marked.add(container)
        self.keys.setdefault(container, []).append(key)
        if stored is not None:
            self.stored.setdefault(container, set()).add(stored)

    def guard(self, container: str, key: Expr) -> None:
        self.guarded.add(container)
        self.keys.setdefault(container, []).append(key)


def _container_name(expr: Expr) -> str | None:
    """`memo` or `self.memo`: the name of the container an expression indexes into."""
    while isinstance(expr, Subscript):
        expr = expr.obj
    if isinstance(expr, Name):
        return expr.id
    if isinstance(expr, Attribute) and isinstance(expr.obj, Name):
        return f"{expr.obj.id}.{expr.attr}"
    return None


def _locals_of(func: FuncDef) -> set[str]:
    names = {p.name for p in func.params}
    for node in walk(func):
        if isinstance(node, Assign):
            for target in node.targets:
                parts = target.elts if isinstance(target, ListLit) else (target,)
                names.update(p.id for p in parts if isinstance(p, Name))
    return names


def _stored_kind(value: Expr | None) -> str:
    if isinstance(value, Const):
        return value.kind  # "true" | "false" | "none"
    return "other" if isinstance(value, Num) else "computed"


def _outer_index(target: Subscript) -> Expr:
    """`memo[r][c]` is keyed by (r, c): gather every index of the chain."""
    indices: list[Expr] = []
    node: Expr = target
    while isinstance(node, Subscript):
        indices.append(node.index)
        node = node.obj
    return ListLit(tuple(reversed(indices)), "tuple")


def _skips_work(it: Interp, func: FuncDef, test: If) -> bool:
    """An `if` that can skip work already done: one branch returns, or the call is made only in
    one branch. (A comparison of two array elements decides which element to copy, not whether to
    recurse, and is not a guard.)"""
    if it.always_exits(test.body) or it.always_exits(test.orelse):
        return True
    for branch in (test.body, test.orelse):
        for stmt in branch:
            for node in walk(stmt):
                if isinstance(node, Call):
                    callee = node.func
                    name = (
                        callee.id
                        if isinstance(callee, Name)
                        else (callee.attr if isinstance(callee, Attribute) else "")
                    )
                    if name == func.name:
                        return True
    return False


def _looked_up_values(
    func: FuncDef, is_container: Callable[[str | None], bool]
) -> dict[str, tuple[str, Expr]]:
    """Variables assigned from a lookup in a container (`v, ok := memo[n]`, `cached =
    memo.get(n)`), with the container and key: testing them is testing the container."""
    found: dict[str, tuple[str, Expr]] = {}
    for node in walk(func):
        if not (isinstance(node, Assign) and node.value is not None):
            continue
        value = node.value
        container: str | None = None
        key: Expr | None = None
        if isinstance(value, Subscript):
            container, key = _container_name(value), value.index
        elif (
            isinstance(value, Call)
            and isinstance(value.func, Attribute)
            and value.func.attr in _GUARD_CALLS
            and value.args
        ):
            container, key = _container_name(value.func.obj), value.args[0]
        if container is None or not is_container(container) or key is None:
            continue
        for target in node.targets:
            for name in target.elts if isinstance(target, ListLit) else (target,):
                if isinstance(name, Name):
                    found[name.id] = (container, key)
    return found


def _scan(it: Interp, func: FuncDef, is_container: Callable[[str | None], bool]) -> Usage:
    usage = Usage()
    looked_up = _looked_up_values(func, is_container)
    for node in walk(func):
        if isinstance(node, Assign):
            for target in node.targets:
                root = _container_name(target) if isinstance(target, Subscript) else None
                if root is not None and is_container(root):
                    usage.mark(root, _outer_index(target), _stored_kind(node.value))  # type: ignore[arg-type]
        elif isinstance(node, Call) and isinstance(node.func, Attribute):
            root = _container_name(node.func.obj)
            if root is not None and is_container(root):
                if node.func.attr in _MARKING_CALLS and node.args:
                    usage.mark(root, node.args[0])
                if node.func.attr in _REMOVING_CALLS:
                    usage.unmarked.add(root)
        if isinstance(node, If) and _skips_work(it, func, node):
            for sub in walk(node.test):
                if isinstance(sub, Expr):
                    _guard_reads(usage, sub, looked_up, is_container)
    for name, kinds in usage.stored.items():
        if {"true", "false"} <= kinds:  # set and cleared again: backtracking, not memoisation
            usage.unmarked.add(name)
    return usage


def _guard_reads(
    usage: Usage,
    sub: Expr,
    looked_up: dict[str, tuple[str, Expr]],
    is_container: Callable[[str | None], bool],
) -> None:
    if isinstance(sub, Name) and sub.id in looked_up:
        usage.guard(*looked_up[sub.id])
    elif isinstance(sub, Subscript):
        root = _container_name(sub)
        if root is not None and is_container(root):
            usage.guard(root, sub.index)
    elif isinstance(sub, Compare) and sub.op in ("in", "not in"):
        root = _container_name(sub.right)
        if root is not None and is_container(root):
            usage.guard(root, sub.left)
    elif isinstance(sub, Call) and isinstance(sub.func, Attribute) and sub.args:
        root = _container_name(sub.func.obj)
        if root is not None and is_container(root) and sub.func.attr in _GUARD_CALLS:
            usage.guard(root, sub.args[0])


def _domain_recursion(
    it: Interp, summary: FuncSummary, calls: tuple[RecCall, ...]
) -> Domain | None:
    """A container that the function both reads in an `if` (is this entry already done?) and
    writes (it is now done): the body runs at most once per entry of the container. The container
    may be a parameter, a variable of an enclosing function, a global or a field (`self.memo`)."""
    func = summary.func
    params = {n for n, v in summary.params.items() if isinstance(v, ContV)}
    local = _locals_of(func)

    def is_container(name: str | None) -> bool:
        if name is None:
            return False
        return name in params or (name not in local and name not in ("self", "this", "cls"))

    usage = _scan(it, func, is_container)
    for name in sorted((usage.marked & usage.guarded) - usage.unmarked):
        size = _domain_size(summary, calls, name, usage.keys.get(name, []), name in params)
        captured = summary.closure.get(name) if summary.closure is not None else None
        uid = captured.uid if isinstance(captured, ContV) and name not in params else None
        return Domain(name, size, shared=name in params and size is not None, uid=uid)
    if any("cache" in d.lower() or "memo" in d.lower() for d in func.decorators):
        return Domain("<cache>", _cache_size(summary, calls))
    return None


def _cache_size(summary: FuncSummary, calls: tuple[RecCall, ...]) -> Poly | None:
    """The number of distinct argument tuples a memoised function can see: how far each argument
    that changes between calls can travel, multiplied."""
    total = ONE
    for name, formal in summary.params.items():
        if not isinstance(formal, IntV) or formal.mag is None:
            continue
        var = _only_var(formal.mag)
        if var is None:
            continue
        if all(_updates(c, summary.params).get(var) == formal.mag for c in calls):
            continue  # never changes: contributes no distinct states
        travel = _travel(summary, calls, name)
        if travel is None:
            return None
        total = (total * travel).order()
    return total


def _key_params(summary: FuncSummary, key_exprs: list[Expr]) -> list[str]:
    names: list[str] = []
    for expr in key_exprs:
        parts = expr.elts if isinstance(expr, ListLit) else (expr,)
        for part in parts:
            if isinstance(part, ListLit):
                continue
            for node in walk(part):
                if isinstance(node, Name) and node.id in summary.params and node.id not in names:
                    names.append(node.id)
        if names:
            break
    return names


def _domain_size(
    summary: FuncSummary,
    calls: tuple[RecCall, ...],
    name: str,
    key_exprs: list[Expr],
    is_param: bool,
) -> Poly | None:
    """How many distinct entries the recursion can touch: the product, over the parameters that
    key the container, of how far each can travel before the recursion stops; failing that, the
    length of a table those parameters index; failing that, the size of the container itself when
    it is a pre-sized table passed in."""
    keyed = _key_params(summary, key_exprs)
    sizes = [_travel(summary, calls, p) for p in keyed]
    if keyed and all(s is not None for s in sizes):
        total = ONE
        for s in sizes:
            assert s is not None
            total = (total * s).order()
        return total
    indexed = _indexed_by_key(summary, keyed)
    if keyed and indexed is not None:  # seen.add(u) next to graph[u]: one entry per vertex
        two_keys = len(keyed) >= 2 and isinstance(indexed.elem, ContV)
        return mem(indexed).order() if two_keys else indexed.length.order()  # grid[r][c]: cells
    if not is_param:
        return None
    container = summary.params[name]
    if isinstance(container, ContV) and container.kind in ("list", "array", "str"):
        return mem(container).order()
    return None


def _indexed_by_key(summary: FuncSummary, keyed: list[str]) -> ContV | None:
    """A table (a parameter or a captured variable) that the function indexes by a key
    parameter: its length is how many distinct keys there can be."""
    for node in walk(summary.func):
        if isinstance(node, Subscript) and isinstance(node.obj, Name):
            index = node.index
            if isinstance(index, Name) and index.id in keyed:
                holder = summary.params.get(node.obj.id)
                if holder is None and summary.closure is not None:
                    holder = summary.closure.get(node.obj.id)
                if isinstance(holder, ContV) and holder.kind in ("list", "array", "dict"):
                    return holder
    return None


def _travel(summary: FuncSummary, calls: tuple[RecCall, ...], param: str) -> Poly | None:
    """The range one integer parameter covers over the chain of calls, from the path conditions
    that mention it (the same potential argument as for the depth, per parameter)."""
    formal = summary.params[param]
    if not isinstance(formal, IntV) or formal.mag is None:
        return None
    var = _only_var(formal.mag)
    if var is None:
        return None
    best: Poly | None = None
    for call in calls:
        mine = tuple(c for c in call.conds if var in c[1].vars() | c[2].vars())
        updates = _updates(call, summary.params)
        found = [chain.measure for phi in _potentials(mine) if (chain := _falls(phi, updates))]
        if not found:
            return None
        widest = _widest(found)
        best = widest if best is None else best.max(widest)
    return best


def _apply_domain(
    it: Interp, summary: FuncSummary, domain: Domain, work: Poly, frame: Poly
) -> None:
    """T = (entries) * (work of one real execution). Through a container the caller hands in, it
    is paid once however often the container is reused; through any other it is this function."""
    size = domain.size
    assert size is not None
    summed = _sum_rows(it, summary, work, size)
    total = (summed if summed is not None else size * work).order()
    if domain.shared:
        summary.shared[domain.container] = total
        summary.time = ONE
    elif domain.uid is not None:
        summary.shared_free[domain.uid] = total
        summary.time = ONE
    else:
        summary.time = total
    summary.space = (size * (frame if not frame.is_zero() else ONE)).order()
    summary.solved = True


def _sum_rows(it: Interp, summary: FuncSummary, work: Poly, entries: Poly) -> Poly | None:
    """Work that scans a row of a table (`graph[node]`) once per entry sums to the total size of
    the table, not entries * longest row. Returns the total over all entries, or None if the work
    does not scan such a row."""
    tables: list[Value] = list(summary.params.values())
    if summary.closure is not None:
        for node in walk(summary.func):
            if isinstance(node, Name) and node.id not in summary.params:
                captured = summary.closure.get(node.id)
                if captured is not None and captured not in tables:
                    tables.append(captured)
    for value in tables:
        if not (isinstance(value, ContV) and isinstance(value.elem, ContV)):
            continue
        row = _only_var(value.elem.length)
        if row is None or not work.mentions(row):
            continue
        total = Poly.var(it.total_var(value.uid))
        result = ZERO
        for power, rest in work.extract_power(row):
            if power == 0:
                result = result + rest * entries
            elif power == 1:
                result = result + total * rest
            else:
                result = result + total * Poly.var(row).power(power - 1) * rest
        return result
    return None
