"""Loop bounds: how many times does a loop's body run?

The interpreter asks `plan_loop` for a `LoopPlan` before it executes a loop body. A plan says how
many iterations there are, which induction variables the body sees (with their ranges, so an inner
loop bounded by an outer one folds into a closed-form sum), and whether the loop is AMORTISED:
its total iterations across an enclosing loop are bounded by something that is not reset, so its
cost is added once, not once per outer iteration.

Strategy for `while`-style loops, in order (the first rule that proves a bound wins):

  1. WORKLIST DRAIN   `while stack:` / `while queue:` / `while head < len(queue):` -- the number
                      of pops is bounded by the number of pushes; if pushes are guarded by a
                      visited marker they are bounded by the size of the domain.
  2. POTENTIAL        some quantity (the distance between compared values, summed over the
                      conjuncts of the test) strictly decreases on every path of the body, by a
                      constant (linear count) or by a constant factor (logarithmic count). This
                      one argument covers `i += 1`, `lo/hi/mid` binary search, `while n > 1:
                      n //= 2`, two pointers, `i * i <= n` and `p *= 2`.
  3. POINTER CHASE    `while cur:` over a linked structure runs at most its node count.
  4. INDEX POINTER    a pointer that only moves one way and indexes a container cannot move more
                      than that container's length in total (a sliding window's `left`).
  5. otherwise        assume linear in the input and RECORD the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

from analysis.env import Env
from analysis.interp import Interp, IterInfo, assigned_names
from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    Break,
    Call,
    Compare,
    Continue,
    Expr,
    ExprStmt,
    FuncDef,
    If,
    ListLit,
    Loop,
    Name,
    Num,
    Return,
    Stmt,
    Subscript,
    Switch,
    Try,
    UnOp,
    children,
    walk,
)
from analysis.poly import Poly, Var, new_var
from analysis.values import (
    SCALAR,
    UNKNOWN,
    ContV,
    IntV,
    NodeV,
    TupleV,
    Value,
    join,
    magnitude,
)

ONE = Poly.const(1)
ZERO = Poly.zero()
MAX_PATHS = 48
ORDER_OPS = frozenset({"<", "<=", ">", ">=", "!="})
PUSH_NAMES = frozenset(
    {"append", "push", "push_back", "add", "offer", "enqueue", "emplace", "emplace_back",
     "appendleft", "push_front", "addLast", "addFirst", "put", "insert", "extend"}
)  # fmt: skip
POP_NAMES = frozenset(
    {"pop", "popleft", "poll", "pollFirst", "pollLast", "remove", "dequeue", "shift", "front",
     "top", "peek", "back", "pop_front", "pop_back", "removeFirst", "removeLast", "Pop",
     "heappop", "first", "last", "peekFirst", "peekLast"}
)  # fmt: skip
MARK_NAMES = frozenset({"add", "insert", "put", "append", "push", "set", "emplace", "update",
                        "setdefault", "discard", "remove"})  # fmt: skip
EMPTY_TESTS = frozenset({"empty", "isEmpty", "is_empty", "isempty"})
INDEXING_METHODS = frozenset(
    {"get", "at", "Get", "charAt", "charCodeAt", "codePointAt", "elementAt"}
)


@dataclass
class LoopPlan:
    iters: Poly
    atoms: list[tuple[Var, IterInfo]] = field(default_factory=list)
    bind: dict[str, Value] = field(default_factory=dict)
    post: dict[str, Value] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    ragged_uid: int | None = None
    amortize_into: int | None = None
    amortized_total: Poly | None = None
    # progress information, kept so an enclosing loop can amortise this one
    phi: Poly | None = None
    phi_atoms: dict[str, Var] = field(default_factory=dict)
    phi_kind: str = "arith"
    phi_step: Fraction = Fraction(1)
    pinned: set[str] = field(default_factory=set)  # names the body re-assigns but may not widen
    drain: str | None = None  # name of the container this loop drains
    index_pointers: list[tuple[str, ContV]] = field(default_factory=list)


@dataclass
class IterSpace:
    count: Poly
    elem: Value
    atoms: list[tuple[Var, IterInfo]] = field(default_factory=list)
    ragged_uid: int | None = None
    notes: list[str] = field(default_factory=list)


# ================================================================================== entry
def plan_loop(it: Interp, loop: Loop, env: Env, loop_id: int) -> LoopPlan:
    if loop.kind == "for_each":
        return _plan_for_each(it, loop, env)
    if loop.kind == "for_c":
        for stmt in loop.init:
            it.absorb(it.exec_stmt(stmt, env))
        plan = _plan_progress(it, loop.test, (*loop.body, *loop.update), env, loop)
    else:
        plan = _plan_progress(it, loop.test, loop.body, env, loop)
    _maybe_amortize(it, plan, loop, env)
    return plan


# ============================================================================== for-each
def _plan_for_each(it: Interp, loop: Loop, env: Env) -> LoopPlan:
    assert loop.iter is not None and loop.target is not None
    space = iteration_space(it, loop.iter, env)
    plan = LoopPlan(
        iters=space.count,
        atoms=space.atoms,
        ragged_uid=space.ragged_uid,
        notes=space.notes,
    )
    bind_target(loop.target, space.elem, plan.bind)
    return plan


def bind_target(target: Expr, value: Value, out: dict[str, Value]) -> None:
    if isinstance(target, Name):
        out[target.id] = value
    elif isinstance(target, ListLit):
        if isinstance(value, TupleV) and len(value.items) == len(target.elts):
            for sub, item in zip(target.elts, value.items, strict=True):
                bind_target(sub, item, out)
        elif isinstance(value, ContV):
            for sub in target.elts:
                bind_target(sub, value.elem if value.elem is not None else SCALAR, out)
        else:
            for sub in target.elts:
                bind_target(sub, UNKNOWN, out)


def _ordered(poly: Poly) -> Poly:
    return poly.order() if not poly.is_zero() else ONE


def iteration_space(it: Interp, expr: Expr, env: Env) -> IterSpace:
    if isinstance(expr, Call) and isinstance(expr.func, Name):
        name = expr.func.id
        if name in ("range", "xrange") and env.get(name) is None:
            return _range_space(it, expr, env)
        if name == "enumerate" and expr.args:
            return _enumerate_space(it, expr, env)
    if (
        isinstance(expr, Call)
        and isinstance(expr.func, Attribute)
        and expr.func.attr in ("items", "entries", "entrySet", "iteritems")
        and not expr.args
    ):
        obj = it.evaluate(expr.func.obj, env)
        if isinstance(obj, ContV):
            elem = obj.elem if obj.elem is not None else SCALAR
            return IterSpace(_ordered(obj.length), TupleV((SCALAR, elem)))
    value = it.evaluate(expr, env)
    return _space_of_value(it, value)


def _space_of_value(it: Interp, value: Value) -> IterSpace:
    if isinstance(value, ContV):
        if value.kind == "str" or value.kind in ("dict", "treemap", "set", "treeset"):
            elem: Value = SCALAR
            if value.kind in ("set", "treeset") and value.elem is not None:
                elem = value.elem
            return IterSpace(_ordered(value.length), elem)
        elem = value.elem if value.elem is not None else SCALAR
        ragged = None
        if isinstance(elem, ContV) and value.kind in ("list", "array") and it.is_ragged(value):
            ragged = value.uid
            elem = replace(elem, length=Poly.var(it.rowlen_var(value.uid)))
        return IterSpace(_ordered(value.length), elem, ragged_uid=ragged)
    if isinstance(value, IntV) and value.mag is not None:
        atom = new_var("i", "iter")
        info = IterInfo("arith", _ordered(value.mag), value.of)
        return IterSpace(_ordered(value.mag), IntV(Poly.var(atom)), [(atom, info)])
    if isinstance(value, TupleV):
        elem = value.items[0] if value.items else SCALAR
        for item in value.items[1:]:
            elem = join(elem, item)
        return IterSpace(Poly.const(max(1, len(value.items))), elem)
    return IterSpace(
        it.default_size,
        UNKNOWN,
        notes=["iterating a value of unknown size; assumed proportional to the input"],
    )


def _range_space(it: Interp, call: Call, env: Env) -> IterSpace:
    values = [it.evaluate(a, env) for a in call.args]
    mags = [magnitude(v) for v in values]
    notes: list[str] = []
    unknown = [i for i, m in enumerate(mags) if m is None]
    if len(values) in (2, 3) and len(unknown) == 1:
        step_known = mags[2].const_value() if len(mags) == 3 and mags[2] is not None else 1
        lower_end = 1 if (step_known is not None and step_known < 0) else 0
        if unknown[0] == lower_end and (len(values) == 2 or unknown[0] != 2):
            # `range(capacity, weights[i] - 1, -1)`: a floor read from data counts as >= 0, so the
            # loop runs at most as many times as its other endpoint says
            mags[lower_end] = ZERO
            notes.append("range() bound read from data assumed non-negative")
    if not mags or any(m is None for m in mags):
        atom = new_var("i", "iter")
        info = IterInfo("arith", it.default_size)
        return IterSpace(
            it.default_size,
            IntV(Poly.var(atom)),
            [(atom, info)],
            notes=["range() bound of unknown size; assumed proportional to the input"],
        )
    polys = [m for m in mags if m is not None]
    if len(polys) == 1:
        low, high, step = ZERO, polys[0], ONE
    elif len(polys) == 2:
        low, high, step = polys[0], polys[1], ONE
    else:
        low, high, step = polys[0], polys[1], polys[2]
    step_value = step.const_value()
    descending = step_value is not None and step_value < 0
    span = (low - high) if descending else (high - low)
    if step_value is not None and step_value != 0:
        count = span / abs(step_value)
    elif len(step.terms) == 1:
        count = span * step.power(Fraction(-1))
    else:
        count = span
    count = _ordered(count)
    upper = _ordered(low if descending else high)
    domain = values[0].of if len(values) == 1 and isinstance(values[0], IntV) else None
    atom = new_var("i", "iter")
    return IterSpace(
        count, IntV(Poly.var(atom)), [(atom, IterInfo("arith", upper, domain))], notes=notes
    )


def _enumerate_space(it: Interp, call: Call, env: Env) -> IterSpace:
    inner = iteration_space(it, call.args[0], env)
    source = it.pure(call.args[0], env)
    uid = source.uid if isinstance(source, ContV) else None
    atom = new_var("index", "iter")
    info = IterInfo("arith", inner.count, uid)
    return IterSpace(
        inner.count,
        TupleV((IntV(Poly.var(atom)), inner.elem)),
        [*inner.atoms, (atom, info)],
        inner.ragged_uid,
        inner.notes,
    )


# ================================================================================== while
def _flatten_and(test: Expr) -> list[Expr]:
    from analysis.nodes import BoolOp

    if isinstance(test, BoolOp) and test.op == "and":
        out: list[Expr] = []
        for value in test.values:
            out.extend(_flatten_and(value))
        return out
    return [test]


def _guard_test(body: tuple[Stmt, ...]) -> Expr | None:
    """`while True: if cond: break ...` is a loop with condition `not cond`."""
    first = body[0] if body else None
    if (
        isinstance(first, If)
        and not first.orelse
        and first.body
        and isinstance(first.body[-1], Break | Return)
    ):
        return UnOp("not", first.test)
    return None


def _plan_progress(
    it: Interp, test: Expr | None, body: tuple[Stmt, ...], env: Env, loop: Loop
) -> LoopPlan:
    if test is None:
        test = _guard_test(loop.body)
    conjuncts = _flatten_and(test) if test is not None else []
    for finder in (
        _worklist_plan,
        _euclid_plan,
        _arithmetic_plan,
        _pointer_plan,
        _index_pointer_plan,
    ):
        plan = finder(it, conjuncts, body, env)
        if plan is not None:
            return plan
    geometric = _geometric_test_plan(it, conjuncts, body, env)
    if geometric is not None:
        return geometric
    note = (
        "this loop's trip count could not be proven; assumed proportional to the input"
        if test is not None
        else "an unconditional loop with no recognisable exit; assumed proportional to the input"
    )
    return LoopPlan(iters=it.default_size, notes=[note])


# ------------------------------------------------------------------------- name utilities
def _names_in(expr: Expr) -> set[str]:
    return {n.id for n in walk(expr) if isinstance(n, Name)}


def _container_of(it: Interp, expr: Expr, env: Env) -> tuple[str, ContV] | None:
    if isinstance(expr, Name):
        value = env.get(expr.id)
        if isinstance(value, ContV):
            return expr.id, value
    return None


def _push_calls(stmts: tuple[Stmt, ...], name: str) -> bool:
    for stmt in stmts:
        for node in walk(stmt):
            if (
                isinstance(node, Call)
                and isinstance(node.func, Attribute)
                and node.func.attr in PUSH_NAMES
                and isinstance(node.func.obj, Name)
                and node.func.obj.id == name
            ):
                return True
            if (
                isinstance(node, Assign)
                and isinstance(node.value, Call)
                and isinstance(node.value.func, Name)
                and node.value.func.id == "append"
                and node.value.args
                and isinstance(node.value.args[0], Name)
                and node.value.args[0].id == name
            ):
                return True
            if (
                isinstance(node, Call)
                and isinstance(node.func, Name)
                and node.func.id in ("heappush",)
                and node.args
                and isinstance(node.args[0], Name)
                and node.args[0].id == name
            ):
                return True
    return False


def _mutates(stmts: tuple[Stmt, ...], name: str) -> bool:
    for stmt in stmts:
        for node in walk(stmt):
            if isinstance(node, Assign):
                for target in node.targets:
                    root = target
                    while isinstance(root, Subscript):
                        root = root.obj
                    if isinstance(root, Name) and root.id == name and isinstance(target, Subscript):
                        return True
            if (
                isinstance(node, Call)
                and isinstance(node.func, Attribute)
                and node.func.attr in MARK_NAMES
                and isinstance(node.func.obj, Name)
                and node.func.obj.id == name
            ):
                return True
    return False


# ------------------------------------------------------------------------------ worklist
def _drain_container(
    it: Interp, conjuncts: list[Expr], env: Env
) -> tuple[str, ContV, str | None] | None:
    """(container name, value, head-pointer name) when the loop test is "the worklist is not
    empty": `while q`, `while len(q) > 0`, `while not q.empty()`, `while head < len(q)`."""
    for c in conjuncts:
        node = c
        if isinstance(node, UnOp) and node.op == "not":
            inner = node.operand
            if (
                isinstance(inner, Call)
                and isinstance(inner.func, Attribute)
                and inner.func.attr in EMPTY_TESTS
            ):
                found = _container_of(it, inner.func.obj, env)
                if found:
                    return found[0], found[1], None
            continue
        if isinstance(node, Name):
            found = _container_of(it, node, env)
            if found and found[1].kind in ("list", "array", "deque", "queue", "stack", "heap"):
                return found[0], found[1], None
        measured = _nonempty_measure(node)
        if measured is not None:  # `len(q) > 0` / `q.length != 0` / `q.size() >= 1` / `0 < len(q)`
            length_of = _length_target(measured)
            found = _container_of(it, length_of, env) if length_of is not None else None
            if found and found[1].kind in ("list", "array", "deque", "queue", "stack", "heap"):
                return found[0], found[1], None
        if isinstance(node, Compare) and node.op == "<" and isinstance(node.left, Name):
            # `head < len(q)`
            length_of = _length_target(node.right)
            found = _container_of(it, length_of, env) if length_of is not None else None
            if found and isinstance(env.get(node.left.id), IntV):
                return found[0], found[1], node.left.id
    return None


def _nonempty_measure(test: Expr) -> Expr | None:
    """`x` when `test` says `x > 0`, `x != 0`, `x >= 1`, `0 < x`, `0 != x` or `1 <= x`."""
    if not isinstance(test, Compare):
        return None
    left, right, op = test.left, test.right, test.op
    if isinstance(left, Num) and not isinstance(right, Num):  # `0 < x`  ==  `x > 0`
        flipped = {"<": ">", "<=": ">=", "!=": "!="}.get(op)
        if flipped is None:
            return None
        left, right, op = right, left, flipped
    if not isinstance(right, Num) or right.value is None:
        return None
    if (op in (">", "!=") and right.value == 0) or (op == ">=" and right.value == 1):
        return left
    return None


def _length_target(expr: Expr) -> Expr | None:
    if (
        isinstance(expr, Call)
        and isinstance(expr.func, Name)
        and expr.func.id == "len"
        and expr.args
    ):
        return expr.args[0]
    if isinstance(expr, Call) and isinstance(expr.func, Attribute) and expr.func.attr in (
        "size", "length",
    ):  # fmt: skip
        return expr.func.obj
    if isinstance(expr, Attribute) and expr.attr in ("length", "size"):
        return expr.obj
    return None


def _popped_names(body: tuple[Stmt, ...], container: str, head: str | None) -> list[str]:
    """Names assigned from a pop/peek/index of the worklist near the top of the body."""
    names: list[str] = []
    for stmt in body[:4]:
        if not isinstance(stmt, Assign) or stmt.value is None:
            continue
        pops = False
        for node in walk(stmt.value):
            if (
                isinstance(node, Call)
                and isinstance(node.func, Attribute)
                and node.func.attr in POP_NAMES
                and isinstance(node.func.obj, Name)
                and node.func.obj.id == container
            ):
                pops = True
            if (
                isinstance(node, Subscript)
                and isinstance(node.obj, Name)
                and node.obj.id == container
            ):
                pops = True
            if (
                isinstance(node, Call)
                and isinstance(node.func, Name)
                and node.func.id in ("heappop", "pop")
                and node.args
                and isinstance(node.args[0], Name)
                and node.args[0].id == container
            ):
                pops = True
        if not pops:
            continue
        for target in stmt.targets:
            if isinstance(target, Name):
                names.append(target.id)
            elif isinstance(target, ListLit):
                names.extend(t.id for t in target.elts if isinstance(t, Name))
    return names


def _guard_container(
    it: Interp, body: tuple[Stmt, ...], env: Env, worklist: str
) -> tuple[str, ContV] | None:
    """A container V tested in an `if` that guards a push onto the worklist, and modified in the
    loop: a visited marker (a set, a boolean array, a distance table, an in-degree counter)."""
    for stmt in body:
        for node in walk(stmt):
            if not isinstance(node, If) or not _push_calls(node.body, worklist):
                continue
            for name in _names_in(node.test):
                if name == worklist:
                    continue
                value = env.get(name)
                if isinstance(value, ContV) and _mutates(body, name):
                    return name, value
    return None


def _index_access(node: object) -> tuple[str, Expr] | None:
    """(container name, index) for `c[i]` and for `c.get(i)` / `c.at(i)`."""
    if isinstance(node, Subscript) and isinstance(node.obj, Name):
        return node.obj.id, node.index
    if (
        isinstance(node, Call)
        and isinstance(node.func, Attribute)
        and node.func.attr in INDEXING_METHODS
        and isinstance(node.func.obj, Name)
        and len(node.args) == 1
    ):
        return node.func.obj.id, node.args[0]
    return None


def _worklist_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    drained = _drain_container(it, conjuncts, env)
    if drained is None:
        return None
    name, value, head = drained
    pushes = _push_calls(body, name)
    plan = LoopPlan(iters=_ordered(value.length), drain=name)
    if not pushes:
        # `while q:` drains; `i < len(a)` over a container nobody pushes to is a counted loop,
        # and only the arithmetic planner binds `i` so an inner `j < i` sums to a triangle
        return plan if head is None else None
    guard = _guard_container(it, body, env, name)
    if guard is None:
        return None  # pushes with no visited marker: no bound
    gname, gvalue = guard
    popped = _popped_names(body, name, head)
    indexed: ContV | None = None
    for stmt in body:
        for node in walk(stmt):
            access = _index_access(node)
            if access is not None and access[0] not in (name, gname):
                if isinstance(access[1], Name) and access[1].id in popped:
                    target = env.get(access[0])
                    if isinstance(target, ContV):
                        indexed = target
    concrete = (
        gvalue.kind in ("list", "array")
        and gvalue.cap is not None
        and not it.cap_initial.get(gvalue.cap, ZERO).is_zero()
    )
    if indexed is not None:
        domain = indexed.length
        uid: int | None = indexed.uid
        it.ragged_uids.add(indexed.uid)  # a traversal that reads `g[node]` walks an adjacency list
    elif concrete:
        from analysis.values import mem

        domain = mem(gvalue)
        uid = None
    else:
        domain = it.default_size
        uid = None
        plan.notes.append("visited-domain size unknown; assumed proportional to the input")
    initial = it.cap_initial.get(value.cap, ZERO) if value.cap is not None else ZERO
    plan.iters = _ordered(domain + initial)
    if value.cap is not None:
        it.cap_override[value.cap] = _ordered(domain + initial)
    if gvalue.cap is not None and gvalue.kind in ("set", "dict", "treeset", "treemap"):
        it.cap_override[gvalue.cap] = _ordered(domain + it.cap_initial.get(gvalue.cap, ZERO))
    if popped:
        atom = new_var("node", "iter")
        info = IterInfo("arith", _ordered(domain), uid)
        plan.atoms.append((atom, info))
        for pname in popped:
            plan.bind[pname] = IntV(Poly.var(atom))
        plan.pinned = set(popped)
        plan.ragged_uid = uid
    return plan


# ------------------------------------------------------------------------------- tracing
class _TooComplex(Exception):
    pass


def trace_paths(it: Interp, stmts: tuple[Stmt, ...], env: Env) -> list[tuple[Env, str]] | None:
    """Run the loop body symbolically once; one (end-state, status) per control-flow path.
    Status is "go" (fell off the end), "cont" (continue) or "exit" (return / break)."""
    try:
        with it.scratch():
            return _trace(it, stmts, [(env.copy(), "go")])
    except _TooComplex:
        return None


def _trace(
    it: Interp, stmts: tuple[Stmt, ...], states: list[tuple[Env, str]]
) -> list[tuple[Env, str]]:
    for stmt in stmts:
        nxt: list[tuple[Env, str]] = []
        for e, status in states:
            if status != "go":
                nxt.append((e, status))
            else:
                nxt.extend(_trace_stmt(it, stmt, e))
        states = nxt
        if len(states) > MAX_PATHS:
            raise _TooComplex
    return states


def _trace_stmt(it: Interp, st: Stmt, env: Env) -> list[tuple[Env, str]]:  # noqa: C901
    if isinstance(st, Assign):
        it.exec_assign(st, env)
        return [(env, "go")]
    if isinstance(st, ExprStmt):
        it.evaluate(st.expr, env)
        return [(env, "go")]
    if isinstance(st, Return | Break):
        return [(env, "exit")]
    if isinstance(st, Continue):
        return [(env, "cont")]
    if isinstance(st, If):
        it.evaluate(st.test, env)
        then = _trace(it, st.body, [(env.copy(), "go")])
        other = _trace(it, st.orelse, [(env.copy(), "go")])
        return then + other
    if isinstance(st, Switch):
        out: list[tuple[Env, str]] = []
        for arm in st.arms:
            out.extend(_trace(it, arm, [(env.copy(), "go")]))
        return out or [(env, "go")]
    if isinstance(st, Try):
        return _trace(it, st.body, [(env, "go")])
    if isinstance(st, Loop):
        _havoc_loop(it, st, env)
        return [(env, "go")]
    if isinstance(st, FuncDef):
        return [(env, "go")]
    return [(env, "go")]


def _additive_sign(loop: Loop, name: str) -> int | None:
    """+1 / -1 if every update of `name` inside `loop` is `name += c` / `name -= c` (c > 0)."""
    sign: int | None = None
    for stmt in (*loop.body, *loop.update):
        for node in walk(stmt):
            if isinstance(node, Assign):
                for target in node.targets:
                    names = (
                        [target]
                        if isinstance(target, Name)
                        else (list(target.elts) if isinstance(target, ListLit) else [])
                    )
                    if any(isinstance(n, Name) and n.id == name for n in names):
                        if node.op not in ("+=", "-=") or not isinstance(node.value, Num):
                            return None
                        s = 1 if node.op == "+=" else -1
                        if sign is not None and sign != s:
                            return None
                        sign = s
    return sign


def _havoc_loop(it: Interp, loop: Loop, env: Env) -> None:
    """A loop nested in the body being traced. If every update of a variable inside it is monotone
    (`v += c` or `v -= c`), the variable ends "at or beyond" where it started, modelled as
    `v +/- w` for a fresh non-negative `w`; any other update makes the variable unknown."""
    for name in assigned_names((*loop.body, *loop.update)):
        value = env.get(name)
        if not isinstance(value, IntV) or value.mag is None:
            continue
        sign = _additive_sign(loop, name)
        extra = new_var(f"{name}'", "free")
        if sign is not None:
            it.monotone[extra] = sign
            env.set(name, IntV(value.mag + Poly.var(extra) * sign))
        else:
            env.set(name, IntV(Poly.var(extra)))


# ======================================================================= geometric, any test
def _geometric_test_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    """`while n % 2 == 0: n //= 2`. The test is not an order comparison, but it mentions one
    variable and that variable shrinks by a constant factor on every path, so the loop cannot run
    more than log(n) times (for a positive start; a start of 0 would never leave the loop at all,
    and that is a bug rather than a cost). A variable that only shrinks by a constant step proves
    nothing here: it would walk through zero and the negatives, so only the geometric case counts.
    """
    if not conjuncts:
        return None
    mentioned: set[str] = set()
    for c in conjuncts:
        mentioned |= {n.id for n in walk(c) if isinstance(n, Name)}
    carried = assigned_names(body)
    candidates = [n for n in mentioned if n in carried and isinstance(env.get(n), IntV)]
    if len(candidates) != 1 or any(n in carried for n in mentioned - set(candidates)):
        return None  # another variable the test depends on also changes: no single measure
    name = candidates[0]
    synthetic: list[Expr] = [Compare(">", Name(name), Num(0))]
    plan = _arithmetic_plan(it, synthetic, body, env)
    if plan is None or not plan.atoms or plan.atoms[0][1].kind != "geom":
        return None
    plan.phi, plan.phi_atoms = None, {}
    return plan


# ================================================================================== Euclid
def _straight_line_definitions(body: tuple[Stmt, ...]) -> dict[str, Expr] | None:
    """Final value of every variable a straight-line body assigns, as an expression over the
    values the variables held on entry. None if the body is anything but assignments over names,
    numbers and arithmetic."""
    defs: dict[str, Expr] = {}

    def subst(expr: Expr) -> Expr | None:
        if isinstance(expr, Name):
            return defs.get(expr.id, expr)
        if isinstance(expr, Num):
            return expr
        if isinstance(expr, BinOp):
            left, right = subst(expr.left), subst(expr.right)
            if left is None or right is None:
                return None
            return BinOp(expr.op, left, right)
        return None

    for stmt in body:
        if not isinstance(stmt, Assign) or stmt.value is None or len(stmt.targets) != 1:
            return None
        target, value = stmt.targets[0], stmt.value
        if (
            isinstance(target, ListLit)
            and isinstance(value, ListLit)
            and len(target.elts) == len(value.elts)
        ):
            pairs = list(zip(target.elts, value.elts, strict=True))
        else:
            pairs = [(target, value)]
        resolved: list[tuple[str, Expr]] = []
        for lhs, rhs in pairs:
            if not isinstance(lhs, Name):
                return None
            new = subst(rhs if stmt.op == "=" else BinOp(stmt.op[:-1], lhs, rhs))
            if new is None:
                return None
            resolved.append((lhs.id, new))
        for name, new in resolved:  # a tuple assignment reads every right-hand side first
            defs[name] = new
    return defs


def _tested_zero(conjuncts: list[Expr]) -> str | None:
    """The variable a loop test compares with zero: `b`, `b != 0`, `b > 0`."""
    for c in conjuncts:
        if isinstance(c, Name):
            return c.id
        if isinstance(c, Compare) and c.op in ("!=", ">"):
            if isinstance(c.left, Name) and isinstance(c.right, Num) and c.right.value == 0:
                return c.left.id
    return None


def _euclid_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    """`while b != 0: a, b = b, a % b`. The remainder is smaller than the divisor and the old
    divisor becomes the next dividend, so the larger value at least halves every two steps and the
    loop runs O(log b) times. Without the swap (`b = a % b` alone) no such bound holds."""
    b = _tested_zero(conjuncts)
    defs = _straight_line_definitions(body) if b is not None else None
    if b is None or defs is None:
        return None
    new_b = defs.get(b)
    if not (
        isinstance(new_b, BinOp)
        and new_b.op == "%"
        and isinstance(new_b.left, Name)
        and isinstance(new_b.right, Name)
        and new_b.right.id == b
    ):
        return None
    a = new_b.left.id
    if a == b or defs.get(a) != Name(b):
        return None
    start = env.get(b)
    if not isinstance(start, IntV) or start.mag is None:
        return None
    return LoopPlan(iters=_ordered(start.mag.log()))


# ============================================================================== arithmetic
@dataclass
class _Path:
    env: Env
    status: str


def _comparisons(conjuncts: list[Expr], env: Env) -> list[tuple[str, Expr, Expr]]:
    out: list[tuple[str, Expr, Expr]] = []
    for c in conjuncts:
        if isinstance(c, Compare) and c.op in ORDER_OPS:
            out.append((c.op, c.left, c.right))
        elif isinstance(c, Name) and isinstance(env.get(c.id), IntV):
            out.append((">", c, Num(0)))
    return out


def _arithmetic_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    comps = _comparisons(conjuncts, env)
    if not comps:
        return None
    carried = assigned_names(body)
    atoms: dict[str, Var] = {}
    trace_env = env.copy()
    for name in carried:
        value = env.get(name)
        if isinstance(value, IntV):
            atom = new_var(name, "free")
            atoms[name] = atom
            trace_env.set(name, IntV(Poly.var(atom)))
    if not atoms:
        return None
    sides: list[tuple[str, Poly, Poly]] = []
    unknowns: set[Var] = set()
    for op, left, right in comps:
        lv, rv = it.pure(left, trace_env), it.pure(right, trace_env)
        lm, rm = magnitude(lv), magnitude(rv)
        if (lm is None) != (rm is None):
            # `w >= weights[i]`: one side is read from data. Stand in a fresh size for it; a plan
            # whose trip count does not depend on that size (a count-down from a known start) is
            # sound if the value is non-negative, one that does is not provable and is dropped
            stand_in = new_var("bound", "free")
            unknowns.add(stand_in)
            lm = lm if lm is not None else Poly.var(stand_in)
            rm = rm if rm is not None else Poly.var(stand_in)
        if lm is None or rm is None:
            continue
        sides.append((op, lm, rm))
    if not sides:
        return None
    paths = trace_paths(it, body, trace_env)
    if paths is None:
        return None
    progressing = [(e, s) for e, s in paths if s != "exit"]
    if not progressing:
        return None
    plan = _single_variable_plan(it, sides, atoms, progressing, env)
    if plan is None:
        plan = _potential_plan(it, sides, atoms, progressing, env)
    if plan is None or not unknowns:
        return plan
    if plan.iters.vars() & unknowns:
        return None
    if plan.phi is not None and plan.phi.vars() & unknowns:
        plan.phi, plan.phi_atoms = None, {}  # no usable potential for an enclosing loop to share
    plan.notes.append("a loop bound read from data was assumed non-negative")
    return plan


def _end_values(paths: list[tuple[Env, str]], names: dict[str, Var]) -> list[dict[Var, Poly]]:
    out: list[dict[Var, Poly]] = []
    for e, _ in paths:
        mapping: dict[Var, Poly] = {}
        for name, atom in names.items():
            value = e.get(name)
            mapping[atom] = (
                value.mag
                if isinstance(value, IntV) and value.mag is not None
                else (Poly.var(new_var(name, "free")))
            )
        out.append(mapping)
    return out


def _entry_values(it: Interp, env: Env, atoms: dict[str, Var]) -> dict[Var, Poly]:
    mapping: dict[Var, Poly] = {}
    for name, atom in atoms.items():
        value = env.get(name)
        if isinstance(value, IntV) and value.mag is not None:
            mapping[atom] = value.mag
    return mapping


def _single_variable_plan(
    it: Interp,
    sides: list[tuple[str, Poly, Poly]],
    atoms: dict[str, Var],
    paths: list[tuple[Env, str]],
    env: Env,
) -> LoopPlan | None:
    """One progress variable and one invariant bound: `i < n`, `i * i <= n`, `p < n` with
    `p *= 2`, `n > 1` with `n //= 2`."""
    used: dict[str, Var] = {}
    for _, lm, rm in sides:
        for name, atom in atoms.items():
            if lm.mentions(atom) or rm.mentions(atom):
                used[name] = atom
    if len(used) != 1:
        return None
    name, atom = next(iter(used.items()))
    candidate: tuple[Poly, Poly, bool] | None = None  # (S, B, up)
    for op, lm, rm in sides:
        small, big = (lm, rm) if op in ("<", "<=", "!=") else (rm, lm)
        if op == "!=" and rm.is_zero() and lm.mentions(atom):
            # `n != 0` counts DOWN to zero (a count that grows would need a non-zero target)
            candidate = (lm, rm, False)
        elif small.mentions(atom) and not big.mentions(atom):
            candidate = (small, big, True)
        elif big.mentions(atom) and not small.mentions(atom):
            candidate = (big, small, False)
        else:
            continue
        break
    if candidate is None:
        return None
    s_poly, bound, increasing = candidate
    if any(bound.mentions(a) for a in atoms.values()):
        return None
    updates: list[tuple[Fraction, Poly]] = []
    for e, _ in paths:
        value = e.get(name)
        if not isinstance(value, IntV) or value.mag is None:
            return None
        affine = value.mag.affine_in(atom)
        if affine is None or any(rest_var in atoms.values() for rest_var in affine[1].vars()):
            return None
        updates.append(affine)
    entry = env.get(name)
    start = entry.mag if isinstance(entry, IntV) and entry.mag is not None else None
    degree = s_poly.degree_in(atom)
    slopes = {k for k, _ in updates}
    deltas = [rest for _, rest in updates]
    plan = LoopPlan(iters=ONE)
    # --- polynomial condition, additive progress:  i * i <= n  ->  sqrt(n) iterations
    if degree > 1 and increasing and slopes == {Fraction(1)}:
        steps = [d.const_value() for d in deltas]
        if all(s is not None and s > 0 for s in steps):
            step = min(s for s in steps if s is not None)
            plan.iters = _ordered(bound.power(Fraction(1) / degree) / step)
            _attach_atom(plan, name, atom, "arith", plan.iters)
            return plan
        return None
    if degree != 1:
        return None
    k_values = slopes
    # --- additive:  i += c  /  i -= c
    if k_values == {Fraction(1)}:
        steps = [d.const_value() for d in deltas]
        if any(s is None for s in steps):
            return _symbolic_stride_plan(plan, name, atom, increasing, bound, start, deltas)
        if any(s == 0 for s in steps):
            return None
        signs = {1 if s > 0 else -1 for s in steps if s is not None}
        if len(signs) != 1:
            return None
        step = min(abs(s) for s in steps if s is not None)
        up = signs == {1}
        if up != increasing:
            return None
        if up:
            span = bound - start if start is not None else bound
        else:
            if start is None:
                return None
            span = start - bound
        plan.iters = _ordered(span / step)
        upper = _ordered(bound if up else start if start is not None else bound)
        _attach_atom(plan, name, atom, "arith", upper)
        plan.phi = span
        plan.phi_atoms = {name: atom}
        plan.phi_step = step
        plan.phi = (bound - Poly.var(atom)) if up else (Poly.var(atom) - bound)
        return plan
    # --- multiplicative: p *= k (up) / n //= k, n >>= 1 (down)
    if all(k > 1 for k in k_values) and increasing:
        plan.iters = _ordered(bound.log())
        _attach_atom(plan, name, atom, "geom", _ordered(bound))
        return plan
    if all(0 < k < 1 for k in k_values) and not increasing:
        base = start if start is not None else None
        if base is None:
            return None
        plan.iters = _ordered(base.log())
        _attach_atom(plan, name, atom, "geom", _ordered(base))
        return plan
    return None


def _symbolic_stride_plan(
    plan: LoopPlan,
    name: str,
    atom: Var,
    increasing: bool,
    bound: Poly,
    start: Poly | None,
    deltas: list[Poly],
) -> LoopPlan | None:
    """`for (j = i; j <= n; j += i)`: a stride that is itself a size or an outer index. The loop
    runs (bound - start) / stride times, and summed over `i` that is the harmonic series."""
    if len(set(deltas)) != 1 or len(deltas[0].terms) != 1:
        return None
    stride = deltas[0]
    up = stride.terms[0][1] > 0
    if up != increasing:
        return None
    if up:
        span = bound - start if start is not None else bound
        positive = stride
    else:
        if start is None:
            return None
        span = start - bound
        positive = Poly.const(-1) * stride
    plan.iters = _ordered(span * positive.power(Fraction(-1)))
    _attach_atom(plan, name, atom, "arith", _ordered(bound if up else start))  # type: ignore[arg-type]
    return plan


def _attach_atom(plan: LoopPlan, name: str, _trace_atom: Var, kind: str, upper: Poly) -> None:
    atom = new_var(name, "iter")
    plan.atoms.append((atom, IterInfo(kind, upper)))
    plan.bind[name] = IntV(Poly.var(atom))


def _potential_plan(
    it: Interp,
    sides: list[tuple[str, Poly, Poly]],
    atoms: dict[str, Var],
    paths: list[tuple[Env, str]],
    env: Env,
) -> LoopPlan | None:
    """A quantity that shrinks on every path: the summed distance between compared values.

    Along each path the quantity either drops by a constant (a linear trip count: its starting
    value divided by that constant) or is multiplied by a constant factor below one (a
    logarithmic trip count). Anything else proves nothing and the caller moves on."""
    phi = ZERO
    for op, lm, rm in sides:
        if op == "!=":
            continue
        phi = phi + ((rm - lm) if op in ("<", "<=") else (lm - rm))
    if phi.is_zero() or not any(phi.mentions(a) for a in atoms.values()):
        return None
    steps: list[Fraction] = []
    ratios: list[Fraction] = []
    for mapping in _end_values(paths, atoms):
        delta = phi - phi.substitute(mapping)
        # a nested monotone loop only ever makes EXTRA progress: ignore it, but refuse if it
        # could have moved the quantity the wrong way (negative coefficient on its variable)
        extra_vars = [v for v in delta.vars() if v in it.monotone]
        for v in extra_vars:
            for power, rest in delta.extract_power(v):
                if power != 0 and any(c < 0 for _, c in rest.terms):
                    return None
        clean = delta.substitute(dict.fromkeys(extra_vars, ZERO)) if extra_vars else delta
        step = clean.const_value()
        if step is not None and step >= 1:
            steps.append(step)
            continue
        ratio = _geometric_ratio(phi, phi.substitute(mapping))
        if ratio is None:
            return None
        ratios.append(ratio)
    if bool(steps) == bool(ratios):
        return None  # mixed linear and geometric paths: no single bound
    entry = _entry_values(it, env, atoms)
    start: dict[Var, Poly] = {}
    for atom in atoms.values():
        if atom in entry:
            start[atom] = entry[atom]
            continue
        affine = phi.affine_in(atom)
        if affine is None or affine[0] > 0:
            return None  # the distance grows with a start we do not know: no sound bound
        start[atom] = ZERO  # an index-like start at or above zero only shrinks the distance
    phi0 = _ordered(phi.substitute(start))
    plan = LoopPlan(iters=ONE, phi=phi, phi_atoms=dict(atoms))
    if steps:
        step = min(steps)
        plan.iters = _ordered(phi0 / step)
        plan.phi_kind, plan.phi_step = "arith", step
    else:
        plan.iters = _ordered(phi0.log())
        plan.phi_kind = "geom"
    return plan


def _geometric_ratio(phi: Poly, phi_next: Poly) -> Fraction | None:
    """If phi_next = alpha * phi + constant with 0 < alpha < 1, return alpha."""
    for mono, coef in phi.terms:
        if mono.is_constant():
            continue
        other = dict(phi_next.terms).get(mono)
        if other is None:
            return None
        alpha = other / coef
        if not (0 < alpha < 1):
            return None
        remainder = phi_next - phi * alpha
        return alpha if remainder.const_value() is not None else None
    return None


# ================================================================================= pointers
def _pointer_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    sizes: list[Poly] = []
    for c in conjuncts:
        for node in walk(c):
            if isinstance(node, Name | Attribute):
                value = it.pure(node, env)
                if isinstance(value, NodeV):
                    sizes.append(value.size)
    if not sizes:
        return None
    best = sizes[0]
    for other in sizes[1:]:
        if (best + other).order() == other.order() and (best + other).order() != best.order():
            best = other
    return LoopPlan(iters=_ordered(best))


def _index_pointer_plan(
    it: Interp, conjuncts: list[Expr], body: tuple[Stmt, ...], env: Env
) -> LoopPlan | None:
    """A monotone pointer that indexes a container cannot pass its end."""
    found: list[tuple[str, ContV]] = []
    for name in assigned_names(body):
        if not isinstance(env.get(name), IntV):
            continue
        sign = None
        monotone = True
        for stmt in body:
            for node in walk(stmt):
                if isinstance(node, Assign):
                    for target in node.targets:
                        if isinstance(target, Name) and target.id == name:
                            if node.op in ("+=", "-=") and isinstance(node.value, Num):
                                s = 1 if node.op == "+=" else -1
                                if sign is not None and s != sign:
                                    monotone = False
                                sign = s
                            else:
                                monotone = False
        if not monotone or sign is None:
            continue
        for stmt in (*body, *[ExprStmt(c) for c in conjuncts]):
            for node in walk(stmt):
                access = _index_access(node)
                if access is not None and isinstance(access[1], Name | ListLit):
                    if name in _names_in(access[1]):
                        value = env.get(access[0])
                        if isinstance(value, ContV):
                            found.append((name, value))
    if not found:
        return None
    length = found[0][1].length
    plan = LoopPlan(iters=_ordered(length), index_pointers=found)
    return plan


# ================================================================================ amortise
def _walk_excluding(root: object, excluded: object):  # noqa: ANN202
    stack = [root]
    while stack:
        node = stack.pop()
        if node is excluded:
            continue
        yield node
        stack.extend(children(node))


def _reset_in(frame_loop: Loop, inner: Loop, name: str) -> bool:
    for stmt in frame_loop.body:
        for node in _walk_excluding(stmt, inner):
            if isinstance(node, Assign) and node.op == "=":
                for target in node.targets:
                    names = (
                        [target]
                        if isinstance(target, Name)
                        else (list(target.elts) if isinstance(target, ListLit) else [])
                    )
                    if any(isinstance(n, Name) and n.id == name for n in names):
                        return True
            if isinstance(node, Loop) and node.target is not None:
                names = (
                    [node.target]
                    if isinstance(node.target, Name)
                    else (list(node.target.elts) if isinstance(node.target, ListLit) else [])
                )
                if any(isinstance(n, Name) and n.id == name for n in names):
                    return True
    return False


def _initialised_by(loop: Loop) -> set[str]:
    """Names the loop's own header assigns each time it is entered: `for (w = cap; ...)`."""
    names: set[str] = set()
    for stmt in loop.init:
        for node in walk(stmt):
            if isinstance(node, Assign):
                for target in node.targets:
                    if isinstance(target, Name):
                        names.add(target.id)
                    elif isinstance(target, ListLit):
                        names.update(t.id for t in target.elts if isinstance(t, Name))
    return names


def _maybe_amortize(it: Interp, plan: LoopPlan, loop: Loop, env: Env) -> None:
    stack = it._loop_stack
    if not stack:
        return
    restarted = _initialised_by(loop)
    # (a) a pointer with a proven potential
    if plan.phi is not None and plan.phi_atoms:
        names = [n for n, a in plan.phi_atoms.items() if plan.phi.mentions(a)]
        for frame in stack:  # outermost first
            if any(n in restarted for n in names):
                break
            if any(_reset_in(frame.loop, loop, n) for n in names):
                continue
            entry = _entry_values(it, frame.entry_env, plan.phi_atoms)
            if not entry or any(
                n not in {nm for nm, a in plan.phi_atoms.items() if a in entry} for n in names
            ):
                continue
            total = plan.phi.substitute(entry)
            bounds = {
                a: info.upper for f in stack for a, info in it.iters.items() if total.mentions(a)
            }
            total = total.substitute(bounds) if bounds else total
            total = _ordered(total)
            plan.amortize_into = frame.id
            plan.amortized_total = (
                _ordered(total / plan.phi_step)
                if plan.phi_kind == "arith"
                else _ordered(total.log())
            )
            return
    # (b) a worklist created before the enclosing loop: its pops total at most its capacity
    if plan.drain is not None:
        current = env.get(plan.drain)
        if isinstance(current, ContV) and current.cap is not None:
            for frame in stack:
                held = frame.entry_env.get(plan.drain)
                if isinstance(held, ContV) and held.uid == current.uid:
                    plan.amortize_into = frame.id
                    plan.amortized_total = Poly.var(current.cap)
                    return
    # (c) an index pointer that is never reset
    if plan.index_pointers:
        names = [n for n, _ in plan.index_pointers]
        if any(n in restarted for n in names):
            return
        for frame in stack:
            if any(_reset_in(frame.loop, loop, n) for n in names):
                continue
            plan.amortize_into = frame.id
            plan.amortized_total = plan.iters
            return
