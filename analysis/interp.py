"""The abstract interpreter: symbolic cost analysis over the engine's AST.

It does not run the program. It walks it once, tracking for every variable an ABSTRACT VALUE
(an integer's magnitude as a polynomial, a container's length, how many nodes a linked structure
reaches) and charging cost for what each statement and call does. Loops are folded with the
bounds found by `analysis/bounds.py`; user functions are summarised once and instantiated at each
call; library calls come from `analysis/stdlib.py`.

Two ideas keep it honest:

  * Every number is an UPPER BOUND on a size. Where nothing better is known the engine assumes
    "proportional to the input" and RECORDS that as a `Note`, so a result built on an assumption
    says so instead of passing as certain.
  * Local containers get a symbolic CAPACITY variable. Their length is whatever they grow to,
    which depends on loops not yet analysed, so it is substituted once the growth is known
    (immediately for a container created inside a loop, at function end otherwise).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction

from analysis.bundle import Bundle, Cost, RecCall
from analysis.env import Env
from analysis.nodes import (
    Assign,
    Attribute,
    BinOp,
    BoolOp,
    Break,
    Call,
    Comp,
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
    Return,
    Slice,
    Star,
    Stmt,
    Str,
    Subscript,
    Switch,
    Try,
    TypeRef,
    Unknown,
    UnOp,
    walk,
)
from analysis.poly import Poly, Var, new_var
from analysis.roles import NODE_ATTRS, _container_kind, params_to_values
from analysis.stdlib import LibCall, call_library
from analysis.values import (
    NONE,
    SCALAR,
    UNKNOWN,
    ContV,
    FuncV,
    IntV,
    NodeV,
    NoneV,
    ScalarV,
    TupleV,
    UnknownV,
    Value,
    const_int,
    join,
    magnitude,
    make_container,
    mem,
    substitute,
)

ONE = Poly.const(1)
ZERO = Poly.zero()
_SELF_NAMES = frozenset({"this", "self", "cls", "super"})
_JAVA_CAPACITY_TYPES = frozenset(
    {"ArrayList", "HashMap", "HashSet", "StringBuilder", "StringBuffer", "ArrayDeque",
     "PriorityQueue", "LinkedHashMap", "LinkedHashSet", "Vector", "Hashtable"}
)  # fmt: skip
MAX_CALL_DEPTH = 24
_ROW_GETTERS = frozenset({"get", "at", "Get"})
_HOF_METHODS = frozenset(
    {"map", "filter", "forEach", "reduce", "reduceRight", "some", "every", "find", "findIndex",
     "findLast", "findLastIndex", "flatMap", "mapToInt", "mapToObj", "mapToLong", "mapToDouble",
     "anyMatch", "allMatch", "noneMatch", "peek", "removeIf", "takeWhile", "dropWhile"}
)  # fmt: skip
_HOF_FOLDS = frozenset({"reduce", "reduceRight"})
_HOF_RECEIVER, _HOF_ITEM, _HOF_ACC = "<hof receiver>", "<hof item>", "<hof acc>"


@dataclass(frozen=True)
class Note:
    kind: str  # "assumed": a bound was assumed;  "unknown": no meaningful answer
    reason: str
    line: int = 0


@dataclass
class IterInfo:
    """What is known about an induction variable (an `iter` atom)."""

    kind: str  # arith | geom
    upper: Poly  # upper limit of the values the variable takes
    domain: int | None = None  # uid of the container whose indices (all of them) it ranges over


@dataclass
class LoopFrame:
    id: int
    iters: Poly
    loop: Loop
    entry_env: Env
    pinned: dict[str, Value] = field(default_factory=dict)  # popped vertices keep their range


@dataclass
class FuncSummary:
    func: FuncDef
    params: dict[str, Value]
    time: Poly
    space: Poly
    ret: Value | None
    rec: list[tuple[RecCall, ...]] = field(default_factory=lambda: [()])
    recursive: bool = False
    solved: bool = True


class Interp:
    def __init__(self, module: Module, language: str) -> None:
        self.module = module
        self.lang = language
        self.funcs: dict[str, list[FuncDef]] = defaultdict(list)
        for func in module.functions:
            self.funcs[func.name].append(func)
        self.class_names = {f.cls for f in module.functions if f.cls}
        self._callee_table: dict[str, FuncDef] = {
            name: candidates[0] for name, candidates in self.funcs.items()
        }
        self.input_var = new_var("n", "input")
        self.input_size = Poly.var(self.input_var)
        self.default_size = self.input_size
        self.notes: list[Note] = []
        self._cur = Bundle()
        self._ids = itertools.count(1)
        self.iters: dict[Var, IterInfo] = {}
        self.monotone: dict[Var, int] = {}  # progress atoms added by nested monotone loops
        self.cap_initial: dict[Var, Poly] = {}
        self.cap_override: dict[Var, Poly] = {}
        self.rowlen_vars: dict[int, Var] = {}
        self.total_vars: dict[int, Var] = {}
        self.cols_vars: dict[int, Var] = {}
        self.row_owner: dict[int, int] = {}  # row uid -> uid of the container whose rows they are
        self.ragged_uids: set[int] = set()  # containers a traversal proves to be adjacency lists
        self._loop_stack: list[LoopFrame] = []
        self._returns: list[Value] = []
        self._summaries: dict[int, FuncSummary] = {}
        self._active: list[FuncDef] = []
        self._cur_func: FuncDef | None = None
        self._last_value: Value = UNKNOWN
        self._depth = 0

    # ================================================================== utilities
    def note(self, kind: str, reason: str, line: int = 0) -> None:
        note = Note(kind, reason, line)
        if note not in self.notes:
            self.notes.append(note)

    @contextmanager
    def scratch(self) -> Iterator[Bundle]:
        """Evaluate without charging the enclosing statement or recording notes."""
        saved_cur, saved_notes = self._cur, len(self.notes)
        self._cur = Bundle()
        try:
            yield self._cur
        finally:
            self._cur = saved_cur
            del self.notes[saved_notes:]

    def pure(self, expr: Expr, env: Env) -> Value:
        with self.scratch():
            return self.evaluate(expr, env)

    def absorb(self, other: Bundle) -> None:
        """Merge `other` into the bundle currently being built, in place."""
        cur = self._cur
        merged = cur.then(other)
        cur.time, cur.grows, cur.allocs = merged.time, merged.grows, merged.allocs
        cur.retained, cur.created, cur.rec = merged.retained, merged.created, merged.rec

    def rowlen_var(self, uid: int) -> Var:
        if uid not in self.rowlen_vars:
            self.rowlen_vars[uid] = new_var("row length", "rowlen")
        return self.rowlen_vars[uid]

    def total_var(self, uid: int) -> Var:
        if uid not in self.total_vars:
            self.total_vars[uid] = new_var("total entries", "total")
        return self.total_vars[uid]

    def cols_var(self, uid: int) -> Var:
        if uid not in self.cols_vars:
            self.cols_vars[uid] = new_var("max row length", "len")
        return self.cols_vars[uid]

    # ================================================================== functions
    def lookup_function(self, name: str) -> FuncDef | None:
        candidates = self.funcs.get(name, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        current = self._cur_func.cls if self._cur_func is not None else None
        for candidate in candidates:
            if candidate.cls == current:
                return candidate
        return candidates[0]

    def summary_for(self, fdef: FuncDef) -> FuncSummary:
        key = id(fdef)
        if key in self._summaries:
            return self._summaries[key]
        self._active.append(fdef)
        saved = (self._cur, self._returns, self._loop_stack, self._cur_func, self.default_size)
        self._cur, self._returns, self._loop_stack, self._cur_func = Bundle(), [], [], fdef
        try:
            params = params_to_values(fdef, self.lang, self._callee_table)
            self.default_size = self._first_size(params)
            env = Env(dict(params))
            body = self.exec_block(fdef.body, env)
            # entries pushed into the rows of a container built here add up to its total
            body.created |= {v for v in body.grows if v.kind == "total"}
            body = self.finalize_caps(body, body.created)
            summary = self._make_summary(fdef, params, body)
        finally:
            self._active.pop()
            self._cur, self._returns, self._loop_stack, self._cur_func, self.default_size = saved
        self._summaries[key] = summary
        return summary

    def _first_size(self, params: dict[str, Value]) -> Poly:
        for value in params.values():
            if isinstance(value, ContV):
                return value.length
            if isinstance(value, IntV) and value.mag is not None:
                return value.mag
            if isinstance(value, NodeV):
                return value.size
        return self.input_size

    def _make_summary(self, fdef: FuncDef, params: dict[str, Value], body: Bundle) -> FuncSummary:
        ret: Value | None = None
        for value in self._returns:
            ret = join(ret, value) if ret is not None else value
        recursive = any(alt for alt in body.rec)
        time = body.time.total()
        space = (body.allocs + body.retained.total()).order()
        summary = FuncSummary(
            func=fdef,
            params=params,
            time=time.order() if not time.is_zero() else ONE,
            space=space if not space.is_zero() else ONE,
            ret=ret,
            rec=body.rec,
            recursive=recursive,
            solved=not recursive,
        )
        if recursive:
            from analysis.recurrence import solve_recurrence

            solve_recurrence(self, summary, body)
        return summary

    def bind_args(self, params: dict[str, Value], args: list[Value]) -> dict[Var, Poly]:
        mapping: dict[Var, Poly] = {}
        for (_, formal), actual in zip(params.items(), args, strict=False):
            self._match(formal, actual, mapping)
        return mapping

    def _match(self, formal: Value, actual: Value, mapping: dict[Var, Poly]) -> None:
        if isinstance(formal, IntV) and formal.mag is not None:
            var = _single_var(formal.mag)
            if var is not None:
                if isinstance(actual, IntV) and actual.mag is not None:
                    mapping[var] = actual.mag
                elif isinstance(actual, ContV):
                    mapping[var] = actual.length
                else:
                    mapping[var] = self.default_size
                    self.note("assumed", f"size of argument for {var.name} unknown")
        elif isinstance(formal, ContV):
            var = _single_var(formal.length)
            if var is not None:
                if isinstance(actual, ContV):
                    mapping[var] = actual.length
                elif isinstance(actual, TupleV):
                    mapping[var] = Poly.const(len(actual.items))
                elif isinstance(actual, IntV) and actual.mag is not None:
                    mapping[var] = actual.mag
                else:
                    mapping[var] = self.default_size
                    self.note("assumed", f"size of argument for {var.name} unknown")
            if formal.elem is not None:
                nested = actual.elem if isinstance(actual, ContV) else None
                if nested is not None:
                    self._match(formal.elem, nested, mapping)
                elif isinstance(formal.elem, ContV):
                    inner = _single_var(formal.elem.length)
                    if inner is not None:
                        mapping[inner] = self.default_size
        elif isinstance(formal, NodeV):
            var = _single_var(formal.size)
            if var is not None:
                if isinstance(actual, NodeV):
                    mapping[var] = actual.size
                elif isinstance(actual, NoneV):
                    mapping[var] = ONE
                else:
                    mapping[var] = self.default_size
                    self.note("assumed", "size of a linked-structure argument unknown")

    # ================================================================== statements
    def exec_block(self, stmts: tuple[Stmt, ...] | list[Stmt], env: Env) -> Bundle:
        total = Bundle()
        for stmt in stmts:
            total = total.then(self.exec_stmt(stmt, env))
        return total

    def exec_stmt(self, st: Stmt, env: Env) -> Bundle:
        saved = self._cur
        b = Bundle()
        self._cur = b
        try:
            b.charge(ONE)
            if isinstance(st, Assign):
                self.exec_assign(st, env)
            elif isinstance(st, ExprStmt):
                self._last_value = self.evaluate(st.expr, env)
            elif isinstance(st, Return):
                value = self.evaluate(st.value, env) if st.value is not None else NONE
                self._returns.append(value)
            elif isinstance(st, Break | Continue):
                pass
            elif isinstance(st, If):
                return self.exec_if(st, env, b)
            elif isinstance(st, Loop):
                return self.exec_loop(st, env, b)
            elif isinstance(st, Switch):
                return self.exec_switch(st, env, b)
            elif isinstance(st, Try):
                return self.exec_try(st, env, b)
            elif isinstance(st, FuncDef):
                env.set(st.name, FuncV(st, env, st.name))
            return b
        finally:
            self._cur = saved

    @staticmethod
    def always_exits(stmts: tuple[Stmt, ...]) -> bool:
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, Return | Break | Continue):
            return True
        if isinstance(last, If):
            return Interp.always_exits(last.body) and Interp.always_exits(last.orelse)
        return False

    def exec_if(self, st: If, env: Env, b: Bundle) -> Bundle:
        self.evaluate(st.test, env)
        env_then, env_else = env.copy(), env.copy()
        then_b = self.exec_block(st.body, env_then)
        else_b = self.exec_block(st.orelse, env_else)
        dead_then, dead_else = self.always_exits(st.body), self.always_exits(st.orelse)
        if dead_then and not dead_else:
            merged = env_else
        elif dead_else and not dead_then:
            merged = env_then
        else:
            merged = env_then.merge(env_else)
        env.vars.clear()
        env.vars.update(merged.vars)
        return b.then(then_b.either(else_b))

    def exec_switch(self, st: Switch, env: Env, b: Bundle) -> Bundle:
        self.evaluate(st.subject, env)
        result: Bundle | None = None
        merged: Env | None = None
        for arm in st.arms:
            arm_env = env.copy()
            arm_b = self.exec_block(arm, arm_env)
            result = arm_b if result is None else result.either(arm_b)
            merged = arm_env if merged is None else merged.merge(arm_env)
        if merged is not None:
            env.vars.clear()
            env.vars.update(merged.vars)
        return b.then(result) if result is not None else b

    def exec_try(self, st: Try, env: Env, b: Bundle) -> Bundle:
        total = b.then(self.exec_block(st.body, env))
        handlers: Bundle | None = None
        for handler in st.handlers:
            h = self.exec_block(handler, env.copy())
            handlers = h if handlers is None else handlers.either(h)
        if handlers is not None:
            total = total.then(handlers)
        return total.then(self.exec_block(st.final, env))

    # ---------------------------------------------------------------- assignment
    def exec_assign(self, st: Assign, env: Env) -> None:
        if st.value is None:
            value = self.default_for_decl(st.decl)
        else:
            value = self.evaluate(st.value, env)
        if st.op != "=":
            value = self.augmented(st, env, value)
        elif self._is_copy_init(st, value):
            value = self._copy_container(value)  # type: ignore[arg-type]
        for target in st.targets:
            self.bind(target, value, env, st)

    def _is_copy_init(self, st: Assign, value: Value) -> bool:
        """C++ value semantics: `vector<int> items = nums;` copies, `vector<int>& items = nums;`
        and `vector<int>* p = &nums;` do not."""
        return (
            self.lang == "cpp"
            and st.decl is not None
            and not st.decl.ref
            and st.decl.dims == 0
            and isinstance(value, ContV)
            and isinstance(st.value, Name | Subscript | Attribute)
        )

    def _copy_container(self, value: ContV) -> ContV:
        self._cur.charge(mem(value))
        copy = make_container(value.kind, value.length, value.elem, owned=True)
        self._cur.alloc(mem(copy))
        return copy

    def default_for_decl(self, decl: TypeRef | None) -> Value:
        if decl is None:
            return UNKNOWN
        name = decl.name
        if decl.dims > 0:
            size = decl.sizes[0] if decl.sizes else None
            if size is not None:
                length = magnitude(self.pure(size, Env())) if not isinstance(size, Num) else None
                if isinstance(size, Num) and size.value is not None:
                    length = Poly.const(int(size.value))
                if length is not None:
                    return self._alloc("array", length, SCALAR)
            if name == "char" and decl.dims == 1:
                return self._alloc("str", ZERO)
            return self._alloc("array", ZERO)
        kind = _container_kind(name)
        if kind is None and (name.endswith("Builder") or name.endswith("Buffer")):
            kind = "str"
        if kind is not None:
            return self._alloc(kind, ZERO)
        if name in ("int", "long", "short", "Integer", "Long", "size_t", "int64", "int32"):
            return IntV(None)
        if name[:1].isupper() or name in ("struct",):
            return NONE
        return SCALAR

    def _alloc(self, kind: str, length: Poly, elem: Value | None = None) -> ContV:
        container = make_container(kind, length, elem, owned=True)
        self._cur.alloc(mem(container))
        return container

    def augmented(self, st: Assign, env: Env, rhs: Value) -> Value:
        op = st.op[:-1]
        target = st.targets[0]
        current = self.pure(target, env)
        if isinstance(current, ContV) and op == "+":
            added = rhs.length if isinstance(rhs, ContV) else ONE
            if current.kind == "str":
                immutable = self.lang != "cpp"
                self._cur.charge(current.length if immutable else added)
                if immutable and isinstance(target, Name):
                    self._cur.alloc(current.length)
            else:
                self._cur.charge(added)
            if current.cap is not None:
                self._cur.grow(current.cap, added if isinstance(rhs, ContV) else ONE)
            return current
        return self.binop_value(op, current, rhs)

    def bind(self, target: Expr, value: Value, env: Env, st: Assign | None = None) -> None:
        if isinstance(target, Name):
            for frame in reversed(self._loop_stack):
                if target.id in frame.pinned:
                    # `node = queue.popleft()`: the pop is what the loop plan already ranged over
                    value = frame.pinned[target.id]
                    break
            old = env.get(target.id)
            if (
                st is not None
                and isinstance(old, ContV)
                and old.cap is not None
                and isinstance(value, ContV)
                and value.uid != old.uid
                and value.length.is_zero()
            ):  # cur = [] / cur = empty string / cur = cur[:0]: the old buffer is emptied
                self._cur.reset(old.cap)
            if st is not None and isinstance(value, ContV) and value.owned and value.cap is None:
                value = self.make_growable(value, target.id)
            env.set(target.id, value)
        elif isinstance(target, ListLit):
            self._bind_tuple(target, value, env, st)
        elif isinstance(target, Subscript):
            container = self.pure(target.obj, env)
            if isinstance(container, ContV):
                if container.kind == "treemap":
                    self._cur.charge(container.length.log())  # an ordered map: log per insert
                if container.kind in ("dict", "treemap") and container.cap is not None:
                    self._cur.grow(container.cap, mem(value))
                restored = isinstance(container.elem, ContV) and (
                    isinstance(value, ContV) and value.uid == container.elem.uid
                )  # `g[u] = append(g[u], v)` stores the very row it read: nothing new retained
                if isinstance(value, ContV) and value.owned and not restored:
                    self._cur.retain(mem(value))
                if (
                    isinstance(value, ContV)
                    and not restored
                    and isinstance(target.obj, Name)
                    and container.kind in ("list", "array")
                    and (container.elem is None or isinstance(container.elem, ContV))
                ):  # `grid[i] = make([]int, m)`: the rows are now known to have that length
                    elem = value if container.elem is None else join(container.elem, value)
                    env.set(
                        target.obj.id,
                        replace(
                            container, elem=elem, ragged=container.ragged and value.length.is_zero()
                        ),
                    )
        elif isinstance(target, Attribute):
            if isinstance(value, ContV) and value.owned:
                self._cur.retain(mem(value))

    def _bind_tuple(self, target: ListLit, value: Value, env: Env, st: Assign | None) -> None:
        names = target.elts
        if isinstance(value, TupleV) and len(value.items) == len(names):
            for sub, item in zip(names, value.items, strict=True):
                self.bind(sub, item, env, st)
            return
        if isinstance(value, ContV):
            for sub in names:
                self.bind(sub, value.elem if value.elem is not None else SCALAR, env, st)
            return
        for sub in names:
            self.bind(sub, UNKNOWN, env, st)

    def make_growable(self, value: ContV, name: str) -> ContV:
        cap = new_var(name, "cap")
        self.cap_initial[cap] = value.length
        self._cur.created.add(cap)
        return replace(value, length=Poly.var(cap), cap=cap)

    # ---------------------------------------------------------------- capacities
    def resolve_caps(self, bundle: Bundle, caps: set[Var]) -> dict[Var, Poly]:
        values: dict[Var, Poly] = {}
        for _ in range(8):
            changed = False
            for cap in caps:
                growth = bundle.grows.get(cap)
                total = self.cap_initial.get(cap, ZERO) + (
                    growth.total() if growth is not None else ZERO
                )
                value = total.substitute(values)
                value = value.order() if not value.is_zero() else ZERO
                if cap in self.cap_override:
                    value = self.cap_override[cap].substitute(values)
                    value = value.order() if not value.is_zero() else ZERO
                if values.get(cap) != value:
                    values[cap] = value
                    changed = True
            if not changed:
                break
        cyclic = {c for c, v in values.items() if v.vars() & caps}
        for cap in cyclic:
            values[cap] = self.default_size
            self.note("assumed", f"size of container {cap.name} depends on itself; assumed linear")
        return {c: (v if not v.is_zero() else ZERO) for c, v in values.items()}

    def finalize_caps(self, bundle: Bundle, caps: set[Var]) -> Bundle:
        if not caps:
            return bundle
        values = self.resolve_caps(bundle, caps)

        def sub_cost(cost: Cost) -> Cost:
            own = cost.own.substitute(values) if not cost.own.is_zero() else cost.own
            lumps = {k: v.substitute(values) for k, v in cost.lumps.items()}
            return Cost(own.order() if not own.is_zero() else own, lumps)

        final_space = ZERO
        for cap in caps:
            final_space = final_space + values.get(cap, ZERO)
        allocs = (
            (bundle.allocs.substitute(values) + final_space)
            if not (bundle.allocs.is_zero() and final_space.is_zero())
            else ZERO
        )
        return Bundle(
            time=sub_cost(bundle.time),
            grows={c: sub_cost(g) for c, g in bundle.grows.items() if c not in caps},
            allocs=allocs.order() if not allocs.is_zero() else allocs,
            retained=sub_cost(bundle.retained),
            created=bundle.created - caps,
            rec=bundle.rec,
            unflushed=bundle.unflushed - caps,
            flushed=bundle.flushed - caps,
            resets=bundle.resets - caps,
        )

    def apply_flushes(self, inner: Bundle) -> Bundle:
        """A buffer consumed (copied, joined, scanned) and then emptied on every path of the loop
        body holds, at each consumption, only what was added since the last reset, so the total
        consumed over the loop is at most the total added: substitute its per-iteration growth
        for its size in the cumulative costs (time, retained memory, growth of other containers).
        Temporaries (allocs) are a peak, not a sum, and keep the full size."""
        mapping: dict[Var, Poly] = {}
        for cap in inner.flushed - inner.unflushed:
            growth = inner.grows.get(cap)
            per_iteration = growth.total() if growth is not None else ZERO
            if not per_iteration.is_zero() and not per_iteration.mentions(cap):
                mapping[cap] = per_iteration
        if not mapping:
            return inner

        def sub(cost: Cost) -> Cost:
            own = cost.own.substitute(mapping).order() if not cost.own.is_zero() else cost.own
            return Cost(own, {k: v.substitute(mapping).order() for k, v in cost.lumps.items()})

        return Bundle(
            time=sub(inner.time),
            grows={c: (g if c in mapping else sub(g)) for c, g in inner.grows.items()},
            allocs=inner.allocs,
            retained=sub(inner.retained),
            created=inner.created,
            rec=inner.rec,
            unflushed=inner.unflushed,
            flushed=inner.flushed - set(mapping),
            resets=inner.resets,
        )

    # ---------------------------------------------------------------- loops
    def exec_loop(self, st: Loop, env: Env, b: Bundle) -> Bundle:
        from analysis.bounds import plan_loop

        loop_id = next(self._ids)
        plan = plan_loop(self, st, env, loop_id)
        for note in plan.notes:
            self.note("assumed", note, st.line)
        body_env = env.copy()
        assigned = assigned_names(st.body)
        for name in assigned:
            existing = body_env.vars.get(name)
            if isinstance(existing, IntV):
                body_env.vars[name] = IntV(None)
        for name, value in plan.bind.items():
            body_env.vars[name] = value
        for atom, info in plan.atoms:
            self.iters[atom] = info
        pinned = {name: plan.bind[name] for name in plan.pinned if name in plan.bind}
        frame = LoopFrame(loop_id, plan.iters, st, env.copy(), pinned)
        self._loop_stack.append(frame)
        try:
            inner = self.exec_block(st.body, body_env)
        finally:
            self._loop_stack.pop()
        overhead = Bundle()
        overhead.charge(ONE)
        inner = overhead.then(inner)
        retired = self.resolve_caps(inner, inner.created) if inner.created else {}
        inner = self.finalize_caps(inner, inner.created)
        inner = self.apply_flushes(inner)
        inner = self._bound_allocs(inner, plan)
        folded = self.fold_loop(inner, plan, loop_id)
        self._post_loop_env(st, env, body_env, plan, assigned)
        if retired:
            self._retire_caps(env, retired)
        return b.then(folded)

    def _retire_caps(self, env: Env, values: dict[Var, Poly]) -> None:
        """Containers created inside a loop were sized when it closed; a name that still refers to
        one afterwards (through a join with another branch) must not keep the stale capacity."""
        for name, value in list(env.vars.items()):
            if isinstance(value, ContV):
                fixed = substitute(value, values)
                if isinstance(fixed, ContV) and fixed.cap in values:
                    fixed = replace(fixed, cap=None)
                env.vars[name] = fixed  # type: ignore[assignment]

    def _bound_allocs(self, inner: Bundle, plan) -> Bundle:  # noqa: ANN001
        """Temporaries are a peak, not a sum, so an induction variable inside one (a slice taken
        between two loop indices) is at most the upper bound of its loop."""
        mapping = {atom: info.upper for atom, info in plan.atoms if inner.allocs.mentions(atom)}
        if not mapping:
            return inner
        bounded = inner.allocs.substitute(mapping)
        return replace(inner, allocs=bounded.order() if not bounded.is_zero() else bounded)

    def fold_loop(self, inner: Bundle, plan, loop_id: int) -> Bundle:  # noqa: ANN001
        if plan.amortize_into is not None and plan.amortized_total is not None:
            return self._amortize(inner, plan)
        return inner.fold(loop_id, lambda p: self.repeat_poly(p, plan), plan.iters)

    def _amortize(self, inner: Bundle, plan) -> Bundle:  # noqa: ANN001
        """The loop runs `amortized_total` times IN TOTAL across the enclosing loop, not per
        enclosing iteration: its whole cost becomes a lump owned by that enclosing loop."""
        target = plan.amortize_into
        total = plan.amortized_total

        def lump(cost: Cost) -> Cost:
            if cost.own.is_zero():
                return Cost(ZERO, dict(cost.lumps))
            per = self._bound_atoms(cost.own, plan)
            return Cost(ZERO, {**cost.lumps, target: (per * total).order()})

        time = lump(inner.time)
        time = Cost(ONE, time.lumps)
        return Bundle(
            time=time,
            grows={c: lump(g) for c, g in inner.grows.items()},
            allocs=inner.allocs,
            retained=lump(inner.retained),
            created=set(inner.created),
            rec=inner.rec,
            unflushed=set(inner.unflushed),
        )

    def _bound_atoms(self, poly: Poly, plan) -> Poly:  # noqa: ANN001
        mapping = {atom: info.upper for atom, info in plan.atoms if poly.mentions(atom)}
        return poly.substitute(mapping) if mapping else poly

    def repeat_poly(self, p: Poly, plan) -> Poly:  # noqa: ANN001
        if p.is_zero():
            return p
        uids = [plan.ragged_uid] if plan.ragged_uid is not None else []
        uids += [i.domain for _, i in plan.atoms if i.domain is not None and i.domain not in uids]
        for uid in uids:
            row = self.rowlen_vars.get(uid)
            if row is not None and p.mentions(row):
                return self._repeat_ragged(p, plan, row, uid)
        return self._repeat_plain(p, plan)

    def _repeat_plain(self, p: Poly, plan) -> Poly:  # noqa: ANN001
        mentioned = [(a, i) for a, i in plan.atoms if p.mentions(a)]
        if not mentioned:
            return p * plan.iters
        atom, info = mentioned[0]
        for other, other_info in mentioned[1:]:
            p = p.substitute({other: other_info.upper})
        if plan.iters.order() == info.upper.order():
            return p.sum_over(atom, info.kind, info.upper)
        return p.substitute({atom: info.upper}) * plan.iters

    def _repeat_ragged(self, p: Poly, plan, row: Var, uid: int) -> Poly:  # noqa: ANN001
        """Summing a per-row cost over ALL rows of a container: sum of row lengths is the total
        number of entries, not rows * (longest row). This is what makes an adjacency-list walk
        O(V + E) instead of O(V * max degree)."""
        total = self.total_var(uid)
        cols = self.cols_var(uid)
        result = ZERO
        for power, rest in p.extract_power(row):
            if power == 0:
                result = result + self._repeat_plain(rest, plan)
            elif power == 1:
                result = result + Poly.var(total) * rest
            else:
                result = result + Poly.var(total) * Poly.var(cols).power(power - 1) * rest
        return result

    def _post_loop_env(
        self,
        st: Loop,
        env: Env,
        body_env: Env,
        plan,
        assigned: set[str],  # noqa: ANN001
    ) -> None:
        for name in assigned | {n for n in plan.post}:
            if name in plan.post:
                env.vars[name] = plan.post[name]
                continue
            value = body_env.vars.get(name)
            before = env.vars.get(name)
            if isinstance(value, IntV) or isinstance(before, IntV):
                env.vars[name] = IntV(None)
            elif value is not None:
                env.vars[name] = join(before, value) if before is not None else value
        for name, before in list(env.vars.items()):
            value = body_env.vars.get(name)
            if (
                name not in assigned
                and isinstance(before, ContV)
                and isinstance(value, ContV)
                and value.uid == before.uid
                and value != before
            ):  # `g.add(new ArrayList<>())` in the body taught the container its row shape
                env.vars[name] = join(before, value)

    # ================================================================== expressions
    def evaluate(self, e: Expr, env: Env) -> Value:  # noqa: C901
        if isinstance(e, Name):
            value = env.get(e.id)
            if value is not None:
                return value
            func = self.lookup_function(e.id)
            return FuncV(func, None, e.id) if func is not None else UNKNOWN
        if isinstance(e, Num):
            return const_int(e.value)
        if isinstance(e, Str):
            # a literal is a fresh object: bound to a name it can grow (`s = ""; s += c`)
            return make_container("str", Poly.const(len(e.value)), owned=True)
        if isinstance(e, Const):
            return NONE if e.kind == "none" else SCALAR
        if isinstance(e, BinOp):
            return self.ev_binop(e, env)
        if isinstance(e, UnOp):
            operand = self.evaluate(e.operand, env)
            if e.op == "-" and isinstance(operand, IntV):
                return IntV(operand.mag)
            return (
                SCALAR if e.op == "not" else (IntV(None) if isinstance(operand, IntV) else SCALAR)
            )
        if isinstance(e, Compare):
            return self.ev_compare(e, env)
        if isinstance(e, BoolOp):
            for part in e.values:
                self.evaluate(part, env)
            return SCALAR
        if isinstance(e, IfExp):
            return self.ev_ifexp(e, env)
        if isinstance(e, Call):
            return self.ev_call(e, env)
        if isinstance(e, Attribute):
            return self.ev_attribute(e, env)
        if isinstance(e, Subscript):
            return self.ev_subscript(e, env)
        if isinstance(e, Slice):
            return self.ev_slice(e, env)
        if isinstance(e, ListLit):
            return self.ev_listlit(e, env)
        if isinstance(e, DictLit):
            items = [(self.evaluate(k, env), self.evaluate(v, env)) for k, v in e.items]
            elem = items[0][1] if items else None
            container = make_container("dict", Poly.const(len(items)), elem, owned=True)
            self._cur.alloc(Poly.const(max(1, len(items))))
            return container
        if isinstance(e, Comp):
            return self.ev_comp(e, env)
        if isinstance(e, Lambda):
            return FuncV(e, env, "<lambda>")
        if isinstance(e, New):
            return self.ev_new(e, env)
        if isinstance(e, Star):
            return self.evaluate(e.value, env)
        if isinstance(e, Unknown):
            self.note("assumed", f"unsupported syntax ({e.text}) treated as O(1)")
            return UNKNOWN
        return UNKNOWN

    # ---- arithmetic
    def ev_binop(self, e: BinOp, env: Env) -> Value:
        if e.op == ">>" and self._is_cin_chain(e):
            return self._bind_cin(e, env)
        left = self.evaluate(e.left, env)
        right = self.evaluate(e.right, env)
        return self.binop_value(e.op, left, right)

    @staticmethod
    def _is_cin_chain(e: Expr) -> bool:
        while isinstance(e, BinOp) and e.op == ">>":
            e = e.left
        return isinstance(e, Name) and e.id in ("cin", "std.cin")

    def _bind_cin(self, e: BinOp, env: Env) -> Value:
        targets: list[Expr] = []
        node: Expr = e
        while isinstance(node, BinOp) and node.op == ">>":
            targets.append(node.right)
            node = node.left
        for target in reversed(targets):
            if isinstance(target, Name):
                env.set(target.id, IntV(self.input_size))
        return SCALAR

    def binop_value(self, op: str, left: Value, right: Value) -> Value:  # noqa: C901
        if isinstance(left, IntV) and isinstance(right, IntV):
            a, b = left.mag, right.mag
            if a is None or b is None:
                return IntV(None)
            if op == "+":
                return IntV(a + b)
            if op == "-":
                return IntV(a - b)
            if op == "*":
                return IntV(a * b)
            if op in ("/", "//"):
                k = b.const_value()
                if k is not None and k != 0:
                    return IntV(a / k)
                if len(b.terms) == 1:
                    return IntV(a * b.power(Fraction(-1)))
                return IntV(a)
            if op == "%":
                return IntV(b if b.const_value() is not None else b)
            if op == "**":
                k = b.const_value()
                base = a.const_value()
                if k is not None:
                    return IntV(a.power(k))
                if base is not None and base == 2 and len(b.terms) == 1:
                    var = _single_var(b)
                    if var is not None:
                        return IntV(Poly.exp(var))
                return IntV(None)
            if op == "<<":
                k = b.const_value()
                if k is not None and k >= 0:
                    return IntV(a * (2 ** int(k)))
                base = a.const_value()
                var = _single_var(b)
                if base == 1 and var is not None:
                    return IntV(Poly.exp(var))
                return IntV(None)
            if op == ">>":
                k = b.const_value()
                if k is not None and k >= 0:
                    return IntV(a / (2 ** int(k)))
                return IntV(a)
            if op in ("&", "|", "^"):
                return IntV(a.max(b))
            return IntV(None)
        if isinstance(left, ContV) and isinstance(right, ContV) and op in ("+", "-"):
            if op == "+":
                total = (left.length + right.length).order()
                self._cur.charge(total)
                result = make_container(left.kind, total, left.elem, owned=True)
                self._cur.alloc(total)
                return result
            self._cur.charge(left.length)
            return make_container(left.kind, left.length, left.elem, owned=True)
        if op in ("+", "-") and isinstance(left, ContV) and isinstance(right, IntV):
            # iterator / pointer arithmetic: `nums.begin() + i`, `arr + i` -- a view onto the
            # same storage, whose length stays an upper bound
            return make_container(left.kind, left.length, left.elem, view=True)
        if op == "+" and isinstance(left, IntV) and isinstance(right, ContV):
            return make_container(right.kind, right.length, right.elem, view=True)
        if op == "*":
            container, count = (left, right) if isinstance(left, ContV) else (right, left)
            if isinstance(container, ContV) and isinstance(count, IntV):
                reps = count.mag if count.mag is not None else self.default_size
                length = (container.length * reps).order()
                self._cur.charge(length)
                result = make_container(container.kind, length, container.elem, owned=True)
                self._cur.alloc(length)
                return result
        if isinstance(left, ContV) and left.kind == "str" and op == "%":
            return make_container("str", left.length, owned=True)
        if isinstance(left, IntV) or isinstance(right, IntV):
            return IntV(None)
        return UNKNOWN

    def ev_compare(self, e: Compare, env: Env) -> Value:
        left = self.evaluate(e.left, env)
        right = self.evaluate(e.right, env)
        if e.op in ("in", "not in"):
            if isinstance(right, ContV):
                if right.kind in ("dict", "set"):
                    cost = ONE
                elif right.kind in ("treemap", "treeset"):
                    cost = right.length.log()
                else:
                    cost = right.length
                self._cur.charge(cost)
        elif e.op in ("==", "!=") and isinstance(left, ContV) and isinstance(right, ContV):
            if self._compares_by_value(left, right):
                self._cur.charge(left.length)
        return SCALAR

    def _compares_by_value(self, left: ContV, right: ContV) -> bool:
        """`==` scans both containers in Python and C++ (and strings in Go/JS); an iterator or
        pointer comparison, and Java/JS/C reference equality, is a single machine word."""
        if left.view or right.view:
            return False
        if self.lang in ("python", "cpp"):
            return True
        return self.lang in ("go", "javascript") and left.kind == "str"

    def ev_ifexp(self, e: IfExp, env: Env) -> Value:
        self.evaluate(e.test, env)
        saved = self._cur
        first, second = Bundle(), Bundle()
        self._cur = first
        a = self.evaluate(e.body, env)
        self._cur = second
        b = self.evaluate(e.orelse, env)
        self._cur = saved
        merged = first.either(second)
        self.absorb(merged)
        return join(a, b)

    # ---- structure access
    def ev_attribute(self, e: Attribute, env: Env) -> Value:
        obj = self.evaluate(e.obj, env)
        attr = e.attr
        if isinstance(obj, ContV):
            if attr in ("length", "size", "len"):
                return IntV(obj.length, obj.uid)
            return UNKNOWN
        if isinstance(obj, NodeV):
            if attr in ("children",):
                return make_container("list", obj.size, obj, view=True)
            if attr in NODE_ATTRS:
                return NodeV(obj.size)
            return SCALAR
        if isinstance(obj, TupleV):
            if attr in ("first", "0") and obj.items:
                return obj.items[0]
            if attr in ("second", "1") and len(obj.items) > 1:
                return obj.items[1]
        if isinstance(obj, NoneV):
            return UNKNOWN
        if attr in NODE_ATTRS:
            return UNKNOWN
        return UNKNOWN

    def ev_subscript(self, e: Subscript, env: Env) -> Value:
        obj = self.evaluate(e.obj, env)
        index = self.evaluate(e.index, env)
        if isinstance(obj, ContV):
            if obj.kind == "str":
                return SCALAR
            if obj.kind == "treemap":
                self._cur.charge(obj.length.log())  # an ordered map: log per lookup
            elem = obj.elem
            if elem is None:
                return SCALAR if obj.kind != "dict" else UNKNOWN
            if isinstance(elem, ContV):
                return self.row_for(obj, elem, index)
            return elem
        if isinstance(obj, TupleV):
            if isinstance(index, IntV) and index.mag is not None:
                k = index.mag.const_value()
                if k is not None and 0 <= int(k) < len(obj.items):
                    return obj.items[int(k)]
            return UNKNOWN
        if isinstance(obj, NodeV):
            return NodeV(obj.size)
        return UNKNOWN

    def row_for(self, obj: ContV, elem: ContV, index: Value) -> ContV:
        """`g[i]`: when `i` ranges over every index of `g` the row length is the summable
        per-row atom; otherwise it is the longest row."""
        if obj.owned and elem.owned:
            self.row_owner[elem.uid] = obj.uid
        if isinstance(index, IntV) and index.mag is not None:
            var = _single_var(index.mag)
            info = self.iters.get(var) if var is not None else None
            if (
                info is not None
                and self.is_ragged(obj)
                and (info.domain == obj.uid or self._spans_rows(info, obj))
            ):
                return replace(elem, length=Poly.var(self.rowlen_var(obj.uid)))
        if elem.owned and elem.length.is_zero():
            # rows that start empty and are grown later through `g[i].push(..)` are not tracked
            # one by one: any single row is at most the longest row
            return replace(elem, length=Poly.var(self.cols_var(obj.uid)))
        return elem

    def is_ragged(self, obj: ContV) -> bool:
        """Rows of different lengths (an adjacency list), not a rectangular grid."""
        return obj.ragged or obj.uid in self.ragged_uids

    def _spans_rows(self, info: IterInfo, obj: ContV) -> bool:
        """An index that runs up to the container's own length (or the length it was built with:
        `g = [[] for _ in range(n)]` and `for u in range(n)`) visits at most every row once, so
        the sum of the row lengths it reads is bounded by the total number of entries."""
        sizes = [obj.length]
        if obj.cap is not None and obj.cap in self.cap_initial:
            sizes.append(self.cap_initial[obj.cap])
        if any(not s.is_zero() and info.upper.order() == s.order() for s in sizes):
            if info.domain is None:
                info.domain = obj.uid  # the loop that owns this index now sums over these rows
            return True
        return False

    def ev_slice(self, e: Slice, env: Env) -> Value:
        obj = self.evaluate(e.obj, env)
        lower = self.evaluate(e.lower, env) if e.lower is not None else None
        upper = self.evaluate(e.upper, env) if e.upper is not None else None
        if e.step is not None:
            self.evaluate(e.step, env)
        if not isinstance(obj, ContV):
            return UNKNOWN
        length = obj.length
        lo = magnitude(lower) if lower is not None else ZERO
        hi = magnitude(upper) if upper is not None else None
        if e.step is None and lo is not None:
            span = (hi if hi is not None else obj.length) - lo
            if span.is_zero():
                length = ZERO  # x[:0] and x[i:i] are empty slices
            else:
                length = span.order() if any(c > 0 for _, c in span.terms) else ONE
        if self.lang == "go" and obj.kind in ("list", "array", "str"):
            self._cur.charge(ONE)
            return make_container(obj.kind, length, obj.elem, view=True)
        self._cur.charge(length)
        result = make_container(obj.kind, length, obj.elem, owned=True)
        self._cur.alloc(length)  # a slice copies the slots; the rows it holds stay shared
        return result

    def ev_listlit(self, e: ListLit, env: Env) -> Value:
        items = [self.evaluate(x, env) for x in e.elts]
        spreads = [isinstance(x, Star) for x in e.elts]
        if e.kind == "tuple" and not any(spreads):
            return TupleV(tuple(items))
        length = ZERO
        elem: Value | None = None
        for spread, item in zip(spreads, items, strict=True):
            if spread and isinstance(item, ContV):  # `[...nums]` / `[*nums]` copies every element
                length = length + item.length
                part: Value | None = item.elem
            else:
                length = length + ONE
                part = item if isinstance(item, ContV | NodeV | IntV) else None
            if part is not None:
                elem = part if elem is None else join(elem, part)
        kind = "set" if e.kind == "set" else "list"
        container = make_container(kind, length, elem, owned=True)
        if items:
            self._cur.charge(length)
            self._cur.alloc(length)
        return container

    def ev_new(self, e: New, env: Env) -> Value:  # noqa: C901
        args = [self.evaluate(a, env) for a in e.args]
        dims = [self.evaluate(d, env) for d in e.dims]
        name = e.type.name
        if dims:
            return self._nested_array(dims, 0)
        kind = _container_kind(name)
        if kind is None and (name.endswith("Builder") or name.endswith("Buffer")):
            kind = "str"
        if e.type.dims > 0 and kind is None:
            kind = "array"
        if kind is None:
            if name in ("Array", "Int32Array", "Float64Array", "Uint8Array"):
                kind = "list"
            elif name[:1].isupper() or name == "struct":
                return NodeV(ONE)
            else:
                return UNKNOWN
        length = ZERO
        first = args[0] if args else None
        fill = args[1] if len(args) > 1 else None
        elem: Value | None = None
        if isinstance(first, ContV):
            length = first.length
            self._cur.charge(length)
        elif isinstance(first, IntV) and first.mag is not None:
            if self.lang == "java" and name in _JAVA_CAPACITY_TYPES:
                length = ZERO  # an int argument is a capacity, not a size
            else:
                length = first.mag
                if isinstance(fill, ContV):
                    elem = fill  # `vector<vector<int>> t(n, row)`: n copies of the row
                self._cur.charge(length)
        elif isinstance(first, IntV):
            length = self.default_size
        if e.type.dims > 1:
            elem = make_container("array", ZERO, owned=True)  # `make([][]int, n)`: rows start nil
        elif elem is None and fill is None and isinstance(first, IntV) and e.type.args:
            inner = _container_kind(e.type.args[0].name)
            if inner is not None and kind in ("list", "array", "deque"):
                elem = make_container(inner, ZERO, owned=True)  # `vector<vector<int>> g(n)`
        container = make_container(
            kind, length, elem, owned=True, ragged=isinstance(elem, ContV) and elem.length.is_zero()
        )
        if elem is not None and not length.is_zero():
            self._cur.charge(mem(container))
        self._cur.alloc(mem(container) if not length.is_zero() else ONE)
        return container

    def _nested_array(self, dims: list[Value], depth: int) -> Value:
        mag = magnitude(dims[depth])
        length = mag if mag is not None else self.default_size
        elem: Value | None = None
        if depth + 1 < len(dims):
            elem = self._nested_array(dims, depth + 1)
        container = make_container("array", length, elem, owned=True)
        if depth == 0:
            total = mem(container)
            self._cur.charge(total)
            self._cur.alloc(total)
        return container

    def ev_comp(self, e: Comp, env: Env) -> Value:
        inner_env = env.copy()
        body: tuple[Stmt, ...] = (ExprStmt(e.elt),)
        count_body: tuple[Stmt, ...] = (ExprStmt(Num(1)),)

        def nest(inner: tuple[Stmt, ...]) -> Loop:
            wrapped = inner
            for generator in reversed(e.generators):
                if generator.conds:
                    cond = (
                        generator.conds[0]
                        if len(generator.conds) == 1
                        else BoolOp("and", tuple(generator.conds))
                    )
                    wrapped = (If(cond, wrapped, ()),)
                wrapped = (
                    Loop(
                        kind="for_each", target=generator.target, iter=generator.iter, body=wrapped
                    ),
                )
            loop = wrapped[0]
            assert isinstance(loop, Loop)
            return loop

        elt_value: Value = SCALAR
        with self.scratch():
            counted = self.exec_stmt(nest(count_body), inner_env.copy())
        count = counted.time.total()
        stmt = nest(body)
        self._last_value = SCALAR
        self.absorb(self.exec_stmt(stmt, inner_env))
        elt_value = self._last_value
        count = count.order() if not count.is_zero() else ONE
        if e.kind == "gen":
            return make_container("list", count, elt_value, view=True)
        kind = {"list": "list", "set": "set", "dict": "dict"}[e.kind]
        elem = elt_value if isinstance(elt_value, ContV | NodeV | IntV) else None
        container = make_container(
            kind, count, elem, owned=True, ragged=isinstance(elem, ContV) and elem.length.is_zero()
        )
        self._cur.alloc(mem(container))
        return container

    # ---- calls
    def ev_call(self, call: Call, env: Env) -> Value:  # noqa: C901
        func = call.func
        if (
            isinstance(func, Attribute)
            and func.attr == "from"
            and isinstance(func.obj, Name)
            and func.obj.id == "Array"
        ):
            return self.ev_array_from(call, env)
        target = self.resolve_user(func, env)
        if target is not None:
            args = [self.evaluate(a, env) for a in call.args]
            for _, kwarg in call.kwargs:
                self.evaluate(kwarg, env)
            return self.call_user(target, args)
        receiver: Value | None = None
        qualifier: str | None = None
        name: str
        if isinstance(func, Attribute):
            name = func.attr
            obj = func.obj
            if isinstance(obj, Name) and env.get(obj.id) is None and obj.id not in _SELF_NAMES:
                qualifier = obj.id
            elif isinstance(obj, Attribute) and _is_qualified_chain(obj, env):
                qualifier = obj.attr
            else:
                receiver = self.evaluate(obj, env)
        elif isinstance(func, Name):
            name = func.id
        else:
            callee = self.evaluate(func, env)
            args = [self.evaluate(a, env) for a in call.args]
            if isinstance(callee, FuncV):
                return self.call_user(callee, args)
            return UNKNOWN
        if (
            name in _HOF_METHODS
            and isinstance(receiver, ContV)
            and self._callback(call, env) is not None
        ):
            return self.ev_hof(name, call, receiver, env)
        args = [self.evaluate(a, env) for a in call.args]
        kwargs = {k: self.evaluate(v, env) for k, v in call.kwargs}
        lib = LibCall(
            lang=self.lang,
            name=name,
            qualifier=qualifier,
            receiver=receiver,
            args=args,
            arg_exprs=list(call.args),
            kwargs=kwargs,
            default_size=self.default_size,
            input_size=self.input_size,
        )
        result = call_library(lib)
        if result is None:
            self.note("assumed", f"call to {name}() assumed O(1)")
            return UNKNOWN
        if result.assumed:
            self.note("assumed", result.assumed)
        self._cur.charge(result.time)
        if result.alloc is not None:
            self._cur.alloc(result.alloc)
        subject = lib.subject()
        if result.grow is not None and subject is not None and subject.cap is not None:
            self._cur.grow(subject.cap, result.grow)
        elif result.grow is not None and subject is not None and subject.uid in self.row_owner:
            self._cur.grow(self.total_var(self.row_owner[subject.uid]), result.grow)
        if result.reset and subject is not None and subject.cap is not None:
            self._cur.reset(subject.cap)
        if result.grow is not None:
            for arg in args:
                if isinstance(arg, ContV) and arg.owned and arg is not subject:
                    self._cur.retain(mem(arg))
        if (
            result.set_receiver is not None
            and isinstance(func, Attribute)
            and isinstance(func.obj, Name)
            and env.get(func.obj.id) is receiver
        ):
            env.set(func.obj.id, result.set_receiver)  # `g.add(new ArrayList<>())` adds a row
        for index, bound in result.binds_args.items():
            if index < len(call.args):
                target_expr = call.args[index]
                if isinstance(target_expr, Name):
                    env.set(target_expr.id, bound)
        value = result.value
        if (
            name in _ROW_GETTERS
            and isinstance(receiver, ContV)
            and isinstance(receiver.elem, ContV)
            and value is receiver.elem
            and args
        ):
            value = self.row_for(receiver, receiver.elem, args[0])  # `graph.get(u)` is `graph[u]`
        return value

    def _callback(self, call: Call, env: Env) -> Expr | None:
        """The function-valued argument of a call (`xs.map(row => ...)`), if there is one."""
        for arg in call.args:
            if isinstance(arg, Lambda):
                return arg
            if isinstance(arg, Name) and self.resolve_user(arg, env) is not None:
                return arg
        return None

    def ev_hof(self, name: str, call: Call, receiver: ContV, env: Env) -> Value:
        """`xs.map(f)`, `xs.filter(f)`, `xs.forEach(f)`, `xs.reduce(f, init)`: the callback runs
        once per element, so the call is a loop over `xs` whose body is the callback."""
        callback = self._callback(call, env)
        assert callback is not None
        params = len(callback.params) if isinstance(callback, Lambda) else 1
        env.set(_HOF_RECEIVER, receiver)
        env.set(_HOF_ACC, SCALAR)
        item = Name(_HOF_ITEM)
        passed = (Name(_HOF_ACC), item) if params >= 2 and name in _HOF_FOLDS else (item,)
        loop = Loop(
            kind="for_each",
            target=item,
            iter=Name(_HOF_RECEIVER),
            body=(ExprStmt(Call(callback, passed)),),
        )
        self._last_value = SCALAR
        try:
            self.absorb(self.exec_stmt(loop, env))
        finally:
            env.vars.pop(_HOF_RECEIVER, None)
            env.vars.pop(_HOF_ACC, None)
            env.vars.pop(_HOF_ITEM, None)
        produced = self._last_value
        length = receiver.length
        if name in ("map", "mapToInt", "mapToObj", "mapToLong", "mapToDouble", "flatMap"):
            elem: Value | None = produced if isinstance(produced, ContV | NodeV | IntV) else None
            if name == "flatMap" and isinstance(produced, ContV):
                length = (length * produced.length).order()
                elem = produced.elem
            result = make_container("list", length, elem, owned=True)
            # rows the callback built are kept by the result; rows it merely returned are shared
            self._cur.alloc(mem(result) if isinstance(elem, ContV) and elem.owned else length)
            self._cur.charge(length)
            return result
        if name in ("filter", "removeIf", "takeWhile", "dropWhile"):
            result = make_container("list", length, receiver.elem, owned=True)
            self._cur.alloc(length)
            return result
        if name in ("reduce", "reduceRight"):
            return produced if produced is not UNKNOWN else SCALAR
        if name in ("find", "findLast"):
            return receiver.elem if receiver.elem is not None else SCALAR
        return SCALAR

    def ev_array_from(self, call: Call, env: Env) -> Value:
        length = self.default_size
        first = call.args[0] if call.args else None
        if isinstance(first, DictLit):
            for key, value in first.items:
                if isinstance(key, Str) and key.value == "length":
                    mag = magnitude(self.evaluate(value, env))
                    if mag is not None:
                        length = mag
        elif first is not None:
            source = self.evaluate(first, env)
            if isinstance(source, ContV):
                length = source.length
        elem: Value | None = None
        if len(call.args) > 1:
            mapper = self.evaluate(call.args[1], env)
            if isinstance(mapper, FuncV) and isinstance(mapper.func, Lambda):
                lam_env = Env({p: SCALAR for p in mapper.func.params}, mapper.closure)  # type: ignore[arg-type]
                if isinstance(mapper.func.body, tuple):
                    elem = None
                else:
                    elem = self.pure(mapper.func.body, lam_env)
        self._cur.charge(length)
        container = make_container(
            "list", length, elem if isinstance(elem, ContV | NodeV) else None, owned=True
        )
        self._cur.alloc(mem(container))
        return container

    def resolve_user(self, func: Expr, env: Env) -> FuncDef | FuncV | None:
        if isinstance(func, Name):
            value = env.get(func.id)
            if isinstance(value, FuncV):
                return value
            if value is None:
                return self.lookup_function(func.id)
            return None
        if isinstance(func, Attribute) and isinstance(func.obj, Name):
            if func.obj.id in _SELF_NAMES or func.obj.id in self.class_names:
                return self.lookup_function(func.attr)
        return None

    def call_user(self, target: FuncDef | FuncV, args: list[Value]) -> Value:
        if isinstance(target, FuncV):
            if target.closure is None and isinstance(target.func, FuncDef):
                return self.call_user(target.func, args)
            return self._call_closure(target, args)
        if any(target is active for active in self._active):
            self._cur.add_rec(RecCall(target.name, tuple(args), ONE, False, target.line))
            return UNKNOWN
        if self._depth >= MAX_CALL_DEPTH:
            self.note("unknown", "call chain too deep to analyse")
            return UNKNOWN
        self._depth += 1
        try:
            summary = self.summary_for(target)
        finally:
            self._depth -= 1
        mapping = self.bind_args(summary.params, args)
        self._cur.charge(summary.time.substitute(mapping))
        self._cur.alloc(summary.space.substitute(mapping))
        if summary.recursive and not summary.solved:
            self.note("unknown", f"recursion in {target.name}() could not be solved")
        return substitute(summary.ret, mapping) or UNKNOWN

    def _call_closure(self, target: FuncV, args: list[Value]) -> Value:
        func = target.func
        closure = target.closure if isinstance(target.closure, Env) else None
        if isinstance(func, Lambda):
            env = Env({}, closure)
            for lambda_param, actual in zip(func.params, args, strict=False):
                env.set(lambda_param, actual)
            if isinstance(func.body, tuple):
                saved_returns = self._returns
                self._returns = []
                self.absorb(self.exec_block(func.body, env))
                result = self._returns[0] if self._returns else UNKNOWN
                self._returns = saved_returns
                return result
            return self.evaluate(func.body, env)
        assert isinstance(func, FuncDef)
        if any(func is active for active in self._active):
            self._cur.add_rec(RecCall(func.name, tuple(args), ONE, False, func.line))
            return UNKNOWN
        if self._depth >= MAX_CALL_DEPTH:
            self.note("unknown", "call chain too deep to analyse")
            return UNKNOWN
        env = Env({}, closure)
        for param, actual in zip(func.params, args, strict=False):
            env.set(param.name, actual)
        self._active.append(func)
        saved = (self._returns, self._loop_stack)
        self._returns, self._loop_stack = [], []
        self._depth += 1
        try:
            body = self.exec_block(func.body, env)
            body = self.finalize_caps(body, body.created)
            ret: Value | None = None
            for value in self._returns:
                ret = join(ret, value) if ret is not None else value
        finally:
            self._depth -= 1
            self._active.pop()
            self._returns, self._loop_stack = saved
        if any(alt for alt in body.rec):
            from analysis.recurrence import solve_nested

            return solve_nested(self, func, args, body, ret)
        self.absorb(body)
        return ret if ret is not None else UNKNOWN


# =================================================================================== helpers
def _single_var(poly: Poly) -> Var | None:
    """The variable of a polynomial that is exactly one variable with coefficient 1."""
    if len(poly.terms) != 1:
        return None
    mono, coef = poly.terms[0]
    if coef != 1 or len(mono.pows) != 1 or mono.exps or mono.facts:
        return None
    var, power, log = mono.pows[0]
    return var if power == 1 and log == 0 else None


def _is_qualified_chain(expr: Expr, env: Env) -> bool:
    """`java.util.Arrays`, `os.path`: an attribute chain rooted at a name that is not a variable."""
    node = expr
    while isinstance(node, Attribute):
        node = node.obj
    return isinstance(node, Name) and env.get(node.id) is None and node.id not in _SELF_NAMES


def assigned_names(stmts: tuple[Stmt, ...]) -> set[str]:
    """Names (re)bound anywhere in `stmts`, including nested loops' targets and aug-assigns."""
    names: set[str] = set()

    def collect(target: Expr) -> None:
        if isinstance(target, Name):
            names.add(target.id)
        elif isinstance(target, ListLit):
            for sub in target.elts:
                collect(sub)

    for stmt in stmts:
        for node in walk(stmt):
            if isinstance(node, Assign):
                for target in node.targets:
                    collect(target)
            elif isinstance(node, Loop):
                if node.target is not None:
                    collect(node.target)
                for init in (*node.init, *node.update):
                    if isinstance(init, Assign):
                        for target in init.targets:
                            collect(target)
    return names


__all__ = [
    "FuncSummary",
    "Interp",
    "IterInfo",
    "LoopFrame",
    "Note",
    "ScalarV",
    "UnknownV",
    "assigned_names",
]
