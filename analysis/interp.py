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

from analysis.bundle import Bundle, Cond, Cost, RecCall
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
_NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
_SIZE_OPS = frozenset(_NEGATED)
_EMPTY_TESTS = frozenset({"empty", "isEmpty", "is_empty", "isempty"})
_KEYED_INSERTS = frozenset(
    {"add", "put", "insert", "emplace", "set", "setdefault", "putIfAbsent", "computeIfAbsent",
     "merge", "compute", "computeIfPresent", "insert_or_assign", "try_emplace", "emplace_hint"}
)  # fmt: skip
_HOF_RECEIVER, _HOF_ITEM, _HOF_ACC = "<hof receiver>", "<hof item>", "<hof acc>"
_REMOVERS = frozenset(
    {"pop", "popleft", "poll", "pollFirst", "pollLast", "remove", "dequeue", "shift",
     "pop_front", "pop_back", "removeFirst", "removeLast", "Pop", "heappop", "erase", "discard",
     "delete"}
)  # fmt: skip
_GROWABLE_KINDS = frozenset(
    {"list", "deque", "set", "dict", "treemap", "treeset", "queue", "stack", "heap"}
)


@dataclass(frozen=True)
class Note:
    kind: str  # "assumed": a bound was assumed;  "unknown": no meaningful answer
    reason: str
    line: int = 0


@dataclass(frozen=True)
class StepRec:
    """One line of the derivation. The polynomials are rendered once the display names of the
    variables are known, so every step uses the same letters as the answer."""

    line: int
    kind: str  # loop | call | recursion | alloc
    fmt: str  # `{0}`, `{1}` stand for the rendered polynomials
    polys: tuple[Poly, ...] = ()
    key: str = ""  # tells apart several steps on one line


_MAX_STEPS = 4000  # a hostile file must not grow the trace without bound


@dataclass
class IterInfo:
    """What is known about an induction variable (an `iter` atom)."""

    kind: str  # arith | geom
    upper: Poly  # upper limit of the values the variable takes
    domain: int | None = None  # uid of the container whose indices (all of them) it ranges over
    lower: Poly | None = None  # first value it takes (None: 0)


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
    # calls this function makes to a function that is still being analysed further out (mutual
    # recursion): the caller adopts them as its own recursive calls
    outer_rec: list[tuple[RecCall, ...]] = field(default_factory=list)
    # cost shared by every call of this function through one container, by parameter name: a
    # visited array or memo table is filled once however many calls reach it
    shared: dict[str, Poly] = field(default_factory=dict)
    closure: Env | None = None  # where a nested function finds the variables it captured
    shared_free: dict[int, Poly] = field(default_factory=dict)  # the same, by captured container
    how: str = ""  # how the recursion was solved, for the derivation
    # what ONE call adds to containers that outlive it (an output list handed in, a captured
    # result list), by the container's growth variable; for a recursion, all its calls together
    effects: dict[Var, Poly] = field(default_factory=dict)
    param_caps: dict[str, Var] = field(default_factory=dict)  # parameter name -> growth variable


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
        self.steps: list[StepRec] = []
        self._line = 0  # line of the statement being executed
        self._cur = Bundle()
        self._ids = itertools.count(1)
        self.iters: dict[Var, IterInfo] = {}
        self.monotone: dict[Var, int] = {}  # progress atoms added by nested monotone loops
        self.cap_initial: dict[Var, Poly] = {}
        self.cap_override: dict[Var, Poly] = {}
        self.cap_distinct: dict[Var, Poly] = {}  # a hash container holds at most this many keys
        self.cap_unbounded: set[Var] = set()  # ...unless some key comes from data
        self.rowlen_vars: dict[int, Var] = {}
        self.total_vars: dict[int, Var] = {}
        self.cols_vars: dict[int, Var] = {}
        self.row_owner: dict[int, int] = {}  # row uid -> uid of the container whose rows they are
        self.autovivify: set[int] = set()  # defaultdicts: reading a missing key inserts it
        self._links_nodes = False  # the function being analysed links nodes through `.next`...
        self._pushed: set[Var] = set()  # containers the function being analysed adds to...
        self._popped: set[Var] = set()  # ...and removes from (a path that goes up and down)
        self.ragged_uids: set[int] = set()  # containers a traversal proves to be adjacency lists
        self._loop_stack: list[LoopFrame] = []
        self._conds: list[tuple[Expr, bool]] = []  # tests known true / false on this path
        self._guessed: dict[Var, str] = {}  # argument sizes bind_args had to assume
        self.kids_vars: dict[Var, Var] = {}  # node size variable -> children of one node
        self._rec_returns: dict[int, tuple[dict[str, Value], Value | None]] = {}
        self._shared_pool: dict[int, Poly] = {}  # container uid -> cost shared through it
        self._returns: list[Value] = []
        self._summaries: dict[tuple[int, int], FuncSummary] = {}
        self._active: list[FuncDef] = []
        self._cur_func: FuncDef | None = None
        self._last_value: Value = UNKNOWN
        self._depth = 0

    # ================================================================== utilities
    def note(self, kind: str, reason: str, line: int = 0) -> None:
        note = Note(kind, reason, line or self._line)
        if note not in self.notes:
            self.notes.append(note)

    def step(self, kind: str, fmt: str, *polys: Poly, line: int = 0, key: str = "") -> None:
        if len(self.steps) < _MAX_STEPS:
            self.steps.append(StepRec(line or self._line, kind, fmt, polys, key))

    def note_alloc(self, amount: Poly, what: str) -> None:
        """Record a sizeable allocation in the derivation (constant ones are not worth a line)."""
        if amount.vars():
            self.step("alloc", f"{what} allocates {{0}}", amount, key=what)

    @contextmanager
    def scratch(self) -> Iterator[Bundle]:
        """Evaluate without charging the enclosing statement or recording notes or steps."""
        saved_cur, saved_notes, saved_steps = self._cur, len(self.notes), len(self.steps)
        self._cur = Bundle()
        try:
            yield self._cur
        finally:
            self._cur = saved_cur
            del self.notes[saved_notes:]
            del self.steps[saved_steps:]

    def pure(self, expr: Expr, env: Env) -> Value:
        with self.scratch():
            return self.evaluate(expr, env)

    def absorb(self, other: Bundle) -> None:
        """Merge `other` into the bundle currently being built, in place."""
        cur = self._cur
        merged = cur.then(other)
        cur.time, cur.grows, cur.allocs = merged.time, merged.grows, merged.allocs
        cur.retained, cur.created, cur.rec = merged.retained, merged.created, merged.rec

    def kids_var(self, size: Poly) -> Var:
        """How many children one node has. Summed over every node of a tree it is size - 1, which
        is what makes a recursion over node.children linear in the nodes."""
        key = _single_var(size)
        if key is None:
            return new_var("children", "rowlen")
        if key not in self.kids_vars:
            self.kids_vars[key] = new_var("children", "rowlen")
        return self.kids_vars[key]

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

    def summary_for(self, fdef: FuncDef, closure: Env | None = None) -> FuncSummary:
        key = (id(fdef), id(closure))
        if key in self._summaries:
            return self._summaries[key]
        summary = self._summarise(fdef, closure)
        if (
            not summary.outer_rec
        ):  # one that leans on a function still being analysed is not reusable
            self._summaries[key] = summary
        return summary

    def _summarise(self, fdef: FuncDef, closure: Env | None) -> FuncSummary:
        self._active.append(fdef)
        saved = (
            self._cur,
            self._returns,
            self._loop_stack,
            self._conds,
            self._cur_func,
            self.default_size,
            self._pushed,
            self._popped,
            self._links_nodes,
        )
        self._cur_func = fdef
        notes_before, steps_before = len(self.notes), len(self.steps)
        try:
            params, body, ret = self._run_body(fdef, closure)
            if self._calls_itself(fdef, body) and id(fdef) not in self._rec_returns:
                # second pass: a recursive call returns what the first pass learned the function
                # returns, so `len(merge_sort(left))` is a length and not an unknown
                self._rec_returns[id(fdef)] = (params, ret)
                del self.notes[notes_before:]
                del self.steps[steps_before:]
                self._shared_pool.clear()
                params, body, ret = self._run_body(fdef, closure)
            summary = self._make_summary(fdef, params, body, ret, closure)
        finally:
            self._active.pop()
            self._rec_returns.pop(id(fdef), None)
            (
                self._cur,
                self._returns,
                self._loop_stack,
                self._conds,
                self._cur_func,
                self.default_size,
                self._pushed,
                self._popped,
                self._links_nodes,
            ) = saved
        return summary

    def _run_body(
        self, fdef: FuncDef, closure: Env | None = None
    ) -> tuple[dict[str, Value], Bundle, Value | None]:
        self._cur, self._returns, self._loop_stack, self._conds = Bundle(), [], [], []
        self._pushed, self._popped = set(), set()
        self._links_nodes = _links_nodes(fdef)
        params = params_to_values(fdef, self.lang, self._callee_table)
        self.default_size = self._first_size(params)
        self._give_growth_vars(fdef, params)
        self._copy_by_value_params(fdef, params)
        prologue = self._cur  # what happens before the first statement: copying the arguments
        env = Env(dict(params), closure)
        body = prologue.then(self.exec_block(fdef.body, env))
        # entries pushed into the rows of a container built here add up to its total
        body.created |= {v for v in body.grows if v.kind == "total"}
        retired = self.resolve_caps(body, body.created) if body.created else {}
        body = self.finalize_caps(body, body.created)
        if retired:  # sizes of containers built here are now known: resolve what is pooled on them
            self._shared_pool = {u: p.substitute(retired) for u, p in self._shared_pool.items()}
        ret: Value | None = None
        for value in self._returns:
            ret = join(ret, value) if ret is not None else value
        if ret is not None and retired:
            fixed = substitute(ret, retired)
            if isinstance(fixed, ContV) and fixed.cap in retired:
                fixed = replace(fixed, cap=None)
            ret = fixed
        return params, body, ret

    def _give_growth_vars(self, fdef: FuncDef, params: dict[str, Value]) -> None:
        """Adding to a container the caller handed in is visible to the caller. Each such
        parameter gets a growth variable, so what a call adds to it can be charged to the
        container the caller passed (a C++ value parameter is a copy, a Go slice a header, a C
        array cannot grow: none of those are shared)."""
        if self.lang == "c":
            return
        for param in fdef.params:
            value = params.get(param.name)
            if not isinstance(value, ContV) or value.cap is not None or value.view:
                continue
            if value.kind not in _GROWABLE_KINDS:
                continue
            if self.lang == "cpp" and (
                param.type is None or not (param.type.ref or param.type.ptr)
            ):
                continue
            if self.lang == "go" and value.kind not in ("dict", "treemap"):
                continue
            params[param.name] = replace(value, cap=new_var(param.name, "pgrow"))

    def _copy_by_value_params(self, fdef: FuncDef, params: dict[str, Value]) -> None:
        """A C++ container parameter that is not a reference or pointer is COPIED for the call:
        O(size) time and space in every frame (a recursion that passes a vector by value holds one
        copy per level). Python, Java, JavaScript and Go pass references or slice headers."""
        if self.lang != "cpp":
            return
        for param in fdef.params:
            value = params.get(param.name)
            typ = param.type
            if (
                typ is not None
                and isinstance(value, ContV)
                and not typ.ref
                and typ.ptr == 0
                and typ.dims == 0
            ):
                size = mem(value)
                self._cur.charge(size)
                self._cur.alloc(size)
                self.note(
                    "info", f"{param.name} is taken by value: copied on every call", fdef.line
                )

    @staticmethod
    def _calls_itself(fdef: FuncDef, body: Bundle) -> bool:
        return any(c.callee == fdef.name for alt in body.rec for c in alt)

    def _first_size(self, params: dict[str, Value]) -> Poly:
        for value in params.values():
            if isinstance(value, ContV):
                return value.length
            if isinstance(value, IntV) and value.mag is not None:
                return value.mag
            if isinstance(value, NodeV):
                return value.size
        return self.input_size

    def _make_summary(
        self,
        fdef: FuncDef,
        params: dict[str, Value],
        body: Bundle,
        ret: Value | None,
        closure: Env | None = None,
    ) -> FuncSummary:
        recursive = self._calls_itself(fdef, body)
        outer = (
            []
            if recursive
            else [alt for alt in body.rec if alt and all(c.callee != fdef.name for c in alt)]
        )
        time = body.time.total()
        space = (body.allocs + body.retained.total()).order()
        balanced = self._pushed & self._popped  # up and down again: no net growth
        effects = {
            cap: growth.total().order()
            for cap, growth in body.grows.items()
            if cap not in balanced and not growth.total().is_zero()
        }
        summary = FuncSummary(
            func=fdef,
            params=params,
            time=time.order() if not time.is_zero() else ONE,
            space=space if not space.is_zero() else ONE,
            ret=ret,
            rec=body.rec,
            recursive=recursive,
            solved=not recursive,
            outer_rec=outer,
            closure=closure,
            effects=effects,
            param_caps={
                name: value.cap
                for name, value in params.items()
                if isinstance(value, ContV) and value.cap is not None and value.cap.kind == "pgrow"
            },
        )
        self._settle_shared(summary)
        if recursive:
            from analysis.recurrence import solve_recurrence

            solve_recurrence(self, summary, body)
            self._scale_effects(summary, body)
            self._step_recursion(summary)
        return summary

    def _scale_effects(self, summary: FuncSummary, body: Bundle) -> None:
        """A recursion adds to its output container on every call: its effects are one call's
        effects times the number of calls (the recurrence solved with one unit of work each)."""
        if not summary.effects or not summary.solved:
            return
        if summary.how.startswith("every distinct state"):
            # a memoised / visited-guarded recursion adds to its domain container once per
            # distinct state however often it is called: that is the domain's own size, not a
            # per-call effect (the container's final size is bounded where it is created)
            summary.effects = {}
            return
        from analysis.recurrence import solve_recurrence

        probe = replace(summary, time=ONE, space=ONE, shared={}, shared_free={}, effects={})
        solve_recurrence(self, probe, body)
        if not probe.solved:
            return
        calls = probe.time
        for extra in (*probe.shared.values(), *probe.shared_free.values()):
            calls = calls + extra
        calls = calls.order() if not calls.is_zero() else ONE
        summary.effects = {cap: (e * calls).order() for cap, e in summary.effects.items()}

    def _step_recursion(self, summary: FuncSummary) -> None:
        name, line = summary.func.name, summary.func.line
        if summary.solved:
            self.step(
                "recursion",
                f"{name}() calls itself ({summary.how or 'solved as a recurrence'}): "
                "time {0}, space {1}",
                summary.time,
                summary.space,
                line=line,
                key=name,
            )
        else:
            self.step(
                "recursion",
                f"{name}() calls itself; the recursion could not be bounded",
                line=line,
                key=name,
            )

    def bind_args(self, params: dict[str, Value], args: list[Value]) -> dict[Var, Poly]:
        mapping: dict[Var, Poly] = {}
        self._guessed.clear()
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
                    self._guessed[var] = f"size of argument for {var.name} unknown"
        elif isinstance(formal, ContV):
            var = _single_var(formal.length)
            if isinstance(actual, ContV) and isinstance(formal.elem, ContV):
                # the table total of the caller stands in for the table total of the callee
                mapping[self.total_var(formal.uid)] = Poly.var(self.total_var(actual.uid))
            if var is not None:
                if isinstance(actual, ContV):
                    mapping[var] = actual.length
                elif isinstance(actual, TupleV):
                    mapping[var] = Poly.const(len(actual.items))
                elif isinstance(actual, IntV) and actual.mag is not None:
                    mapping[var] = actual.mag
                else:
                    mapping[var] = self.default_size
                    self._guessed[var] = f"size of argument for {var.name} unknown"
            if formal.elem is not None:
                nested = actual.elem if isinstance(actual, ContV) else None
                if (
                    isinstance(formal.elem, ContV)
                    and isinstance(nested, ContV)
                    and nested.owned
                    and nested.length.is_zero()
                ):
                    # rows that start empty and are filled in later (by the callee, as a merge
                    # sort tree fills its nodes): they are not empty when the callee reads them
                    inner = _single_var(formal.elem.length)
                    if inner is not None:
                        mapping[inner] = self.default_size
                        self._guessed[inner] = (
                            "the rows of a table the callee fills in are assumed as long as "
                            "the input"
                        )
                elif nested is not None:
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
                    self._guessed[var] = "size of a linked-structure argument unknown"

    # ================================================================== statements
    def exec_block(self, stmts: tuple[Stmt, ...] | list[Stmt], env: Env) -> Bundle:
        total = Bundle()
        for index, stmt in enumerate(stmts):
            if isinstance(stmt, If):
                dead_then, dead_else = self.always_exits(stmt.body), self.always_exits(stmt.orelse)
                if dead_then != dead_else:
                    # a guard clause (`if n <= 1: return 1`): what follows runs only on the other
                    # path, so a call after it is an alternative to the one inside, not an addition
                    head, dead, live = self.exec_if_split(stmt, env)
                    self._conds.append((stmt.test, dead_else))
                    try:
                        rest = self.exec_block(stmts[index + 1 :], env)
                    finally:
                        self._conds.pop()
                    return total.then(head.then(dead.either(live.then(rest))))
            total = total.then(self.exec_stmt(stmt, env))
        return total

    def exec_stmt(self, st: Stmt, env: Env) -> Bundle:
        saved, saved_line = self._cur, self._line
        self._line = getattr(st, "line", 0) or saved_line
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
            self._cur, self._line = saved, saved_line

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
        head, then_b, else_b = self.exec_if_split(st, env)
        return b.then(head).then(then_b.either(else_b))

    def exec_if_split(self, st: If, env: Env) -> tuple[Bundle, Bundle, Bundle]:
        """Interpret an `if`: the cost of its test, of the branch that exits (or the then branch),
        and of the other one. The environment afterwards is that of the branch that can fall
        through (or the merge of both)."""
        head = Bundle()
        saved = self._cur
        self._cur = head
        try:
            head.charge(ONE)
            self.evaluate(st.test, env)
        finally:
            self._cur = saved
        env_then, env_else = env.copy(), env.copy()
        self._conds.append((st.test, True))
        try:
            then_b = self.exec_block(st.body, env_then)
        finally:
            self._conds.pop()
        self._conds.append((st.test, False))
        try:
            else_b = self.exec_block(st.orelse, env_else)
        finally:
            self._conds.pop()
        dead_then, dead_else = self.always_exits(st.body), self.always_exits(st.orelse)
        if dead_then and not dead_else:
            merged = env_else
            parts = (head, then_b, else_b)
        elif dead_else and not dead_then:
            merged = env_then
            parts = (head, else_b, then_b)
        else:
            merged = env_then.merge(env_else)
            parts = (head, then_b, else_b)
        env.vars.clear()
        env.vars.update(merged.vars)
        return parts

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
                    stored = mem(value)
                    self._cur.grow(container.cap, stored if not stored.is_zero() else ONE)
                    self._note_key(container.cap, self.pure(target.index, env))
                restored = isinstance(container.elem, ContV) and (
                    isinstance(value, ContV) and value.uid == container.elem.uid
                )  # `g[u] = append(g[u], v)` stores the very row it read: nothing new retained
                if isinstance(value, ContV) and value.owned and not restored:
                    self._cur.retain(mem(value))
                if (
                    isinstance(value, ContV)
                    and not restored
                    and isinstance(target.obj, Name)
                    and container.kind in ("list", "array", "dict", "treemap")
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
    def resolve_caps(
        self, bundle: Bundle, caps: set[Var], skip_lump: int | None = None
    ) -> dict[Var, Poly]:
        """The final size of each growable container. With `skip_lump` the growth that belongs to
        that loop as a whole (its amortised part) is left out: what is left is what a single
        iteration adds by itself."""
        values: dict[Var, Poly] = {}
        for _ in range(8):
            changed = False
            for cap in caps:
                growth = bundle.grows.get(cap)
                added = ZERO
                if growth is not None:
                    added = growth.own
                    for key, lump in growth.lumps.items():
                        if key != skip_lump:
                            added = added + lump
                total = self.cap_initial.get(cap, ZERO) + added
                value = total.substitute(values)
                value = value.order() if not value.is_zero() else ZERO
                if cap in self.cap_override:
                    value = self.cap_override[cap].substitute(values)
                    value = value.order() if not value.is_zero() else ZERO
                elif cap in self.cap_distinct and cap not in self.cap_unbounded:
                    # a set / dict never holds more entries than there are distinct keys: adding
                    # the row index i for every cell of an n x m grid stores n entries, not n * m
                    capped = (self.cap_initial.get(cap, ZERO) + self.cap_distinct[cap]).substitute(
                        values
                    )
                    capped = capped.order() if not capped.is_zero() else ZERO
                    if value.max(capped).order() == value.order() != capped.order():
                        value = capped
                if values.get(cap) != value:
                    values[cap] = value
                    changed = True
            if not changed:
                break
        cyclic = {c for c, v in values.items() if v.vars() & caps}
        for cap in cyclic:
            values[cap] = self.default_size
            self.note("assumed", f"size of container {cap.name} depends on itself; assumed linear")
        for cap, size in list(values.items()):
            if skip_lump is None and size.is_zero() and self._paid_for(bundle, cap):
                values[cap] = self.default_size
                self.note(
                    "assumed",
                    f"the size of `{cap.name}` could not be tracked; assumed proportional to the "
                    "input",
                )
        return {c: (v if not v.is_zero() else ZERO) for c, v in values.items()}

    @staticmethod
    def _paid_for(bundle: Bundle, cap: Var) -> bool:
        """Does the time of `bundle` depend on this container's size (a loop over it)?"""
        if bundle.allocs.mentions(cap):
            return True
        for cost in (bundle.time, bundle.retained, *bundle.grows.values()):
            if cost.own.mentions(cap) or any(lump.mentions(cap) for lump in cost.lumps.values()):
                return True
        return False

    def finalize_caps(self, bundle: Bundle, caps: set[Var], loop_id: int | None = None) -> Bundle:
        if not caps:
            return bundle
        values = self.resolve_caps(bundle, caps)
        # A container built in a loop body whose growth is amortised over the whole loop (a level
        # list filled from a shared queue) is only as big as one round makes it, and all rounds
        # together add up to the amortised total. A cost per round that mentions its size is paid
        # in full once, not once per round: split it into the round's own part and that lump.
        split = loop_id is not None and any(
            loop_id in bundle.grows[c].lumps for c in caps if c in bundle.grows
        )
        own_values = self.resolve_caps(bundle, caps, loop_id) if split else values

        def sub_cost(cost: Cost) -> Cost:
            full = cost.own.substitute(values) if not cost.own.is_zero() else cost.own
            part = cost.own.substitute(own_values) if split and not cost.own.is_zero() else full
            lumps = {k: v.substitute(values) for k, v in cost.lumps.items()}
            if split and loop_id is not None and not (full - part).is_zero():
                extra = (full - part).order()
                lumps[loop_id] = (lumps[loop_id] + extra).order() if loop_id in lumps else extra
            return Cost(part.order() if not part.is_zero() else part, lumps)

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
        test_cost = self._test_cost(st, body_env)
        if test_cost is not None:
            overhead.charge(test_cost)
        inner = overhead.then(inner)
        retired = self.resolve_caps(inner, inner.created) if inner.created else {}
        inner = self.finalize_caps(inner, inner.created, loop_id)
        inner = self.apply_flushes(inner)
        inner = self._bound_allocs(inner, plan)
        folded = self.fold_loop(inner, plan, loop_id)
        if plan.amortize_into is not None and plan.amortized_total is not None:
            self.step(
                "loop",
                "loop runs {0} times in total across the enclosing loop (amortised)",
                plan.amortized_total,
                line=st.line,
            )
        else:
            self.step(
                "loop",
                "loop runs {0} times; the whole loop costs {1}",
                plan.iters,
                folded.time.own,
                line=st.line,
            )
        self._post_loop_env(st, env, body_env, plan, assigned)
        if retired:
            self._retire_caps(env, retired)
        return b.then(folded)

    def _test_cost(self, st: Loop, env: Env) -> Poly | None:
        """What evaluating the loop's condition costs, paid once per iteration: `strlen(s)` in
        `i < strlen(s)` is O(n) every time round. None when it is constant (the usual case)."""
        if st.test is None or st.kind not in ("while", "for_c", "do_while"):
            return None
        with self.scratch() as probe:
            self.evaluate(st.test, env.copy())
        cost = probe.time.total()
        return cost.order() if cost.vars() else None

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
        return inner.fold(
            loop_id, lambda p: self.repeat_poly(p, plan), plan.iters, plan.exact_iters
        )

    def _amortize(self, inner: Bundle, plan) -> Bundle:  # noqa: ANN001
        """The loop runs `amortized_total` times IN TOTAL across the enclosing loop, not per
        enclosing iteration: its whole cost becomes a lump owned by that enclosing loop."""
        target = plan.amortize_into
        total = plan.amortized_total

        def lump(cost: Cost) -> Cost:
            if cost.own.is_zero():
                return Cost(ZERO, dict(cost.lumps))
            per = self._bound_atoms(cost.own, plan)
            return Cost(ZERO, {**cost.lumps, target: self._amortized_sum(per, plan, total)})

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

    def _amortized_sum(self, per: Poly, plan, total: Poly) -> Poly:  # noqa: ANN001
        """`per` paid for each of `total` distinct visits. When each visit scans one row of an
        adjacency list (`for v in graph[u]` with `u` a vertex taken off the worklist, each vertex
        at most once) the rows add up to the entries of the whole list, not rows * longest row:
        a walk is vertices + edges."""
        uid = plan.ragged_uid
        row = self.rowlen_vars.get(uid) if uid is not None else None
        if uid is not None and row is not None and per.mentions(row):
            return self._repeat_ragged(per, plan, row, uid)
        return (per * total).order()

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
                self.note_alloc(total, "concatenation")
                return result
            self._cur.charge(left.length)
            return make_container(left.kind, left.length, left.elem, owned=True)
        if op in ("+", "-") and isinstance(left, ContV) and isinstance(right, IntV):
            # iterator / pointer arithmetic: begin() + i, end() - k. An iterator is the number of
            # elements from it to the end, so moving forward removes that many
            step = right.mag
            if step is None:
                length = left.length
            elif op == "+":
                length = left.length - step
            else:
                length = left.length + step
            if not any(coef > 0 for _, coef in length.terms):
                length = ZERO
            return make_container(left.kind, length, left.elem, view=True)
        if op == "+" and isinstance(left, IntV) and isinstance(right, ContV):
            step = left.mag
            length = right.length - step if step is not None else right.length
            if not any(coef > 0 for _, coef in length.terms):
                length = ZERO
            return make_container(right.kind, length, right.elem, view=True)
        if (
            op == "-"
            and isinstance(left, ContV)
            and isinstance(right, ContV)
            and left.view
            and right.view
        ):
            return IntV(right.length - left.length)  # last - first: the elements between them
        if op == "*":
            container, count = (left, right) if isinstance(left, ContV) else (right, left)
            if isinstance(container, ContV) and isinstance(count, IntV):
                reps = count.mag if count.mag is not None else self.default_size
                length = (container.length * reps).order()
                self._cur.charge(length)
                result = make_container(container.kind, length, container.elem, owned=True)
                self._cur.alloc(length)
                self.note_alloc(length, "repetition")
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
                # the scan stops at the shorter operand: against a literal that is a constant
                constant = not left.length.vars() or not right.length.vars()
                self._cur.charge(ONE if constant else left.length)
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
                return make_container("list", Poly.var(self.kids_var(obj.size)), obj, view=True)
            if attr.lower() in NODE_ATTRS:
                return NodeV(obj.size)
            return SCALAR
        if isinstance(obj, TupleV):
            if attr in ("first", "0") and obj.items:
                return obj.items[0]
            if attr in ("second", "1") and len(obj.items) > 1:
                return obj.items[1]
        if isinstance(obj, NoneV):
            return UNKNOWN
        if attr.lower() in NODE_ATTRS:
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
            if obj.kind == "dict" and obj.uid in self.autovivify and obj.cap is not None:
                self._cur.grow(obj.cap, ONE)  # `groups[x]` on a defaultdict inserts the key
                self._note_key(obj.cap, index)
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
        # a negative constant counts from the end: s[1:-1] is s without its first and last
        if lo is not None and (lo_const := lo.const_value()) is not None and lo_const < 0:
            lo = obj.length + lo
        if hi is not None and (hi_const := hi.const_value()) is not None and hi_const < 0:
            hi = obj.length + hi
        if e.step is None and lo is not None:
            span = (hi if hi is not None else obj.length) - lo
            if span.is_zero():
                length = ZERO  # x[:0] and x[i:i] are empty slices
            else:
                # kept exact (L - 1, L / 2): a recursion on a slice is measured by how much it
                # shrinks, and charges are ordered when they are made anyway
                length = span if any(c > 0 for _, c in span.terms) else ONE
        if self.lang == "go" and obj.kind in ("list", "array", "str"):
            self._cur.charge(ONE)
            return make_container(obj.kind, length, obj.elem, view=True)
        self._cur.charge(length)
        result = make_container(obj.kind, length, obj.elem, owned=True)
        self._cur.alloc(length)  # a slice copies the slots; the rows it holds stay shared
        self.note_alloc(length, "slice")
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
                # an int element is not remembered: counts = [0] * k is incremented later, so
                # its entries are not zero
                part = item if isinstance(item, ContV | NodeV) else None
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
                # a fresh node is one node, unless this function strings nodes together (`tail.next
                # = new Node(x)`): then what a pointer to one of them reaches is as long as the
                # list that was built, which the engine does not follow node by node
                return NodeV(self.default_size if self._links_nodes else ONE)
            else:
                return UNKNOWN
        length = ZERO
        first = args[0] if args else None
        fill = args[1] if len(args) > 1 else None
        elem: Value | None = None
        if isinstance(first, ContV) and isinstance(fill, ContV) and first.view and fill.view:
            length = first.length - fill.length  # vector<int>(first, last): what lies between
            if not any(coef > 0 for _, coef in length.terms):
                length = first.length
            self._cur.charge(length)
        elif isinstance(first, ContV):
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
        self.note_alloc(mem(container), "new container")
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
            self.note_alloc(total, "new array")
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
        self.note_alloc(mem(container), "comprehension")
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
            return self.call_user(target, args, env, tuple(call.args))
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
                return self.call_user(callee, args, env, tuple(call.args))
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
        if name == "defaultdict" and isinstance(result.value, ContV):
            self.autovivify.add(result.value.uid)
        counted = lib.subject()
        if counted is not None and counted.cap is not None:
            if name in _REMOVERS:
                self._popped.add(counted.cap)
            elif result.grow is not None:
                self._pushed.add(counted.cap)
        self._cur.charge(result.time)
        if result.time.vars():
            self.step("call", f"call to {name}() costs {{0}}", result.time, key=name)
        if result.alloc is not None:
            self._cur.alloc(result.alloc)
            self.note_alloc(result.alloc, f"{name}()")
        subject = lib.subject()
        if result.grow is not None and subject is not None and subject.cap is not None:
            self._cur.grow(subject.cap, result.grow)
        elif result.grow is not None and subject is not None and subject.uid in self.row_owner:
            self._cur.grow(self.total_var(self.row_owner[subject.uid]), result.grow)
        if result.reset and subject is not None and subject.cap is not None:
            self._cur.reset(subject.cap)
        if (
            name == "setdefault"
            and isinstance(receiver, ContV)
            and receiver.owned
            and isinstance(result.value, ContV)
            and result.value.owned
        ):
            self.row_owner[result.value.uid] = receiver.uid
        if (
            result.grow is not None
            and subject is not None
            and subject.cap is not None
            and subject.kind in ("set", "dict")
        ):
            if name in _KEYED_INSERTS and args:
                self._note_key(subject.cap, args[0])
            else:  # `update`, `addAll`, ...: keys we cannot see, so the distinct count is unknown
                self.cap_unbounded.add(subject.cap)
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

    def call_user(
        self,
        target: FuncDef | FuncV,
        args: list[Value],
        env: Env | None = None,
        arg_exprs: tuple[Expr, ...] = (),
    ) -> Value:
        closure: Env | None = None
        if isinstance(target, FuncV):
            if isinstance(target.func, Lambda):
                return self._call_lambda(target, args)
            closure = target.closure if isinstance(target.closure, Env) else None
            func = target.func
        else:
            func = target
        assert isinstance(func, FuncDef)
        if any(func is active for active in self._active):
            self._cur.add_rec(self._rec_call(func, args, env, arg_exprs))
            return self._recursive_result(func, args)
        if self._depth >= MAX_CALL_DEPTH:
            self.note("unknown", "call chain too deep to analyse")
            return UNKNOWN
        self._depth += 1
        try:
            summary = self.summary_for(func, closure)
        finally:
            self._depth -= 1
        mapping = self.bind_args(summary.params, args)
        costs = (summary.time, summary.space, *summary.shared.values())
        for var, reason in self._guessed.items():  # only a guess the callee actually depends on
            if any(cost.mentions(var) for cost in costs):
                self.note("assumed", reason)
        callee_time = summary.time.substitute(mapping)
        self._cur.charge(callee_time)
        self._cur.alloc(summary.space.substitute(mapping))
        self._apply_effects(summary, args, mapping)
        if callee_time.vars():
            self.step("call", f"call to {func.name}() costs {{0}}", callee_time, key=func.name)
        self._adopt_outer_calls(summary, mapping, env)
        self._share(summary, args, mapping)
        if summary.recursive and not summary.solved:
            self.note("unknown", f"recursion in {func.name}() could not be solved")
        result = substitute(summary.ret, mapping) or UNKNOWN
        if isinstance(result, ContV) and result.cap is not None and result.cap.kind == "pgrow":
            result = replace(result, cap=None)  # the callee's own handle on a parameter
        return result

    def _apply_effects(
        self, summary: FuncSummary, args: list[Value], mapping: dict[Var, Poly]
    ) -> None:
        """What the call adds to containers that outlive it is added to the caller's containers:
        the one passed for a parameter, or a captured one (the same variable in the caller)."""
        if not summary.effects:
            return
        actual: dict[Var, Value] = {}
        for index, name in enumerate(summary.params):
            cap = summary.param_caps.get(name)
            if cap is not None and index < len(args):
                actual[cap] = args[index]
        for cap, delta in summary.effects.items():
            amount = delta.substitute(mapping)
            if cap not in actual:
                self._cur.grow(cap, amount)  # captured: the caller sees the same container
                continue
            target = actual[cap]
            if isinstance(target, ContV) and target.cap is not None:
                self._cur.grow(target.cap, amount)

    def _key_domain(self, value: Value) -> Poly | None:
        """How many distinct values a key can take, if every part of it is a loop index (or a
        constant): the product of the ranges of the loops. None when it comes from data."""
        if isinstance(value, TupleV):
            total = ONE
            for part in value.items:
                domain = self._key_domain(part)
                if domain is None:
                    return None
                total = (total * domain).order()
            return total
        if isinstance(value, IntV) and value.mag is not None:
            atoms = value.mag.vars()
            if not atoms:
                return ONE
            if all(v.kind == "iter" and v in self.iters for v in atoms):
                bound = value.mag.substitute({v: self.iters[v].upper for v in atoms})
                return bound.order() if not bound.is_zero() else ONE
        return None

    def _note_key(self, cap: Var, key: Value) -> None:
        domain = self._key_domain(key)
        if domain is None:
            self.cap_unbounded.add(cap)
            return
        previous = self.cap_distinct.get(cap)
        self.cap_distinct[cap] = domain if previous is None else (previous + domain).order()

    def _call_lambda(self, target: FuncV, args: list[Value]) -> Value:
        func = target.func
        assert isinstance(func, Lambda)
        closure = target.closure if isinstance(target.closure, Env) else None
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

    # ------------------------------------------------------------------ recursion bookkeeping
    def _rec_call(
        self, func: FuncDef, args: list[Value], env: Env | None, arg_exprs: tuple[Expr, ...]
    ) -> RecCall:
        return RecCall(
            func.name, tuple(args), ONE, False, func.line, self.path_conds(env), arg_exprs
        )

    def path_conds(self, env: Env | None) -> tuple[Cond, ...]:
        """The size comparisons known to hold on the path being interpreted."""
        if env is None:
            return ()
        conds: list[Cond] = []
        for test, polarity in self._conds:
            conds.extend(self.cond_atoms(test, polarity, env))
        return tuple(conds)

    def _recursive_result(self, func: FuncDef, args: list[Value]) -> Value:
        """What a recursive call returns: the first pass idea of the function result, with the
        arguments of this call substituted in (unknown on the first pass)."""
        known = self._rec_returns.get(id(func))
        if known is None or known[1] is None:
            return UNKNOWN
        params, ret = known
        return substitute(ret, self.bind_args(params, args)) or UNKNOWN

    def cond_atoms(self, test: Expr, polarity: bool, env: Env) -> list[Cond]:
        """The comparisons over sizes that `test` being `polarity` tells us (empty if none)."""
        if isinstance(test, UnOp) and test.op == "not":
            return self.cond_atoms(test.operand, not polarity, env)
        if isinstance(test, BoolOp):
            if (test.op == "and") == polarity:  # a conjunction: every part holds
                return [c for part in test.values for c in self.cond_atoms(part, polarity, env)]
            return []  # a disjunction says nothing about either part
        if isinstance(test, Compare):
            op = test.op if polarity else _NEGATED.get(test.op)
            if op is None or op not in _SIZE_OPS:
                return []
            left = magnitude(self.pure(test.left, env))
            right = magnitude(self.pure(test.right, env))
            if left is None or right is None:
                return []
            return [(op, left, right)]
        if (
            isinstance(test, Call)
            and isinstance(test.func, Attribute)
            and test.func.attr in _EMPTY_TESTS
            and not test.args
        ):  # nums.empty(), nums.isEmpty(): the length is zero
            holder = self.pure(test.func.obj, env)
            if isinstance(holder, ContV):
                return [("==" if polarity else ">", holder.length, ZERO)]
            return []
        if isinstance(test, Name | Attribute | Call | Subscript):
            value = self.pure(test, env)
            if isinstance(value, ContV):
                return [(">" if polarity else "==", value.length, ZERO)]
            if isinstance(value, IntV) and value.mag is not None:
                return [(">" if polarity else "==", value.mag, ZERO)]
        return []

    def _adopt_outer_calls(
        self, summary: FuncSummary, mapping: dict[Var, Poly], env: Env | None
    ) -> None:
        """Mutual recursion: the callee calls a function that is still being analysed further out.
        Those calls become the recursive calls of the function being analysed here (arguments
        rewritten from the callee parameters to ours), so the cycle is one recurrence."""
        if not summary.outer_rec:
            return
        here = self.path_conds(env)
        alternatives: list[tuple[RecCall, ...]] = []
        for alt in summary.outer_rec:
            adopted = []
            for call in alt:
                conds = tuple(
                    (op, left.substitute(mapping), right.substitute(mapping))
                    for op, left, right in call.conds
                )
                adopted.append(
                    replace(
                        call,
                        args=tuple(substitute(a, mapping) for a in call.args),  # type: ignore[arg-type]
                        mult=call.mult.substitute(mapping),
                        conds=(*here, *conds),
                    )
                )
            alternatives.append(tuple(adopted))
        self.absorb(Bundle(rec=alternatives))

    def _share(self, summary: FuncSummary, args: list[Value], mapping: dict[Var, Poly]) -> None:
        """Cost a callee shares through one container (a visited array, a memo table) is paid once
        however often the container is reused: pool it by the container of the caller."""
        for uid, poly in summary.shared_free.items():
            pooled = poly.substitute(mapping)
            current = self._shared_pool.get(uid)
            self._shared_pool[uid] = pooled if current is None else current.max(pooled)
        names = list(summary.params)
        for name, poly in summary.shared.items():
            index = names.index(name)
            actual = args[index] if index < len(args) else None
            pooled = poly.substitute(mapping)
            if not isinstance(actual, ContV):
                self._cur.charge(pooled)  # not a container we can follow
                continue
            current = self._shared_pool.get(actual.uid)
            self._shared_pool[actual.uid] = pooled if current is None else current.max(pooled)

    def _settle_shared(self, summary: FuncSummary) -> None:
        """Close the shared pool of a function: cost through one of its parameter containers stays
        shared with its callers; cost through a container built here is simply its own time."""
        by_uid = {
            value.uid: name for name, value in summary.params.items() if isinstance(value, ContV)
        }
        extra = ZERO
        for uid, poly in self._shared_pool.items():
            if uid in by_uid:
                name = by_uid[uid]
                previous = summary.shared.get(name)
                summary.shared[name] = poly if previous is None else previous.max(poly)
            else:
                extra = extra + poly
        self._shared_pool.clear()
        if not extra.is_zero():
            summary.time = (summary.time + extra).order()


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


def _links_nodes(fdef: FuncDef) -> bool:
    """Does the function store a node into a link field of another (`prev.next = cur`)?"""
    for node in walk(fdef):
        if isinstance(node, Assign) and node.op == "=":
            for target in node.targets:
                if isinstance(target, Attribute) and target.attr.lower() in NODE_ATTRS:
                    return True
    return False


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
