"""The symbolic complexity engine: source text in, time and space out.

    analysis = analyze(source, "cpp")
    analysis.time.text      # "O(n^2)"
    analysis.time.cls       # "O(n^2)"   (the legacy 7-class taxonomy, rounded up if needed)
    analysis.certainty      # "certain" | "assumed" | "unknown"

`analyze` lowers the source, picks the function(s) that define the answer (never a `main` or test
harness that merely calls the solution on a literal), interprets them symbolically and reports the
polynomial-with-logs result, a legacy class, and how much of it rests on assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.env import Env
from analysis.interp import FuncSummary, Interp, Note, StepRec
from analysis.lower import lower_source
from analysis.nodes import Attribute, Call, FuncDef, Module, Name, walk
from analysis.poly import Poly, Var, project_space, project_time, project_time_extended
from analysis.roles import params_to_values
from analysis.values import ContV, IntV, NodeV, Value

ONE = Poly.const(1)
_LETTERS = "nmkpqrstuvwxyz"
_NON_ENTRY = frozenset({"main", "Main", "__init__", "__main__", "setUp", "tearDown"})
_INPUT_CALLS = frozenset(
    {"input", "raw_input", "readline", "readLine", "nextInt", "nextLine", "next", "Fscan", "Scan",
     "Fscanln", "readFileSync", "scanf", "getline", "ReadString", "Text", "parseInt", "nextLong"}
)  # fmt: skip


@dataclass
class Result:
    expr: Poly
    text: str
    cls: str  # the legacy class, rounded up when the expression is not exactly one of them
    lossy: bool  # `cls` is not exactly the expression
    ext_cls: str = ""  # the extended class (time: adds sqrt n, n^2 log n, n!); space: same as cls


@dataclass(frozen=True)
class Step:
    """One line of the derivation: what the engine concluded about a piece of code."""

    line: int  # 1-based source line, 0 for the totals
    kind: str  # loop | call | recursion | alloc | total
    text: str


@dataclass(frozen=True)
class Assumption:
    """A bound the engine could not prove and assumed (it errs high, never low)."""

    line: int
    reason: str


@dataclass
class Analysis:
    time: Result
    space: Result
    certainty: str
    notes: list[Note]
    entry: str
    names: dict[Var, str] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)


# ---------------------------------------------------------------------------- entry selection
def _callees(func: FuncDef, names: set[str]) -> set[str]:
    out: set[str] = set()
    for node in walk(func):
        if isinstance(node, Call):
            if isinstance(node.func, Name) and node.func.id in names:
                out.add(node.func.id)
            elif isinstance(node.func, Attribute) and node.func.attr in names:
                out.add(node.func.attr)
    return out


def _reads_input(func: FuncDef) -> bool:
    """True if the function reads standard input (`cin >> n`, `sc.nextInt()`, `input()`)."""
    for node in walk(func):
        if isinstance(node, Name) and node.id in ("cin", "stdin"):
            return True
        if isinstance(node, Call):
            callee = node.func
            name = (
                callee.id
                if isinstance(callee, Name)
                else (callee.attr if isinstance(callee, Attribute) else "")
            )
            if name in _INPUT_CALLS:
                return True
    return False


def select_entries(module: Module) -> list[FuncDef]:
    """The functions whose cost is the answer: those nothing else in the file calls (a recursive
    function calling itself does not count), excluding `main`-like drivers. If every function is
    called by another (mutual recursion), all of them are entries.

    A `main` that reads input IS the program rather than a harness around a solution: it is the
    only entry (the cost of whatever it calls is part of its own)."""
    programs = [f for f in module.functions if f.name in ("main", "Main") and _reads_input(f)]
    if programs:
        return programs[:1]
    candidates = [f for f in module.functions if f.name not in _NON_ENTRY]
    if not candidates:
        return []
    names = {f.name for f in candidates}
    graph = {f.name: _callees(f, names) - {f.name} for f in candidates}
    called: set[str] = set().union(*graph.values()) if graph else set()
    roots = [f for f in candidates if f.name not in called]
    # functions no root can reach call each other in a cycle: one of each group is an entry too
    reachable: set[str] = set()
    stack = [f.name for f in roots]
    while stack:
        current = stack.pop()
        if current not in reachable:
            reachable.add(current)
            stack.extend(graph.get(current, ()))
    leftovers = [f for f in candidates if f.name not in reachable]
    if not leftovers:
        return roots
    inner = {f.name: graph[f.name] & {g.name for g in leftovers} for f in leftovers}
    return [*roots, *_cycle_representatives(leftovers, inner)]


def _cycle_representatives(candidates: list[FuncDef], graph: dict[str, set[str]]) -> list[FuncDef]:
    """Every function is called by another: mutual recursion. One function per group of functions
    that call each other is enough (the rest are part of its recurrence), and only groups nothing
    outside them calls count."""
    reach: dict[str, set[str]] = {}
    for name in graph:
        seen: set[str] = set()
        stack = list(graph[name])
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(graph.get(current, ()))
        reach[name] = seen
    entries: list[FuncDef] = []
    covered: set[str] = set()
    for func in candidates:
        if func.name in covered:
            continue
        group = {
            n for n in graph if n == func.name or (n in reach[func.name] and func.name in reach[n])
        }
        outside_callers = {n for n in graph if n not in group and graph[n] & group}
        covered |= group
        if not outside_callers:
            entries.append(func)
    return entries or candidates[:1]


def _is_program(module: Module) -> bool:
    """Module-level code that reads input or loops is a program, not a library of functions."""
    for stmt in module.toplevel:
        for node in walk(stmt):
            if isinstance(node, Call):
                callee = node.func
                name = (
                    callee.id
                    if isinstance(callee, Name)
                    else (callee.attr if isinstance(callee, Attribute) else "")
                )
                if name in _INPUT_CALLS:
                    return True
        from analysis.nodes import Loop

        if any(isinstance(n, Loop) for n in walk(stmt)):
            return True
    return False


# ------------------------------------------------------------------------------- naming
def _vars_of(value: Value) -> list[Var]:
    out: list[Var] = []
    if isinstance(value, IntV) and value.mag is not None:
        out.extend(sorted(value.mag.vars(), key=lambda v: v.uid))
    elif isinstance(value, ContV):
        out.extend(sorted(value.length.vars(), key=lambda v: v.uid))
        if value.elem is not None:
            out.extend(_vars_of(value.elem))
    elif isinstance(value, NodeV):
        out.extend(sorted(value.size.vars(), key=lambda v: v.uid))
    return out


def _display_names(params: dict[str, Value], polys: list[Poly], it: Interp) -> dict[Var, str]:
    """Letters n, m, k, ... for the variables that appear, in parameter order."""
    appearing: set[Var] = set()
    for poly in polys:
        appearing |= poly.vars()
    ordered: list[Var] = []
    for value in params.values():
        for var in _vars_of(value):
            if var in appearing and var not in ordered:
                ordered.append(var)
    if it.input_var in appearing and it.input_var not in ordered:
        ordered.insert(0, it.input_var)
    for var in sorted(appearing, key=lambda v: v.uid):
        if var not in ordered:
            ordered.append(var)
    return {var: _LETTERS[i] if i < len(_LETTERS) else var.name for i, var in enumerate(ordered)}


def _entry_time(summary: FuncSummary) -> Poly:
    """The time of a function called from outside: cost it shares through its parameter
    containers (a visited array, a memo table) is paid here, once."""
    total = summary.time
    for shared in summary.shared.values():
        total = total + shared
    return total.order() if not total.is_zero() else ONE


_KIND_ORDER = {"loop": 0, "recursion": 1, "call": 2, "alloc": 3}


def _letters_for_steps(records: list[StepRec], names: dict[Var, str]) -> dict[Var, str]:
    """`names` plus a fresh letter for each size variable only the derivation mentions, so a step
    never prints an internal variable name where the answer would have used a letter."""
    out = dict(names)
    free = [c for c in _LETTERS if c not in out.values()]
    for rec in records:
        for poly in rec.polys:
            for var in sorted(poly.vars(), key=lambda v: v.uid):
                if var not in out and var.kind != "iter" and free:
                    out[var] = free.pop(0)
    return out


def _render_steps(it: Interp, names: dict[Var, str], time: Poly, space: Poly) -> list[Step]:
    """The recorded steps by source line (a later record of the same loop, call or allocation
    replaces an earlier one: a recursion is analysed twice and only the second pass counts), then
    the two totals."""
    latest: dict[tuple[str, int, str], StepRec] = {}
    for rec in it.steps:
        latest[(rec.kind, rec.line, rec.key)] = rec
    records = sorted(latest.values(), key=lambda r: (r.line, _KIND_ORDER.get(r.kind, 9)))
    shown = _letters_for_steps(records, names)
    steps = [
        Step(rec.line, rec.kind, rec.fmt.format(*(p.render(shown) for p in rec.polys)))
        for rec in records
    ]
    steps.append(Step(0, "total", f"time: {time.render(shown)}"))
    steps.append(Step(0, "total", f"space: {space.render(shown)}"))
    return steps


# ---------------------------------------------------------------------------------- analyze
def _summarise_program(it: Interp, module: Module) -> FuncSummary:
    pseudo = FuncDef(name="<program>", params=(), body=module.toplevel)
    return it.summary_for(pseudo)


def analyze(source: str, language: str) -> Analysis:
    module = lower_source(source, language)
    it = Interp(module, language)
    entries: list[FuncDef] = []
    summaries: list[FuncSummary] = []
    if _is_program(module) or (not module.functions and module.toplevel):
        summaries.append(_summarise_program(it, module))
        entry_name = "<program>"
    else:
        entries = select_entries(module)
        if not entries and any(f.name in _NON_ENTRY for f in module.functions):
            entries = [f for f in module.functions if f.name == "main"][:1] or list(
                module.functions
            )
        for func in entries:
            summaries.append(it.summary_for(func))
        entry_name = ", ".join(f.name for f in entries) if entries else "<none>"
    if not summaries:
        it.note("unknown", "no function to analyse")
        empty = Result(ONE, "O(1)", "O(1)", False, "O(1)")
        return Analysis(
            empty, empty, "unknown", it.notes, entry_name, assumptions=_assumptions(it.notes)
        )

    time = _entry_time(summaries[0])
    space = summaries[0].space
    for summary in summaries[1:]:
        time, space = time.max(_entry_time(summary)), space.max(summary.space)
    time, space = time.at_least_one().order(), space.at_least_one().order()
    for summary in summaries:
        if summary.recursive and not summary.solved:
            it.note("unknown", f"recursion in {summary.func.name}() could not be solved")

    params = summaries[0].params if summaries else {}
    if len(summaries) > 1:
        params = {}
        for summary in summaries:
            params.update(summary.params)
    names = _display_names(params, [time, space], it)
    time_cls, time_lossy = project_time(time)
    space_cls, space_lossy = project_space(space)
    time_ext, _ = project_time_extended(time)
    if any(n.kind == "unknown" for n in it.notes):
        certainty = "unknown"
    elif any(n.kind == "assumed" for n in it.notes):
        certainty = "assumed"
    else:
        certainty = "certain"
    return Analysis(
        time=Result(time.order(), time.order().render(names), time_cls, time_lossy, time_ext),
        space=Result(space.order(), space.order().render(names), space_cls, space_lossy, space_cls),
        certainty=certainty,
        notes=it.notes,
        entry=entry_name,
        names=names,
        steps=_render_steps(it, names, time.order(), space.order()),
        assumptions=_assumptions(it.notes),
    )


def _assumptions(notes: list[Note]) -> list[Assumption]:
    return [Assumption(n.line, n.reason) for n in notes if n.kind == "assumed"]


__all__ = [
    "Analysis",
    "Assumption",
    "Env",
    "Result",
    "Step",
    "analyze",
    "params_to_values",
    "select_entries",
]
