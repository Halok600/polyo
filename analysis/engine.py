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
from analysis.interp import FuncSummary, Interp, Note
from analysis.lower import lower_source
from analysis.nodes import Attribute, Call, FuncDef, Module, Name, walk
from analysis.poly import Poly, Var, project_space, project_time
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
    cls: str
    lossy: bool


@dataclass
class Analysis:
    time: Result
    space: Result
    certainty: str
    notes: list[Note]
    entry: str
    names: dict[Var, str] = field(default_factory=dict)


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
    called: set[str] = set()
    for func in candidates:
        called |= _callees(func, names) - {func.name}
    roots = [f for f in candidates if f.name not in called]
    return roots or candidates


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
        empty = Result(ONE, "O(1)", "O(1)", False)
        return Analysis(empty, empty, "unknown", it.notes, entry_name)

    time = summaries[0].time
    space = summaries[0].space
    for summary in summaries[1:]:
        time, space = time.max(summary.time), space.max(summary.space)
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
    if any(n.kind == "unknown" for n in it.notes):
        certainty = "unknown"
    elif any(n.kind == "assumed" for n in it.notes):
        certainty = "assumed"
    else:
        certainty = "certain"
    return Analysis(
        time=Result(time.order(), time.order().render(names), time_cls, time_lossy),
        space=Result(space.order(), space.order().render(names), space_cls, space_lossy),
        certainty=certainty,
        notes=it.notes,
        entry=entry_name,
        names=names,
    )


__all__ = ["Analysis", "Env", "Result", "analyze", "params_to_values", "select_entries"]
