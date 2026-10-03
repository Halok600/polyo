"""Recursion: turn the recursive calls recorded while interpreting a function into a closed form.

(Phase 3. Until it lands a recursive function keeps only its non-recursive work and is marked
unsolved, which the engine reports as "unknown" instead of guessing.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.bundle import Bundle
from analysis.nodes import FuncDef
from analysis.values import UNKNOWN, Value

if TYPE_CHECKING:
    from analysis.interp import FuncSummary, Interp


def solve_recurrence(it: Interp, summary: FuncSummary, body: Bundle) -> None:
    summary.solved = False


def solve_nested(
    it: Interp, func: FuncDef, args: list[Value], body: Bundle, ret: Value | None
) -> Value:
    it.note("unknown", f"recursion in nested function {func.name}() could not be solved")
    it.absorb(body)
    return ret if ret is not None else UNKNOWN
