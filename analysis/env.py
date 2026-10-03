"""Variable environments for the abstract interpreter."""

from __future__ import annotations

from analysis.values import Value, join


class Env:
    """A scope: name -> abstract value, with an optional enclosing scope (a closure).

    Writes always go to the innermost scope; reads fall through to the parent. `copy()` is a
    cheap branch point, and `merge()` is the join at the end of an `if`."""

    __slots__ = ("parent", "vars")

    def __init__(self, vars: dict[str, Value] | None = None, parent: Env | None = None) -> None:
        self.vars: dict[str, Value] = vars if vars is not None else {}
        self.parent = parent

    def get(self, name: str) -> Value | None:
        env: Env | None = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        return None

    def set(self, name: str, value: Value) -> None:
        self.vars[name] = value

    def copy(self) -> Env:
        return Env(dict(self.vars), self.parent)

    def merge(self, other: Env) -> Env:
        """The join of two branch-end environments (variables only one branch defined are kept)."""
        merged: dict[str, Value] = {}
        for name in self.vars.keys() | other.vars.keys():
            a, b = self.vars.get(name), other.vars.get(name)
            merged[name] = join(a, b)
        return Env(merged, self.parent)

    def local_names(self) -> frozenset[str]:
        return frozenset(self.vars)
