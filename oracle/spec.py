"""Oracle test-spec schema (plan §6): declares an entrypoint, its parameters,
and which parameter(s) scale with n, so codegen/runner can build inputs at a
size grid without ever inspecting the solution's source. This is what a
driver is generated from -- not the solution code itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ParamType(str, Enum):
    INT = "int"
    INT_ARRAY = "int[]"


class ParamDist(str, Enum):
    UNIFORM = "uniform"
    SORTED = "sorted"
    CONST = "const"


class InvalidSpecError(ValueError):
    """A test-spec is structurally invalid. Raised, never guessed around --
    an oracle run on a malformed spec would silently mislabel data."""


@dataclass(frozen=True, slots=True)
class ParamSpec:
    type: ParamType
    size: str | None = None  # n_grid variable this param's length/value scales with
    dist: ParamDist = ParamDist.UNIFORM
    const: float | int | None = None  # fixed value, used when size is None

    def __post_init__(self) -> None:
        if self.size is None and self.dist == ParamDist.CONST and self.const is None:
            raise InvalidSpecError("a sizeless const param needs a const value")
        if self.type == ParamType.INT_ARRAY and self.size is None and self.const is None:
            raise InvalidSpecError("an int[] param needs either size or a const length")


@dataclass(frozen=True, slots=True)
class NGrid:
    """A geometric n_grid by default (plan §6: "2^8 -> 2^20"); pass
    `explicit_values` instead for algorithms where a fixed base blows up too
    fast for a wide geometric range (e.g. exponential recursion)."""

    variable: str = "n"
    start_exp: int = 8
    stop_exp: int = 20
    base: int = 2
    explicit_values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.explicit_values is not None:
            if len(self.explicit_values) < 2:
                raise InvalidSpecError("explicit_values needs >=2 points to fit a shape")
            if any(v < 2 for v in self.explicit_values):
                raise InvalidSpecError("n values must be >=2 (log(n) must be defined)")
            return
        if self.stop_exp < self.start_exp:
            raise InvalidSpecError("n_grid stop_exp must be >= start_exp")
        if self.base ** self.start_exp < 2:
            raise InvalidSpecError("n_grid must start at n >= 2 (log(n) must be defined)")

    def values(self) -> list[int]:
        if self.explicit_values is not None:
            return list(self.explicit_values)
        return [self.base**e for e in range(self.start_exp, self.stop_exp + 1)]


@dataclass(frozen=True, slots=True)
class TestSpec:
    __test__ = False  # not a pytest test class despite the name (plan §6's own term)

    entrypoint: str
    params: tuple[ParamSpec, ...]
    n_grid: NGrid = field(default_factory=NGrid)
    budget_ms: int = 2000

    def __post_init__(self) -> None:
        if not self.entrypoint:
            raise InvalidSpecError("entrypoint must be non-empty")
        if not self.params:
            raise InvalidSpecError("params must declare at least one parameter")
        if not any(p.size == self.n_grid.variable for p in self.params):
            raise InvalidSpecError(
                f"no parameter sizes on n_grid variable {self.n_grid.variable!r}"
            )
        if self.budget_ms <= 0:
            raise InvalidSpecError("budget_ms must be positive")
