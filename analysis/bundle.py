"""Cost composition for the abstract interpreter.

A `Cost` is the order-level cost of a region plus pending LUMPS. A lump is a cost that belongs to
the whole run of one enclosing loop (identified by its id) rather than to each of that loop's
iterations. This is what makes amortised code come out right:

    for right in range(n):          # loop 1
        ...
        while left < right: left += 1   # `left` is never reset, so across ALL n iterations the
                                        # while runs at most n times in total: a lump for loop 1.

When loop 1 closes, `fold` multiplies the per-iteration cost by its iteration count and adds the
lump exactly once. Loops between the lump's origin and its target do not multiply it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from analysis.poly import Poly, Var


def _ordered(poly: Poly) -> Poly:
    return poly.order() if not poly.is_zero() else poly


@dataclass(frozen=True, eq=False)
class Cost:
    own: Poly = field(default_factory=Poly.zero)
    lumps: dict[int, Poly] = field(default_factory=dict)

    @staticmethod
    def zero() -> Cost:
        return Cost()

    def is_zero(self) -> bool:
        return self.own.is_zero() and not self.lumps

    def then(self, other: Cost) -> Cost:
        """This region followed by `other`."""
        lumps = dict(self.lumps)
        for key, poly in other.lumps.items():
            lumps[key] = _ordered(lumps[key] + poly) if key in lumps else poly
        return Cost(_ordered(self.own + other.own), lumps)

    def either(self, other: Cost) -> Cost:
        """Control flow takes one of two paths: the worst case of each part."""
        lumps = dict(self.lumps)
        for key, poly in other.lumps.items():
            lumps[key] = _ordered(lumps[key] + poly) if key in lumps else poly
        return Cost(_ordered(self.own + other.own), lumps)

    def with_lump(self, loop_id: int, poly: Poly) -> Cost:
        lumps = dict(self.lumps)
        lumps[loop_id] = _ordered(lumps[loop_id] + poly) if loop_id in lumps else _ordered(poly)
        return Cost(self.own, lumps)

    def fold(self, loop_id: int, repeat: Callable[[Poly], Poly]) -> Cost:
        """Close loop `loop_id`: repeat the per-iteration cost, add this loop's own lump once, and
        pass every other loop's lump through untouched."""
        lumps = dict(self.lumps)
        mine = lumps.pop(loop_id, None)
        own = _ordered(repeat(self.own)) if not self.own.is_zero() else self.own
        if mine is not None:
            own = _ordered(own + mine)
        return Cost(own, lumps)

    def total(self) -> Poly:
        """Everything, with every still-pending lump added once."""
        out = self.own
        for poly in self.lumps.values():
            out = out + poly
        return _ordered(out)


# ----------------------------------------------------------------------------------- Bundle
@dataclass(frozen=True)
class RecCall:
    """A recursive call made by the function being analysed, recorded instead of costed.

    `args` are the abstract values passed (polynomials in the function's OWN parameter
    variables), `mult` how many times it runs per invocation (enclosing loop counts)."""

    callee: str
    args: tuple[object, ...]
    mult: Poly
    in_loop: bool = False
    line: int = 0


@dataclass
class Bundle:
    """Everything a region of code does: its time, how it grows local containers, the memory it
    allocates, and the recursive calls it makes.

    * `time`      - `Cost` of the region.
    * `grows`     - growth (in memory units) of each GROWABLE local container, by its capacity
                    variable; composed through loops like time, and read off to bound the
                    container's final size.
    * `allocs`    - memory allocated by the region that dies with it (per-iteration temporaries
                    are NOT multiplied by the loop around them).
    * `retained`  - memory the region keeps alive across iterations (stored into a container),
                    which IS multiplied by enclosing loops.
    * `created`   - capacity variables created here and not yet finalised.
    * `rec`       - alternatives (one per control-flow path) of recursive calls made.
    * `unflushed` / `flushed` / `resets` - which growable containers' CONTENT is consumed here
                    (a copy, a join, a scan: any cost that mentions its capacity) with no later
                    reset on that path, with a later reset, and which are reset on EVERY path. A
                    buffer that is consumed and then cleared each iteration costs its total growth
                    over the loop, not its size once per iteration.
    """

    time: Cost = field(default_factory=Cost.zero)
    grows: dict[Var, Cost] = field(default_factory=dict)
    allocs: Poly = field(default_factory=Poly.zero)
    retained: Cost = field(default_factory=Cost.zero)
    created: set[Var] = field(default_factory=set)
    rec: list[tuple[RecCall, ...]] = field(default_factory=lambda: [()])
    unflushed: set[Var] = field(default_factory=set)
    flushed: set[Var] = field(default_factory=set)
    resets: set[Var] = field(default_factory=set)

    # ---- in-place accumulation (used while a statement is being evaluated)
    def charge(self, poly: Poly) -> None:
        self.time = self.time.then(Cost(poly.order()))
        self.unflushed |= _caps_of(poly)

    def alloc(self, poly: Poly) -> None:
        self.allocs = _ordered(self.allocs + poly)

    def retain(self, poly: Poly) -> None:
        self.retained = self.retained.then(Cost(poly.order()))
        self.unflushed |= _caps_of(poly)

    def reset(self, cap: Var) -> None:
        """The container with capacity variable `cap` was emptied (`clear()`, `x = []`)."""
        if cap in self.unflushed:
            self.unflushed.discard(cap)
            self.flushed.add(cap)
        self.resets.add(cap)

    def grow(self, cap: Var, poly: Poly) -> None:
        current = self.grows.get(cap, Cost.zero())
        self.grows[cap] = current.then(Cost(poly.order()))

    def add_rec(self, call: RecCall) -> None:
        self.rec = [(*alt, call) for alt in self.rec]

    # ---- composition
    def then(self, other: Bundle) -> Bundle:
        grows = dict(self.grows)
        for cap, cost in other.grows.items():
            grows[cap] = grows[cap].then(cost) if cap in grows else cost
        rec = [(*a, *b) for a in self.rec for b in other.rec][:_MAX_REC_ALTERNATIVES]
        return Bundle(
            time=self.time.then(other.time),
            grows=grows,
            allocs=_ordered(self.allocs + other.allocs),
            retained=self.retained.then(other.retained),
            created=self.created | other.created,
            rec=rec,
            unflushed=(self.unflushed - other.resets) | other.unflushed,
            flushed=self.flushed | other.flushed | (self.unflushed & other.resets),
            resets=self.resets | other.resets,
        )

    def either(self, other: Bundle) -> Bundle:
        grows = dict(self.grows)
        for cap, cost in other.grows.items():
            grows[cap] = grows[cap].either(cost) if cap in grows else cost
        rec = (self.rec + other.rec)[:_MAX_REC_ALTERNATIVES]
        return Bundle(
            time=self.time.either(other.time),
            grows=grows,
            allocs=_ordered(self.allocs + other.allocs),
            retained=self.retained.either(other.retained),
            created=self.created | other.created,
            rec=rec,
            unflushed=self.unflushed | other.unflushed,
            flushed=self.flushed | other.flushed,
            resets=self.resets & other.resets,
        )

    def fold(self, loop_id: int, repeat: Callable[[Poly], Poly], multiplicity: Poly) -> Bundle:
        """Close a loop: repeat per-iteration time, growth and retained memory; temporaries
        (`allocs`) are reused each iteration; recursive calls inside run `multiplicity` times."""
        rec: list[tuple[RecCall, ...]] = []
        for alt in self.rec:
            rec.append(
                tuple(
                    RecCall(c.callee, c.args, _ordered(c.mult * multiplicity), True, c.line)
                    for c in alt
                )
            )
        return Bundle(
            time=self.time.fold(loop_id, repeat),
            grows={cap: cost.fold(loop_id, repeat) for cap, cost in self.grows.items()},
            allocs=self.allocs,
            retained=self.retained.fold(loop_id, repeat),
            created=set(self.created),
            rec=rec,
            unflushed=set(self.unflushed),  # a loop may run zero times: nothing is reset for sure
            flushed=set(),
            resets=set(),
        )


_MAX_REC_ALTERNATIVES = 16


def _caps_of(poly: Poly) -> set[Var]:
    return {v for v in poly.vars() if v.kind == "cap"}
