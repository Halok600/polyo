"""The cost algebra: polynomials-with-logs over input-size variables.

A running time or a space bound is a sum of monomials like `n^2 log n`, `n * m`, `2^n` or `n!`,
where `n`, `m` are `Var`s standing for input sizes (a list's length, an integer parameter's value,
a loop induction variable, ...). The engine needs four things from it, and each is a method:

  * exact arithmetic with rational coefficients, so `range(n - 5, n)` gives the constant 5 rather
    than "n", and `(i + k + 1) - (i - k)` gives `2k + 1`;
  * `order()`: the asymptotic order, dropping coefficients, lower-order terms and negative terms
    (a negative term only ever lowers a count, so dropping it is a sound upper bound);
  * `sum_over(var, kind, upper)`: closed forms for a loop whose body cost mentions its own
    induction variable. This is where the two classic traps are decided correctly:
    `sum n/i = n log n` (harmonic) and `1 + 2 + 4 + ... + n = O(n)` (geometric), neither of which
    a plain product of loop bounds gets right;
  * `project_time` / `project_space`: round onto the legacy taxonomy, rounding UP and saying so.

Monomial dominance is componentwise per variable, lexicographic in (factorial, exponential,
power, log power). Terms in different variables are incomparable, so `n + m` stays `n + m`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

_uid = itertools.count(1)
Number = int | Fraction


@dataclass(frozen=True, slots=True)
class Var:
    """An input-size variable. Equality is by identity (`uid`), never by name: two parameters
    that are both called `n` in different functions are different variables."""

    uid: int
    name: str
    kind: str = "val"  # val | len | iter | rowlen | total | free


def new_var(name: str, kind: str = "val") -> Var:
    return Var(next(_uid), name, kind)


@dataclass(frozen=True, slots=True)
class Mono:
    """One product of factors; the coefficient lives in the owning `Poly`."""

    pows: tuple[tuple[Var, Fraction, int], ...] = ()  # (var, power, log power), sorted by uid
    exps: tuple[tuple[Var, int], ...] = ()  # base ** var, sorted by uid
    facts: tuple[Var, ...] = ()  # var!, sorted by uid, repeats allowed

    def sort_key(self) -> tuple:
        return (
            tuple((v.uid, p, lg) for v, p, lg in self.pows),
            tuple((v.uid, b) for v, b in self.exps),
            tuple(v.uid for v in self.facts),
        )

    def vars(self) -> set[Var]:
        return {v for v, _, _ in self.pows} | {v for v, _ in self.exps} | set(self.facts)

    def growth(self, var: Var) -> tuple[int, int, Fraction, int]:
        """Growth of this monomial in `var`: (factorials, exponential base, power, log power)."""
        fact = sum(1 for v in self.facts if v == var)
        base = max((b for v, b in self.exps if v == var), default=0)
        power, log = Fraction(0), 0
        for v, p, lg in self.pows:
            if v == var:
                power, log = p, lg
        return (fact, base, power, log)

    def is_constant(self) -> bool:
        return not self.pows and not self.exps and not self.facts


def _normalise_pows(
    items: dict[Var, tuple[Fraction, int]],
) -> tuple[tuple[Var, Fraction, int], ...]:
    return tuple(
        sorted(
            ((v, p, lg) for v, (p, lg) in items.items() if p != 0 or lg != 0),
            key=lambda t: t[0].uid,
        )
    )


def _mul_mono(a: Mono, b: Mono) -> Mono:
    pows: dict[Var, tuple[Fraction, int]] = {v: (p, lg) for v, p, lg in a.pows}
    for v, p, lg in b.pows:
        op, olg = pows.get(v, (Fraction(0), 0))
        pows[v] = (op + p, olg + lg)
    exps: dict[Var, int] = dict(a.exps)
    for v, base in b.exps:
        exps[v] = exps[v] * base if v in exps else base
    facts = tuple(sorted([*a.facts, *b.facts], key=lambda v: v.uid))
    return Mono(
        _normalise_pows(pows),
        tuple(sorted(exps.items(), key=lambda t: t[0].uid)),
        facts,
    )


_MAY_BE_ZERO = frozenset({"rowlen"})


def _dominates(a: Mono, b: Mono) -> bool:
    """True if `a` grows at least as fast as `b` in every variable (so `b` can be dropped).

    A per-row length can be 0, so it never dominates a constant: summed over the rows of an
    adjacency list `1 + row` is `V + E`, and dropping the 1 early would lose the V."""
    if not b.vars() and a.vars() and all(v.kind in _MAY_BE_ZERO for v in a.vars()):
        return False
    return all(a.growth(v) >= b.growth(v) for v in a.vars() | b.vars())


def _frac(value: Number) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _is_integer(value: Fraction) -> bool:
    return value.denominator == 1


@dataclass(frozen=True, slots=True)
class Poly:
    terms: tuple[tuple[Mono, Fraction], ...] = ()

    # ---------------------------------------------------------------- construction
    @staticmethod
    def _make(items: Mapping[Mono, Fraction]) -> Poly:
        kept = [(m, c) for m, c in items.items() if c != 0]
        kept.sort(key=lambda t: t[0].sort_key())
        return Poly(tuple(kept))

    @staticmethod
    def zero() -> Poly:
        return Poly()

    @staticmethod
    def const(value: Number) -> Poly:
        return Poly._make({Mono(): _frac(value)})

    @staticmethod
    def var(v: Var) -> Poly:
        return Poly._make({Mono(((v, Fraction(1), 0),)): Fraction(1)})

    @staticmethod
    def exp(v: Var, base: int = 2) -> Poly:
        return Poly._make({Mono(exps=((v, base),)): Fraction(1)})

    @staticmethod
    def fact(v: Var) -> Poly:
        return Poly._make({Mono(facts=(v,)): Fraction(1)})

    @staticmethod
    def _coerce(other: object) -> Poly:
        if isinstance(other, Poly):
            return other
        if isinstance(other, int | Fraction):
            return Poly.const(other)
        raise TypeError(f"cannot combine Poly with {type(other).__name__}")

    # ---------------------------------------------------------------- arithmetic
    def __add__(self, other: object) -> Poly:
        o = Poly._coerce(other)
        items = dict(self.terms)
        for m, c in o.terms:
            items[m] = items.get(m, Fraction(0)) + c
        return Poly._make(items)

    __radd__ = __add__

    def __neg__(self) -> Poly:
        return Poly._make({m: -c for m, c in self.terms})

    def __sub__(self, other: object) -> Poly:
        return self + (-Poly._coerce(other))

    def __rsub__(self, other: object) -> Poly:
        return Poly._coerce(other) - self

    def __mul__(self, other: object) -> Poly:
        o = Poly._coerce(other)
        items: dict[Mono, Fraction] = {}
        for ma, ca in self.terms:
            for mb, cb in o.terms:
                key = _mul_mono(ma, mb)
                items[key] = items.get(key, Fraction(0)) + ca * cb
        return Poly._make(items)

    __rmul__ = __mul__

    def __truediv__(self, other: Number) -> Poly:
        divisor = _frac(other)
        if divisor == 0:
            raise ZeroDivisionError("division of a Poly by zero")
        return Poly._make({m: c / divisor for m, c in self.terms})

    # ---------------------------------------------------------------- queries
    def is_zero(self) -> bool:
        return not self.terms

    def const_value(self) -> Fraction | None:
        """The value, if this polynomial is a plain constant."""
        if not self.terms:
            return Fraction(0)
        if len(self.terms) == 1 and self.terms[0][0].is_constant():
            return self.terms[0][1]
        return None

    def vars(self) -> set[Var]:
        out: set[Var] = set()
        for m, _ in self.terms:
            out |= m.vars()
        return out

    def mentions(self, v: Var) -> bool:
        return v in self.vars()

    def extract_power(self, v: Var) -> list[tuple[Fraction, Poly]]:
        """Group the terms by the power of `v` they contain: [(power, rest)], where `rest` is the
        sum of those terms with `v` divided out (power 0 collects the terms without `v`)."""
        groups: dict[Fraction, Poly] = {}
        for mono, coef in self.terms:
            power = Fraction(0)
            rest_pows = []
            for var, p, lg in mono.pows:
                if var == v:
                    power = p
                else:
                    rest_pows.append((var, p, lg))
            rest = Poly._make({Mono(tuple(rest_pows), mono.exps, mono.facts): coef})
            groups[power] = groups[power] + rest if power in groups else rest
        return list(groups.items())

    def affine_in(self, v: Var) -> tuple[Fraction, Poly] | None:
        """Write this polynomial as `k * v + rest` with `rest` free of `v`; None if `v` appears
        any other way (squared, multiplied by another variable, inside a log, ...)."""
        slope = Fraction(0)
        rest: dict[Mono, Fraction] = {}
        for mono, coef in self.terms:
            if v not in mono.vars():
                rest[mono] = rest.get(mono, Fraction(0)) + coef
                continue
            if mono.exps or mono.facts or len(mono.pows) != 1:
                return None
            _var, power, log = mono.pows[0]
            if power != 1 or log != 0:
                return None
            slope += coef
        return slope, Poly._make(rest)

    def degree_in(self, v: Var) -> Fraction:
        """The largest power of `v` among the terms (0 if `v` does not appear)."""
        return max((p for p, _ in self.extract_power(v)), default=Fraction(0))

    # ---------------------------------------------------------------- order
    def order(self) -> Poly:
        """The asymptotic order: positive terms only, unit coefficients, dominated terms removed.
        The order of zero is Theta(1)."""
        monos = [m for m, c in self.terms if c > 0]
        kept = [m for m in monos if not any(o != m and _dominates(o, m) for o in monos)]
        if not kept:
            return Poly.const(1)
        return Poly._make(dict.fromkeys(kept, Fraction(1)))

    def max(self, other: Poly) -> Poly:
        return (self + other).order()

    # ---------------------------------------------------------------- functions
    def _dominant_monos(self) -> list[Mono]:
        return [m for m, _ in self.order().terms]

    def log(self) -> Poly:
        """log2 of this quantity, as an order-level polynomial. log(n^p m^q) = p log n + q log m;
        log(2^n) = n; log(n!) = n log n; log of a constant is a constant."""
        result = Poly.zero()
        for mono in self._dominant_monos():
            piece = Poly.zero()
            for v, p, _lg in mono.pows:
                if p > 0:
                    piece += Poly._make({Mono(((v, Fraction(0), 1),)): Fraction(1)})
            for v, _base in mono.exps:
                piece += Poly.var(v)
            for v in mono.facts:
                piece += Poly._make({Mono(((v, Fraction(1), 1),)): Fraction(1)})
            result += piece
        return result.order() if not result.is_zero() else Poly.const(1)

    def power(self, k: Number) -> Poly:
        """`self ** k`. Exact for a single monomial and for non-negative integer `k`; otherwise the
        dominant monomials are used, which is an upper bound for 0 < k <= 1."""
        exponent = _frac(k)
        if exponent == 0:
            return Poly.const(1)
        if len(self.terms) == 1:
            return self._power_of_monomial(*self.terms[0], exponent)
        if _is_integer(exponent) and exponent > 0:
            result = Poly.const(1)
            for _ in range(int(exponent)):
                result = result * self
            return result
        result = Poly.zero()
        for mono in self._dominant_monos():
            result += self._power_of_monomial(mono, Fraction(1), exponent)
        return result.order()

    @staticmethod
    def _power_of_monomial(mono: Mono, coef: Fraction, k: Fraction) -> Poly:
        pows = {
            v: (p * k, lg * int(k) if _is_integer(k) else -(-lg * k.numerator // k.denominator))
            for v, p, lg in mono.pows
        }
        exps = tuple(
            (v, base ** int(k) if _is_integer(k) and k > 0 else base) for v, base in mono.exps
        )
        facts = mono.facts * int(k) if _is_integer(k) and k > 0 else mono.facts
        new_mono = Mono(_normalise_pows(pows), exps, tuple(sorted(facts, key=lambda v: v.uid)))
        new_coef = coef ** int(k) if _is_integer(k) else Fraction(1)
        return Poly._make({new_mono: new_coef})

    def sqrt(self) -> Poly:
        return self.power(Fraction(1, 2))

    # ---------------------------------------------------------------- substitution
    def substitute(self, mapping: Mapping[Var, Poly]) -> Poly:
        if not mapping:
            return self
        result = Poly.zero()
        for mono, coef in self.terms:
            piece = Poly.const(coef)
            kept_pows: dict[Var, tuple[Fraction, int]] = {}
            for v, p, lg in mono.pows:
                if v in mapping:
                    base = mapping[v]
                    if p != 0:
                        piece = piece * base.power(p)
                    if lg > 0:
                        piece = piece * base.log().power(lg)
                else:
                    kept_pows[v] = (p, lg)
            kept_exps: list[tuple[Var, int]] = []
            for v, base_num in mono.exps:
                if v in mapping:
                    piece = piece * _exp_of(mapping[v], base_num)
                else:
                    kept_exps.append((v, base_num))
            kept_facts: list[Var] = []
            for v in mono.facts:
                if v in mapping:
                    piece = piece * _fact_of(mapping[v])
                else:
                    kept_facts.append(v)
            rest = Mono(_normalise_pows(kept_pows), tuple(kept_exps), tuple(kept_facts))
            result += piece * Poly._make({rest: Fraction(1)})
        return result

    # ---------------------------------------------------------------- closed-form loop sums
    def sum_over(self, var: Var, kind: str, upper: Poly) -> Poly:
        """Sum this body cost over a loop whose induction variable is `var`.

        kind="arith": var takes the values 1..upper (a linear sweep, `range(n)`, `i += 1`).
        kind="geom":  var takes the values 1, 2, 4, ... up to `upper` (or upper, upper/2, ...:
                      the same sum), so there are log(upper) iterations.
        Terms that do not mention `var` simply multiply by the iteration count.
        """
        if kind not in ("arith", "geom"):
            raise ValueError(f"unknown sum kind {kind!r}")
        iterations = upper if kind == "arith" else upper.log()
        result = Poly.zero()
        for mono, coef in self.terms:
            if any(v == var for v, _ in mono.exps) or var in mono.facts:
                # unsupported shape: bound the variable by its upper limit (sound, not tight)
                bounded = Poly._make({mono: coef}).substitute({var: upper})
                result += bounded * iterations
                continue
            power, log = Fraction(0), 0
            rest_pows = []
            found = False
            for v, p, lg in mono.pows:
                if v == var:
                    power, log, found = p, lg, True
                else:
                    rest_pows.append((v, p, lg))
            rest = Poly._make({Mono(tuple(rest_pows), mono.exps, mono.facts): coef})
            if not found:
                result += rest * iterations
                continue
            result += rest * _closed_form(kind, power, log, upper)
        return result

    # ---------------------------------------------------------------- rendering
    def render(self, names: Mapping[Var, str] | None = None) -> str:
        """`O(...)` for an order-level polynomial. `names` maps variables to display letters and
        its insertion order is the display order of variables."""
        names = names or {}
        order_of = {v: i for i, v in enumerate(names)}

        def rank(v: Var) -> int:
            return order_of.get(v, 10_000 + v.uid)

        def label(v: Var) -> str:
            return names.get(v, v.name)

        pieces: list[tuple[tuple, str]] = []
        for mono, _ in self.terms:
            groups: list[tuple[int, str]] = []
            for v, p, lg in mono.pows:
                groups.append((rank(v), _render_group(label(v), p, lg)))
            groups.sort()
            parts = [text for _, text in groups]
            parts += [f"{base}^{label(v)}" for v, base in mono.exps]
            parts += [f"{label(v)}!" for v in mono.facts]
            text = " * ".join(parts) if parts else "1"
            first = min((rank(v) for v in mono.vars()), default=-1)
            growth_key = tuple(
                -x
                for v in sorted(mono.vars(), key=rank)
                for x in (
                    mono.growth(v)[0],
                    mono.growth(v)[1],
                    float(mono.growth(v)[2]),
                    mono.growth(v)[3],
                )
            )
            pieces.append(((first, growth_key), text))
        pieces.sort(key=lambda t: t[0])
        return "O(" + (" + ".join(t for _, t in pieces) if pieces else "1") + ")"


def _closed_form(kind: str, power: Fraction, log: int, upper: Poly) -> Poly:
    """Order of sum over the loop of `var^power * (log var)^log`."""
    log_upper = upper.log()
    if kind == "arith":
        if power > -1:
            return upper.power(power + 1) * (log_upper.power(log) if log else Poly.const(1))
        if power == -1:
            return log_upper.power(log + 1)
        return Poly.const(1)
    # geometric: var = 2^k for k = 0 .. log(upper)
    if power > 0:
        return upper.power(power) * (log_upper.power(log) if log else Poly.const(1))
    if power == 0:
        return log_upper.power(log + 1)
    return Poly.const(1)


def _exp_of(poly: Poly, base: int) -> Poly:
    """base ** poly, exactly when poly is `c * v` for an integer c, else bounded by its dominant
    variable (an upper bound)."""
    if len(poly.terms) == 1:
        mono, coef = poly.terms[0]
        if len(mono.pows) == 1 and not mono.exps and not mono.facts:
            v, p, lg = mono.pows[0]
            if p == 1 and lg == 0 and _is_integer(coef) and coef >= 1:
                return Poly._make({Mono(exps=((v, base ** int(coef)),)): Fraction(1)})
    dominant = next(iter(poly.order().terms), None)
    if dominant is None or dominant[0].is_constant():
        return Poly.const(1)
    v = dominant[0].pows[0][0] if dominant[0].pows else next(iter(dominant[0].vars()))
    return Poly.exp(v, base)


def _fact_of(poly: Poly) -> Poly:
    dominant = next(iter(poly.order().terms), None)
    if dominant is None or dominant[0].is_constant():
        return Poly.const(1)
    v = dominant[0].pows[0][0] if dominant[0].pows else next(iter(dominant[0].vars()))
    return Poly.fact(v)


def _render_group(label: str, power: Fraction, log: int) -> str:
    if power == 1:
        text = label
    elif power == Fraction(1, 2):
        text = f"sqrt {label}"
    elif power == 0:
        text = ""
    elif _is_integer(power):
        text = f"{label}^{int(power)}"
    elif power.denominator in (2, 4, 5, 10):
        text = f"{label}^{float(power):g}"
    else:
        text = f"{label}^({power.numerator}/{power.denominator})"
    if log == 1:
        logs = f"log {label}"
    elif log > 1:
        logs = f"log^{log} {label}"
    else:
        logs = ""
    return f"{text} {logs}".strip()


# -------------------------------------------------------------------------------- projection
def _collapse(mono: Mono) -> tuple[int, int, Fraction, int, bool]:
    """Collapse every variable of `mono` into one `n`.

    Returns (factorials, exponential base, power, log power, mixes polynomial with exp/fact).
    """
    power = sum((p for _, p, _ in mono.pows), Fraction(0))
    log = sum(lg for _, _, lg in mono.pows)
    base = max((b for _, b in mono.exps), default=0)
    facts = len(mono.facts)
    extras = bool(mono.pows) and bool(mono.exps or mono.facts)
    return (facts, base, power, log, extras)


def _dominant_collapsed(poly: Poly) -> tuple[int, int, Fraction, int, bool]:
    options = [_collapse(m) for m, _ in poly.order().terms]
    return (
        max(options, key=lambda t: (t[0], t[1], t[2], t[3]))
        if options
        else (0, 0, Fraction(0), 0, False)
    )


def project_time(poly: Poly) -> tuple[str, bool]:
    """(legacy time class, lossy). Rounds UP to the next legacy class: O(sqrt n) -> O(n),
    O(n^2 log n) -> O(n^3), beyond n^3 -> O(n^3), n! / 3^n / n*2^n -> O(2^n)."""
    facts, base, power, log, extras = _dominant_collapsed(poly)
    if facts:
        return ("O(2^n)", True)
    if base:
        return ("O(2^n)", base != 2 or extras)
    if power >= 3:
        return ("O(n^3)", not (power == 3 and log == 0))
    if power > 2:
        return ("O(n^3)", True)
    if power == 2:
        return ("O(n^2)", False) if log == 0 else ("O(n^3)", True)
    if power > 1:
        return ("O(n^2)", True)
    if power == 1:
        if log == 0:
            return ("O(n)", False)
        return ("O(n log n)", False) if log == 1 else ("O(n^2)", True)
    if power > 0:
        return ("O(n)", True)
    if log == 0:
        return ("O(1)", False)
    return ("O(log n)", False) if log == 1 else ("O(n)", True)


def project_space(poly: Poly) -> tuple[str, bool]:
    """(legacy space class, lossy), onto O(1) / O(log n) / O(n) / O(n log n) / O(n^2)."""
    facts, base, power, log, _extras = _dominant_collapsed(poly)
    if facts or base:
        return ("O(n^2)", True)
    if power >= 2:
        return ("O(n^2)", not (power == 2 and log == 0))
    if power > 1:
        return ("O(n^2)", True)
    if power == 1:
        if log == 0:
            return ("O(n)", False)
        return ("O(n log n)", False) if log == 1 else ("O(n^2)", True)
    if power > 0:
        return ("O(n)", True)
    if log == 0:
        return ("O(1)", False)
    return ("O(log n)", False) if log == 1 else ("O(n)", True)
