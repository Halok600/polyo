"""Cost composition (`analysis/bundle.py`).

A `Cost` is the order-level cost of a region plus pending "lumps": costs that belong to the WHOLE
run of an enclosing loop rather than to each of its iterations. That is how an amortised inner
loop (a sliding window's `while left < right`, a monotonic stack's pops) is counted once across
the outer loop instead of being multiplied by it."""

from __future__ import annotations

from analysis.bundle import Cost
from analysis.poly import Poly, new_var


def order(poly: Poly) -> str:
    return poly.order().render({})


def test_sequential_costs_add() -> None:
    n, m = new_var("n"), new_var("m")
    total = Cost(Poly.var(n)).then(Cost(Poly.var(m)))
    assert order(total.own) == "O(n + m)"


def test_branches_take_the_larger_cost() -> None:
    n = new_var("n")
    cheap, dear = Cost(Poly.const(1)), Cost(Poly.var(n) * Poly.var(n))
    assert order(cheap.either(dear).own) == "O(n^2)"


def test_a_lump_is_not_multiplied_by_the_loops_it_passes_through() -> None:
    n = new_var("n")
    inner = Cost(Poly.const(1), {7: Poly.var(n)})  # a lump that belongs to loop 7
    # An unrelated loop (id 3) with n iterations multiplies the per-iteration cost only.
    folded = inner.fold(loop_id=3, repeat=lambda p: p * Poly.var(n))
    assert order(folded.own) == "O(n)"
    assert folded.lumps == {7: Poly.var(n)}


def test_a_lump_is_added_once_when_its_own_loop_closes() -> None:
    n = new_var("n")
    body = Cost(Poly.const(1), {7: Poly.var(n)})
    folded = body.fold(loop_id=7, repeat=lambda p: p * Poly.var(n))
    # n iterations of O(1) plus the lump n, once: still O(n), NOT n * n.
    assert order(folded.own) == "O(n)"
    assert folded.lumps == {}


def test_sliding_window_shape_is_linear_not_quadratic() -> None:
    # outer for over n; inner while amortised into the outer loop (id 1): total n, once.
    n = new_var("n")
    inner_loop_total = Poly.var(n)  # the whole inner while, across all outer iterations
    body = Cost(Poly.const(1)).then(Cost(Poly.const(1), {1: inner_loop_total}))
    folded = body.fold(loop_id=1, repeat=lambda p: p * Poly.var(n))
    assert order(folded.own) == "O(n)"


def test_branch_merge_takes_the_larger_lump_per_loop() -> None:
    n, m = new_var("n"), new_var("m")
    a = Cost(Poly.const(1), {1: Poly.var(n)})
    b = Cost(Poly.const(1), {1: Poly.var(m), 2: Poly.var(n)})
    merged = a.either(b)
    assert order(merged.lumps[1]) == "O(n + m)"
    assert order(merged.lumps[2]) == "O(n)"


def test_zero_cost_is_the_identity() -> None:
    n = new_var("n")
    c = Cost(Poly.var(n))
    assert c.then(Cost.zero()).own == c.own
    assert Cost.zero().then(c).own == c.own


def test_total_collapses_every_pending_lump() -> None:
    n, m = new_var("n"), new_var("m")
    c = Cost(Poly.var(n), {1: Poly.var(m)})
    assert order(c.total()) == "O(n + m)"


def test_costs_stay_order_level_and_never_negative() -> None:
    n = new_var("n")
    big = Cost(Poly.var(n) * 5 + 7).then(Cost(Poly.var(n) * 3))
    assert big.own == Poly.var(n).order()
