"""The cost algebra (`analysis/poly.py`).

A cost is a polynomial-with-logs over input-size variables: exact arithmetic with rational
coefficients (so `range(n - 5, n)` cancels to the constant 5), reduction to an asymptotic ORDER
(dominated terms dropped), closed-form loop sums, substitution for interprocedural calls, and a
projection onto the legacy taxonomy. These are the properties the whole engine rests on."""

from __future__ import annotations

from fractions import Fraction

import pytest

from analysis.poly import Poly, Var, new_var, project_space, project_time


def names(*vars_: Var) -> dict[Var, str]:
    return {v: label for v, label in zip(vars_, "nmkpq", strict=False)}


@pytest.fixture
def n() -> Var:
    return new_var("n")


@pytest.fixture
def m() -> Var:
    return new_var("m")


def order_text(poly: Poly, *vars_: Var) -> str:
    return poly.order().render(names(*vars_))


# ------------------------------------------------------------------------------- arithmetic
def test_exact_arithmetic_keeps_rational_coefficients(n: Var) -> None:
    p = Poly.var(n)
    assert (p + 1) * (p - 1) == p * p - 1
    assert (p / 2) * 2 == p
    assert p - p == Poly.zero()


def test_a_constant_difference_cancels_exactly(n: Var) -> None:
    # range(n - 5, n): the iteration count is hi - lo = 5, a constant, not "n".
    p = Poly.var(n)
    assert (p - (p - 5)) == Poly.const(5)
    assert (p - (p - 5)).order().render({}) == "O(1)"


def test_a_window_count_depends_only_on_the_window_size(n: Var, m: Var) -> None:
    i, k = Poly.var(n), Poly.var(m)
    assert ((i + k + 1) - (i - k)) == k * 2 + 1  # range(i - k, i + k + 1)


# ------------------------------------------------------------------------------------ order
def test_order_drops_coefficients_and_lower_terms(n: Var) -> None:
    p = Poly.var(n)
    assert order_text(p * p * 3 + p * 5 + 2, n) == "O(n^2)"
    assert order_text(Poly.const(7), n) == "O(1)"
    assert order_text(Poly.zero(), n) == "O(1)"


def test_order_drops_negative_terms_as_an_upper_bound(n: Var, m: Var) -> None:
    # triangular inner loop: n - i is at most n.
    assert order_text(Poly.var(n) - Poly.var(m), n, m) == "O(n)"


def test_incomparable_variables_both_survive(n: Var, m: Var) -> None:
    p = Poly.var(n) + Poly.var(m)
    assert order_text(p, n, m) == "O(n + m)"
    assert order_text(p * p, n, m) == "O(m * n + m^2 + n^2)" or order_text(p * p, n, m) in (
        "O(n^2 + m * n + m^2)",
        "O(n^2 + n * m + m^2)",
    )


def test_pruning_removes_only_dominated_terms(n: Var, m: Var) -> None:
    n_sq = Poly.var(n) * Poly.var(n)
    n_log_n = Poly.var(n) * Poly.var(n).log()
    p = n_sq + n_log_n + Poly.var(m)
    assert order_text(p, n, m) == "O(n^2 + m)"


def test_products_of_independent_variables(n: Var, m: Var) -> None:
    assert order_text(Poly.var(n) * Poly.var(m), n, m) == "O(n * m)"


# ------------------------------------------------------------------------------------ logs
def test_log_of_a_monomial_is_a_sum_of_logs(n: Var, m: Var) -> None:
    p = Poly.var(n) * Poly.var(m)
    assert order_text(p.log(), n, m) == "O(log n + log m)"
    assert order_text((Poly.var(n) * Poly.var(n)).log(), n) == "O(log n)"


def test_n_log_n_and_n_squared_log_n_render(n: Var) -> None:
    p = Poly.var(n)
    assert order_text(p * p.log(), n) == "O(n log n)"
    assert order_text(p * p * p.log(), n) == "O(n^2 log n)"


def test_log_of_a_constant_is_constant(n: Var) -> None:
    assert order_text(Poly.const(8).log(), n) == "O(1)"


def test_sqrt_halves_the_exponent(n: Var) -> None:
    assert order_text(Poly.var(n).sqrt(), n) == "O(sqrt n)"
    assert order_text((Poly.var(n) * Poly.var(n) * Poly.var(n)).sqrt(), n) == "O(n^1.5)"


def test_power_of_a_single_monomial(n: Var) -> None:
    assert order_text(Poly.var(n).power(Fraction(3)), n) == "O(n^3)"
    assert order_text(Poly.var(n).power(Fraction(1, 2)), n) == "O(sqrt n)"


# ------------------------------------------------------------------- exponentials / factorial
def test_exponential_and_factorial_render_and_dominate(n: Var) -> None:
    assert order_text(Poly.exp(n), n) == "O(2^n)"
    assert order_text(Poly.fact(n), n) == "O(n!)"
    assert order_text(Poly.exp(n) + Poly.var(n) * Poly.var(n), n) == "O(2^n)"
    assert order_text(Poly.fact(n) + Poly.exp(n), n) == "O(n!)"
    assert order_text(Poly.var(n) * Poly.exp(n), n) == "O(n * 2^n)"


def test_exponential_with_a_different_base() -> None:
    n = new_var("n")
    assert order_text(Poly.exp(n, base=3), n) == "O(3^n)"
    assert order_text(Poly.exp(n, base=3) + Poly.exp(n, base=2), n) == "O(3^n)"


# ---------------------------------------------------------------------------- substitution
def test_substitution_replaces_a_variable_by_a_polynomial(n: Var, m: Var) -> None:
    v = new_var("v")
    callee = Poly.var(v) * Poly.var(v) + Poly.var(v)
    caller = callee.substitute({v: Poly.var(n) / 2})
    assert caller.order() == (Poly.var(n) * Poly.var(n)).order()


def test_substitution_into_a_log_expands_the_log_of_a_product(n: Var, m: Var) -> None:
    v = new_var("v")
    callee = Poly.var(v).log()
    caller = callee.substitute({v: Poly.var(n) * Poly.var(m)})
    assert order_text(caller, n, m) == "O(log n + log m)"


def test_substituting_a_sum_into_a_linear_term_distributes(n: Var, m: Var) -> None:
    v = new_var("v")
    assert order_text(Poly.var(v).substitute({v: Poly.var(n) + Poly.var(m)}), n, m) == "O(n + m)"


# ------------------------------------------------------------------------- closed-form sums
def test_arithmetic_sum_of_the_index_is_quadratic(n: Var) -> None:
    i = new_var("i", "iter")
    total = Poly.var(i).sum_over(i, "arith", Poly.var(n))  # sum_{i<n} i
    assert order_text(total, n) == "O(n^2)"


def test_arithmetic_sum_of_a_constant_is_the_bound(n: Var) -> None:
    i = new_var("i", "iter")
    assert order_text(Poly.const(1).sum_over(i, "arith", Poly.var(n)), n) == "O(n)"


def test_harmonic_sum_is_n_log_n(n: Var) -> None:
    # for i in 1..n: for j in range(0, n, i): the inner count is n / i, and sum_{i<=n} n/i = n ln n.
    i = new_var("i", "iter")
    inner = Poly.var(n) * Poly.var(i).power(Fraction(-1))
    assert order_text(inner.sum_over(i, "arith", Poly.var(n)), n) == "O(n log n)"


def test_geometric_outer_with_linear_inner_in_the_index_is_linear(n: Var) -> None:
    # i = 1, 2, 4, ... < n; inner loop runs i times: 1 + 2 + 4 + ... = O(n), NOT n log n.
    i = new_var("i", "iter")
    assert order_text(Poly.var(i).sum_over(i, "geom", Poly.var(n)), n) == "O(n)"


def test_geometric_outer_with_independent_inner_multiplies_by_log(n: Var) -> None:
    i = new_var("i", "iter")
    body = Poly.var(n)  # does not mention i
    assert order_text(body.sum_over(i, "geom", Poly.var(n)), n) == "O(n log n)"


def test_geometric_sum_of_a_constant_is_log(n: Var) -> None:
    i = new_var("i", "iter")
    assert order_text(Poly.const(1).sum_over(i, "geom", Poly.var(n)), n) == "O(log n)"


def test_sum_over_leaves_other_variables_alone(n: Var, m: Var) -> None:
    i = new_var("i", "iter")
    body = Poly.var(i) * Poly.var(m)
    assert (
        order_text(body.sum_over(i, "arith", Poly.var(n)), n, m) == "O(m * n^2)"
        or order_text(body.sum_over(i, "arith", Poly.var(n)), n, m) == "O(n^2 * m)"
    )


# ------------------------------------------------------------------------------ projection
@pytest.mark.parametrize(
    ("build", "expected", "lossy"),
    [
        (lambda n: Poly.const(1), "O(1)", False),
        (lambda n: Poly.var(n).log(), "O(log n)", False),
        (lambda n: Poly.var(n), "O(n)", False),
        (lambda n: Poly.var(n) * Poly.var(n).log(), "O(n log n)", False),
        (lambda n: Poly.var(n) * Poly.var(n), "O(n^2)", False),
        (lambda n: Poly.var(n).power(Fraction(3)), "O(n^3)", False),
        (lambda n: Poly.exp(n), "O(2^n)", False),
        # rounded UP to the next legacy class, flagged lossy
        (lambda n: Poly.var(n).sqrt(), "O(n)", True),
        (lambda n: Poly.var(n) * Poly.var(n) * Poly.var(n).log(), "O(n^3)", True),
        (lambda n: Poly.var(n).power(Fraction(4)), "O(n^3)", True),
        (lambda n: Poly.fact(n), "O(2^n)", True),
        (lambda n: Poly.var(n) * Poly.exp(n), "O(2^n)", True),
        (lambda n: Poly.exp(n, base=3), "O(2^n)", True),
    ],
)
def test_time_projection_rounds_up_and_flags_loss(build, expected: str, lossy: bool) -> None:  # noqa: ANN001
    n = new_var("n")
    assert project_time(build(n)) == (expected, lossy)


def test_multiple_variables_collapse_to_one_for_the_legacy_class() -> None:
    n, m = new_var("n"), new_var("m")
    assert project_time(Poly.var(n) * Poly.var(m)) == ("O(n^2)", False)
    assert project_time(Poly.var(n) + Poly.var(m)) == ("O(n)", False)
    assert project_time(Poly.var(m) * Poly.var(n).log()) == ("O(n log n)", False)


@pytest.mark.parametrize(
    ("build", "expected", "lossy"),
    [
        (lambda n: Poly.const(1), "O(1)", False),
        (lambda n: Poly.var(n).log(), "O(log n)", False),
        (lambda n: Poly.var(n), "O(n)", False),
        (lambda n: Poly.var(n) * Poly.var(n).log(), "O(n log n)", False),
        (lambda n: Poly.var(n) * Poly.var(n), "O(n^2)", False),
        (lambda n: Poly.var(n).sqrt(), "O(n)", True),
        (lambda n: Poly.var(n).power(Fraction(3)), "O(n^2)", True),
        (lambda n: Poly.exp(n), "O(n^2)", True),
    ],
)
def test_space_projection_uses_the_five_space_classes(build, expected: str, lossy: bool) -> None:  # noqa: ANN001
    n = new_var("n")
    assert project_space(build(n)) == (expected, lossy)


# --------------------------------------------------------------------------- rendering order
def test_render_names_variables_through_the_mapping_and_falls_back_to_origin(n: Var) -> None:
    assert (Poly.var(n) * Poly.var(n)).order().render({n: "k"}) == "O(k^2)"
    assert Poly.var(n).order().render({}) == "O(n)"


def test_vars_are_distinct_even_with_the_same_origin_name() -> None:
    a, b = new_var("n"), new_var("n")
    assert a != b
    assert Poly.var(a) + Poly.var(b) != Poly.var(a) * 2


def test_poly_is_hashable_and_comparable_by_value(n: Var) -> None:
    a = Poly.var(n) * 2 + 1
    b = Poly.var(n) * 2 + 1
    assert a == b and hash(a) == hash(b)


# ------------------------------------------------------------------------- helpers for bounds
def test_extract_power_splits_terms_by_the_power_of_a_variable(n: Var, m: Var) -> None:
    i = Poly.var(n)
    rows = Poly.var(m)
    p = i * i * 3 + i * rows + rows
    parts = dict(p.extract_power(n))
    assert set(parts) == {Fraction(2), Fraction(1), Fraction(0)}
    assert parts[Fraction(2)] == Poly.const(3)
    assert parts[Fraction(1)] == rows
    assert parts[Fraction(0)] == rows


def test_affine_in_recovers_the_slope_and_offset(n: Var, m: Var) -> None:
    a, b = Poly.var(n), Poly.var(m)
    assert (a * 2 + 3).affine_in(n) == (Fraction(2), Poly.const(3))
    assert (a / 2 - 1).affine_in(n) == (Fraction(1, 2), Poly.const(-1))
    assert (a + b).affine_in(n) == (Fraction(1), b)
    assert Poly.const(5).affine_in(n) == (Fraction(0), Poly.const(5))


def test_affine_in_rejects_nonlinear_use(n: Var, m: Var) -> None:
    a, b = Poly.var(n), Poly.var(m)
    assert (a * a).affine_in(n) is None
    assert (a * b).affine_in(n) is None


def test_degree_in_reports_the_dominant_power(n: Var) -> None:
    a = Poly.var(n)
    assert (a * a + a).degree_in(n) == Fraction(2)
    assert Poly.const(4).degree_in(n) == Fraction(0)
    assert a.sqrt().degree_in(n) == Fraction(1, 2)
