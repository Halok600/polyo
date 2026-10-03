"""The library cost table (`analysis/stdlib.py`).

Hidden library cost was the single largest failure bucket of the shipped model (43% of its time
errors): `x in list` looks like O(1) and is O(n), `list.pop(0)` looks like O(1) and is O(n),
`s += c` copies the string. Each row here states one such convention."""

from __future__ import annotations

import pytest

from analysis.poly import Poly, new_var
from analysis.stdlib import LibCall, call_library
from analysis.values import SCALAR, ContV, IntV, Value, make_container

N = new_var("n")
M = new_var("m")
LEN = Poly.var(N)
ONE = Poly.const(1)


def seq(kind: str = "list", length: Poly = LEN) -> ContV:
    return make_container(kind, length)


def call(
    name: str,
    receiver: Value | None = None,
    args: list[Value] | None = None,
    qualifier: str | None = None,
    lang: str = "python",
) -> object:
    result = call_library(
        LibCall(
            lang=lang,
            name=name,
            qualifier=qualifier,
            receiver=receiver,
            args=args or [],
            arg_exprs=[],
            kwargs={},
            default_size=Poly.var(N),
        )
    )
    assert result is not None, f"{name} is not in the table"
    return result


def time_of(result: object) -> str:
    return result.time.order().render({N: "n", M: "m"})  # type: ignore[attr-defined]


# ------------------------------------------------------------------ membership and lookup
@pytest.mark.parametrize(
    ("name", "kind", "expected"),
    [
        ("index", "list", "O(n)"),
        ("count", "list", "O(n)"),
        ("contains", "list", "O(n)"),
        ("indexOf", "list", "O(n)"),
        ("includes", "list", "O(n)"),
        ("remove", "list", "O(n)"),
        ("contains", "set", "O(1)"),
        ("has", "set", "O(1)"),
        ("containsKey", "dict", "O(1)"),
        ("get", "dict", "O(1)"),
        ("getOrDefault", "dict", "O(1)"),
        ("count", "set", "O(1)"),
        ("find", "dict", "O(1)"),
        ("contains", "treeset", "O(log n)"),
        ("get", "treemap", "O(log n)"),
        ("containsKey", "treemap", "O(log n)"),
    ],
)
def test_membership_and_lookup_cost_by_container_kind(name: str, kind: str, expected: str) -> None:
    assert time_of(call(name, seq(kind), [SCALAR])) == expected


def test_positional_access_is_constant_on_arrays_and_strings() -> None:
    for name in ("get", "at", "charAt"):
        assert time_of(call(name, seq("list"), [IntV(Poly.const(3))])) == "O(1)"
    assert time_of(call("charAt", seq("str"), [IntV(Poly.const(3))])) == "O(1)"


# ----------------------------------------------------------------------------- mutators
@pytest.mark.parametrize("name", ["append", "push", "push_back", "add", "offer", "emplace_back"])
def test_appending_is_constant_and_grows_the_container(name: str) -> None:
    result = call(name, seq("list"), [SCALAR])
    assert time_of(result) == "O(1)"
    assert result.grow is not None  # type: ignore[attr-defined]


def test_pop_from_the_end_is_constant_but_pop_from_the_front_of_a_list_is_linear() -> None:
    assert time_of(call("pop", seq("list"), [])) == "O(1)"
    assert time_of(call("pop", seq("list"), [IntV(Poly.const(0))])) == "O(n)"
    assert time_of(call("shift", seq("list"), [], lang="javascript")) == "O(n)"
    assert time_of(call("erase", seq("list"), [SCALAR], lang="cpp")) == "O(n)"


def test_a_deque_is_constant_at_both_ends() -> None:
    for name in ("popleft", "appendleft", "poll", "pollFirst", "pop_front", "push_front"):
        assert (
            time_of(
                call(name, seq("deque"), [SCALAR] if "push" in name or "append" in name else [])
            )
            == "O(1)"
        )


def test_inserting_at_the_front_of_a_list_is_linear() -> None:
    assert time_of(call("insert", seq("list"), [IntV(Poly.const(0)), SCALAR])) == "O(n)"
    assert time_of(call("unshift", seq("list"), [SCALAR], lang="javascript")) == "O(n)"


def test_stack_and_queue_operations_are_constant() -> None:
    for name in ("push", "pop", "top", "peek", "empty", "isEmpty", "size", "front", "back"):
        assert time_of(call(name, seq("stack"), [SCALAR] if name == "push" else [])) == "O(1)"


# --------------------------------------------------------------------------------- sorting
def test_sorting_is_n_log_n_and_sorted_returns_a_copy() -> None:
    result = call("sorted", None, [seq("list")])
    assert time_of(result) == "O(n log n)"
    assert result.alloc is not None  # type: ignore[attr-defined]
    assert isinstance(result.value, ContV)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("name", "qualifier", "lang"),
    [
        ("sort", None, "python"),
        ("sort", "std", "cpp"),
        ("stable_sort", "std", "cpp"),
        ("sort", "Arrays", "java"),
        ("sort", "Collections", "java"),
        ("sort", None, "javascript"),
        ("Ints", "sort", "go"),
        ("Slice", "sort", "go"),
        ("Sort", "slices", "go"),
    ],
)
def test_every_languages_sort_is_n_log_n(name: str, qualifier: str | None, lang: str) -> None:
    receiver = seq("list") if qualifier is None else None
    args = [] if qualifier is None else [seq("array")]
    assert time_of(call(name, receiver, args, qualifier=qualifier, lang=lang)) == "O(n log n)"


def test_library_sort_auxiliary_space_differs_by_language() -> None:
    py = call("sort", seq("list"), [])
    cpp = call("sort", None, [seq("list")], qualifier="std", lang="cpp")
    assert py.alloc is not None and py.alloc.order().render({N: "n"}) == "O(n)"  # type: ignore[attr-defined]
    assert cpp.alloc is not None and cpp.alloc.order().render({N: "n"}) == "O(log n)"  # type: ignore[attr-defined]


# ----------------------------------------------------------------------- heaps and search
@pytest.mark.parametrize(
    ("name", "qualifier", "lang"),
    [("heappush", "heapq", "python"), ("heappop", "heapq", "python"), ("push", None, "cpp")],
)
def test_heap_operations_are_logarithmic(name: str, qualifier: str | None, lang: str) -> None:
    heap = seq("heap")
    if qualifier:
        result = call(name, None, [heap, SCALAR], qualifier=qualifier, lang=lang)
    else:
        result = call(name, heap, [SCALAR], lang=lang)
    assert time_of(result) == "O(log n)"


def test_heapify_is_linear() -> None:
    assert time_of(call("heapify", None, [seq("list")], qualifier="heapq")) == "O(n)"


@pytest.mark.parametrize(
    ("name", "qualifier", "lang"),
    [
        ("bisect_left", "bisect", "python"),
        ("lower_bound", "std", "cpp"),
        ("upper_bound", "std", "cpp"),
        ("binarySearch", "Arrays", "java"),
        ("SearchInts", "sort", "go"),
    ],
)
def test_binary_search_is_logarithmic(name: str, qualifier: str, lang: str) -> None:
    assert (
        time_of(call(name, None, [seq("array"), SCALAR], qualifier=qualifier, lang=lang))
        == "O(log n)"
    )


# ------------------------------------------------------------------------ copies and builtins
@pytest.mark.parametrize("name", ["list", "set", "tuple", "dict", "sorted"])
def test_conversions_copy_their_argument(name: str) -> None:
    result = call(name, None, [seq("list")])
    assert time_of(result).startswith("O(n")
    assert isinstance(result.value, ContV)  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["sum", "max", "min", "any", "all"])
def test_reductions_over_a_container_are_linear(name: str) -> None:
    assert time_of(call(name, None, [seq("list")])) == "O(n)"


def test_max_and_min_of_two_scalars_are_constant() -> None:
    two = [IntV(Poly.const(1)), IntV(Poly.const(2))]
    assert time_of(call("max", None, two)) == "O(1)"
    assert time_of(call("min", None, two)) == "O(1)"


def test_len_is_constant_and_returns_the_length() -> None:
    result = call("len", None, [seq("list")])
    assert time_of(result) == "O(1)"
    assert isinstance(result.value, IntV) and result.value.mag == LEN  # type: ignore[attr-defined]


def test_size_and_length_methods_return_the_length() -> None:
    for name in ("size", "length"):
        result = call(name, seq("list"), [])
        assert isinstance(result.value, IntV) and result.value.mag == LEN  # type: ignore[attr-defined]


def test_sqrt_and_log_return_the_matching_magnitude() -> None:
    root = call("sqrt", None, [IntV(LEN)], qualifier="math")
    assert root.value.mag.order().render({N: "n"}) == "O(sqrt n)"  # type: ignore[attr-defined]
    logged = call("log2", None, [IntV(LEN)], qualifier="math")
    assert logged.value.mag.order().render({N: "n"}) == "O(log n)"  # type: ignore[attr-defined]


def test_magnitude_preserving_conversions() -> None:
    for name, qualifier in (("int", None), ("abs", None), ("floor", "math"), ("floor", "Math")):
        result = call(name, None, [IntV(LEN)], qualifier=qualifier)
        assert result.value == IntV(LEN)  # type: ignore[attr-defined]


# -------------------------------------------------------------------------------- strings
@pytest.mark.parametrize(
    "name", ["split", "strip", "lower", "upper", "replace", "toLowerCase", "trim"]
)
def test_string_methods_are_linear_in_the_string(name: str) -> None:
    assert time_of(call(name, seq("str"), [])) == "O(n)"


def test_substring_costs_its_own_length() -> None:
    result = call("substring", seq("str"), [IntV(Poly.const(0)), IntV(Poly.const(5))], lang="java")
    assert time_of(result) == "O(1)"


def test_string_join_is_linear_in_the_parts() -> None:
    assert time_of(call("join", seq("str", Poly.const(1)), [seq("list")])) == "O(n)"


def test_string_find_and_contains_are_linear() -> None:
    for name in ("find", "indexOf", "startsWith", "endsWith", "contains"):
        assert time_of(call(name, seq("str"), [seq("str", Poly.const(3))])) in ("O(n)", "O(1)")


def test_string_builder_append_is_constant_and_to_string_is_linear() -> None:
    sb = seq("str")
    assert time_of(call("append", sb, [SCALAR], lang="java")) == "O(1)"
    assert time_of(call("toString", sb, [], lang="java")) == "O(n)"


# ---------------------------------------------------------------------------- program input
def test_reading_input_yields_values_sized_by_the_input() -> None:
    result = call("input", None, [])
    assert isinstance(result.value, ContV) and result.value.kind == "str"  # type: ignore[attr-defined]
    assert time_of(result) == "O(1)" or time_of(result) == "O(n)"


def test_unknown_calls_are_not_in_the_table() -> None:
    assert (
        call_library(
            LibCall("python", "frobnicate", None, None, [], [], {}, default_size=Poly.var(N))
        )
        is None
    )
