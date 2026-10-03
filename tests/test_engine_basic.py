"""End-to-end behaviour of the engine on small snippets (`analysis/engine.py`).

Each test states one construct and the answer a careful reader would give. The golden suite
covers the same ground across languages; these exist so that a failure there can be reduced to
a five-line reproduction and pinned."""

from __future__ import annotations

import pytest

from analysis.engine import analyze


def answer(source: str, language: str = "python") -> tuple[str, str]:
    result = analyze(source, language)
    return result.time.text, result.space.text


# ---------------------------------------------------------------------- straight-line and loops
def test_straight_line_code_is_constant() -> None:
    assert answer("def add(a, b):\n    return a + b\n") == ("O(1)", "O(1)")


def test_a_loop_over_the_input_is_linear() -> None:
    assert answer("def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n") == (
        "O(n)",
        "O(1)",
    )


def test_nested_loops_multiply() -> None:
    source = "def f(xs):\n    for x in xs:\n        for y in xs:\n            pass\n"
    assert answer(source)[0] == "O(n^2)"


def test_sequential_loops_add_and_the_larger_wins() -> None:
    source = (
        "def f(xs):\n    for x in xs:\n        pass\n    for x in xs:\n        for y in xs:\n"
        "            pass\n"
    )
    assert answer(source)[0] == "O(n^2)"


def test_triangular_loops_are_quadratic() -> None:
    source = (
        "def f(xs):\n    for i in range(len(xs)):\n        for j in range(i):\n            pass\n"
    )
    assert answer(source)[0] == "O(n^2)"


def test_a_fixed_count_loop_is_constant() -> None:
    assert (
        answer("def f(c):\n    s = 0\n    for i in range(26):\n        s += c[i]\n    return s\n")[
            0
        ]
        == "O(1)"
    )


def test_a_countdown_range_is_linear() -> None:
    source = "def f(xs):\n    s = 0\n    for i in range(len(xs) - 1, -1, -1):\n        s += xs[i]\n    return s\n"
    assert answer(source)[0] == "O(n)"


# ------------------------------------------------------------------------------ logarithms
def test_halving_loop_is_logarithmic() -> None:
    assert answer("def f(n):\n    while n > 1:\n        n //= 2\n    return n\n")[0] == "O(log n)"


def test_doubling_loop_is_logarithmic() -> None:
    assert answer("def f(n):\n    p = 1\n    while p < n:\n        p *= 2\n    return p\n")[0] == (
        "O(log n)"
    )


def test_binary_search_is_logarithmic() -> None:
    source = (
        "def search(nums, t):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n        if nums[mid] == t:\n            return mid\n"
        "        if nums[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
        "    return -1\n"
    )
    assert answer(source)[0] == "O(log n)"


def test_square_root_loop() -> None:
    source = "def f(n):\n    i = 2\n    while i * i <= n:\n        i += 1\n    return i\n"
    assert answer(source)[0] == "O(sqrt n)"


# ----------------------------------------------------------------------------- allocations
def test_a_comprehension_builds_a_two_dimensional_table() -> None:
    source = "def f(n):\n    t = [[0] * n for _ in range(n)]\n    return t\n"
    assert answer(source) == ("O(n^2)", "O(n^2)")


def test_a_one_dimensional_allocation_is_linear_space() -> None:
    assert answer("def f(n):\n    return [0] * n\n") == ("O(n)", "O(n)")


def test_a_temporary_freed_each_iteration_is_not_accumulated() -> None:
    source = (
        "def f(n):\n    t = 0\n    for i in range(n):\n        tmp = [0] * n\n        t += tmp[0]\n"
        "    return t\n"
    )
    assert answer(source) == ("O(n^2)", "O(n)")


def test_appending_in_a_loop_grows_a_container_linearly() -> None:
    source = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out\n"
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------------------------------------ language-neutral
@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "cpp",
            "int f(vector<int>& a) {\n  int s = 0;\n  for (int i = 0; i < a.size(); i++) s += a[i];\n  return s;\n}\n",
        ),
        (
            "java",
            "class S { int f(int[] a) { int s = 0; for (int i = 0; i < a.length; i++) s += a[i]; return s; } }",
        ),
        (
            "javascript",
            "function f(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }",
        ),
        (
            "go",
            "func f(a []int) int { s := 0; for i := 0; i < len(a); i++ { s += a[i] }; return s }",
        ),
        (
            "c",
            "int f(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }",
        ),
    ],
)
def test_the_same_linear_loop_is_linear_in_every_language(language: str, source: str) -> None:
    assert answer(source, language) == ("O(n)", "O(1)")
