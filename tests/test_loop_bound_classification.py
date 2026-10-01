"""Loop-bound *shape* classification tests (plan §8's LOOP_CONST_BOUND /
LOOP_N_BOUND / LOOP_HALVING trio, `parsing/normalize.py:_classify_loop`).

Each test names the ONE loop-bound symbol a snippet should produce. The
condition-style cases are the ones that actually probe the reported bug: a
bound heuristic that only inspects the comparison's right-hand operand
can't tell "counting up to an input-derived limit written on the right"
(`i < n`) from "counting down to a fixed sentinel also written on the
right" (`count > 0`, `node is not None`) -- both put a syntactically
constant token on the right, but only the first is genuinely bound by that
constant. A `while` loop's compared variable is initialised somewhere
outside the loop node entirely, so its real trip count is never provable
from the loop node's own syntax without dataflow this static IR doesn't
attempt (the same scope limit `_classify_loop` already documents for the
while-loop HALVING idiom) -- so no `while` loop should ever be classified
LOOP_CONST_BOUND, whatever its condition looks like.
"""
from __future__ import annotations

from parsing.normalize import normalize_source

_LOOP_BOUND_SYMBOLS = ("LOOP_CONST_BOUND", "LOOP_N_BOUND", "LOOP_HALVING")


def _loop_symbol(src: str, language: str) -> str:
    hist = normalize_source(src, language).symbol_histogram()
    found = [s for s in _LOOP_BOUND_SYMBOLS if hist[s]]
    assert len(found) == 1, f"expected exactly one loop-bound symbol, got {found} in {hist}"
    return found[0]


def test_for_range_of_a_parameter_is_input_bound():
    src = "def f(n):\n    for i in range(n):\n        pass\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_for_range_of_a_literal_is_const_bound():
    src = "def f():\n    for i in range(10):\n        pass\n"
    assert _loop_symbol(src, "python") == "LOOP_CONST_BOUND"


def test_for_each_over_a_container_is_input_bound():
    src = "def f(arr):\n    for x in arr:\n        pass\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_while_counting_up_to_a_parameter_is_input_bound():
    src = "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_while_counting_up_to_a_literal_is_still_input_bound():
    # A while-loop's compared variable is initialised outside the loop
    # node entirely -- unlike a C-style for-loop's own init clause, there
    # is no syntax here proving `i` starts at 0, so a literal bound alone
    # is not enough to call this LOOP_CONST_BOUND.
    src = "def f():\n    i = 0\n    while i < 10:\n        i += 1\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_while_counting_down_to_zero_is_input_bound_not_const():
    # The reported bug: a fixed sentinel (0) on the right of the
    # comparison does not mean the loop runs a constant number of times --
    # `count`'s own starting value is what actually determines that, and
    # it is invisible here.
    src = "def f(count):\n    while count > 0:\n        count -= 1\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_while_null_check_traversal_is_input_bound_not_const():
    # Same bug, different sentinel: a linked-list walk compared against a
    # constant `None` is not O(1) -- it runs once per node.
    src = "def f(node):\n    while node is not None:\n        node = node.next\n"
    assert _loop_symbol(src, "python") == "LOOP_N_BOUND"


def test_cpp_for_loop_counting_up_to_a_parameter_is_input_bound():
    src = "int f(int n) {\n    for (int i = 0; i < n; i++) {\n    }\n    return 0;\n}\n"
    assert _loop_symbol(src, "cpp") == "LOOP_N_BOUND"


def test_cpp_for_loop_halving_update_is_halving():
    src = "int f(int n) {\n    for (int i = n; i > 0; i /= 2) {\n    }\n    return 0;\n}\n"
    assert _loop_symbol(src, "cpp") == "LOOP_HALVING"


def test_cpp_for_loop_counting_up_to_a_literal_is_const_bound():
    # Unlike a `while` loop, a C-style for-loop's own init clause is right
    # there in the same node -- `for (int i = 0; ...)` conventionally sets
    # the counter to a literal in the same statement, which this project
    # accepts as a reasonable non-dataflow signal (still just a heuristic,
    # not proof -- see `_classify_condition_loop`'s docstring).
    src = "int f() {\n    for (int i = 0; i < 10; i++) {\n    }\n    return 0;\n}\n"
    assert _loop_symbol(src, "cpp") == "LOOP_CONST_BOUND"
