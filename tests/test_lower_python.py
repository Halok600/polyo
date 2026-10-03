"""Python -> engine AST (`analysis/lower_python.py`), via the stdlib `ast` module.

Expected text is `render_module`'s canonical form, so each test reads as "this source lowers to
exactly this program"."""
from __future__ import annotations

import pytest

from analysis.lower_python import LoweringError, lower_python
from analysis.nodes import Call, FuncDef, Loop, walk
from analysis.nodes import render_module as render


def lowered(source: str) -> str:
    return render(lower_python(source))


def test_two_sum_lowers_to_the_obvious_program() -> None:
    source = """
def twoSum(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""
    assert lowered(source).splitlines() == [
        "def twoSum(nums, target):",
        "    n = len(nums)",
        "    for i in range(n):",
        "        for j in range(i + 1, n):",
        "            if nums[i] + nums[j] == target:",
        "                return [i, j]",
        "    return []",
    ]


def test_augmented_assignment_keeps_its_operator() -> None:
    text = lowered("def f(n):\n    n //= 2\n    p = 1\n    p *= 2\n    n >>= 1\n    return n\n")
    assert "n //= 2" in text and "p *= 2" in text and "n >>= 1" in text


def test_tuple_assignment_and_swap_are_kept_as_tuples() -> None:
    text = lowered("def f(a, b):\n    a, b = b, a % b\n    nums[i], nums[j] = nums[j], nums[i]\n")
    assert "(a, b) = (b, a % b)" in text
    assert "(nums[i], nums[j]) = (nums[j], nums[i])" in text


def test_chained_comparison_is_split_into_a_conjunction() -> None:
    assert "if 0 <= nr and nr < rows:" in lowered(
        "def f(nr, rows):\n    if 0 <= nr < rows:\n        return 1\n"
    )


def test_while_true_has_no_test_and_while_keeps_its_condition() -> None:
    text = lowered("def f(i):\n    while True:\n        i += 1\n    while i < 3:\n        i += 1\n")
    assert "while True:" in text and "while i < 3:" in text


def test_elif_chain_renders_as_nested_ifs() -> None:
    text = lowered(
        "def f(a):\n    if a == 1:\n        return 1\n    elif a == 2:\n        return 2\n"
        "    else:\n        return 3\n"
    )
    assert text.splitlines()[1:] == [
        "    if a == 1:",
        "        return 1",
        "    elif a == 2:",
        "        return 2",
        "    else:",
        "        return 3",
    ]


def test_methods_are_flattened_with_self_dropped_and_class_recorded() -> None:
    module = lower_python(
        "class Solution:\n    def twoSum(self, nums, target):\n        return nums\n"
    )
    (func,) = module.functions
    assert func.cls == "Solution" and func.name == "twoSum"
    assert [p.name for p in func.params] == ["nums", "target"]


def test_nested_function_stays_inside_its_enclosing_function() -> None:
    module = lower_python(
        "def outer(grid):\n    def dfs(i):\n        return dfs(i + 1)\n    return dfs(0)\n"
    )
    assert [f.name for f in module.functions] == ["outer"]
    nested = [n for n in walk(module.functions[0]) if isinstance(n, FuncDef) and n.name == "dfs"]
    assert len(nested) == 1


def test_main_guard_is_separated_from_module_code() -> None:
    module = lower_python(
        "def f(x):\n    return x\n\n\nif __name__ == '__main__':\n    print(f(3))\n"
    )
    assert module.toplevel == ()
    assert len(module.main_guard) == 1


def test_script_level_code_becomes_toplevel() -> None:
    module = lower_python("n = int(input())\ntotal = 0\nfor _ in range(n):\n    total += 1\n")
    assert module.functions == ()
    assert len(module.toplevel) == 3


def test_comprehensions_and_list_repetition_render_back() -> None:
    text = lowered("def f(n, m):\n    g = [[0] * m for _ in range(n)]\n    return [x for x in g if x]\n")
    assert "g = [[0] * m for _ in range(n)]" in text
    assert "return [x for x in g if x]" in text


def test_slices_and_negative_steps() -> None:
    text = lowered("def f(s, i):\n    a = s[i:]\n    b = s[::-1]\n    c = s[:i]\n    d = s[1:i:2]\n")
    assert "a = s[i:]" in text and "b = s[::-1]" in text
    assert "c = s[:i]" in text and "d = s[1:i:2]" in text


def test_annotations_become_typed_parameters() -> None:
    module = lower_python("def f(nums: List[int], k: int, g: list[list[int]]) -> int:\n    return k\n")
    (func,) = module.functions
    assert func.params[0].type is not None and func.params[0].type.render() == "List<int>"
    assert func.params[1].type is not None and func.params[1].type.render() == "int"
    assert func.params[2].type is not None and func.params[2].type.render() == "list<list<int>>"


def test_lambda_and_keyword_arguments_survive() -> None:
    text = lowered("def f(xs):\n    return sorted(xs, key=lambda v: v[0], reverse=True)\n")
    assert "return sorted(xs, key=lambda v: v[0], reverse=True)" in text


def test_for_else_runs_the_else_block_once_after_the_loop() -> None:
    text = lowered("def f(xs):\n    for x in xs:\n        pass\n    else:\n        done = 1\n")
    lines = text.splitlines()
    assert lines[-1] == "    done = 1"
    assert lines[-2] == "    for x in xs:" or lines[-2].startswith("    for x in xs")


def test_try_and_with_are_lowered_not_dropped() -> None:
    text = lowered(
        "def f(p):\n    with open(p) as fh:\n        x = fh.read()\n    try:\n        y = 1\n"
        "    except ValueError:\n        y = 2\n    finally:\n        z = 3\n"
    )
    assert "x = fh.read()" in text
    assert "try:" in text and "except:" in text and "finally:" in text


def test_docstrings_imports_and_pass_are_dropped() -> None:
    text = lowered('import heapq\n\n\ndef f(x):\n    """doc"""\n    pass\n    return x\n')
    assert text.splitlines() == ["def f(x):", "    return x"]


def test_assert_and_raise_do_not_crash_the_lowering() -> None:
    text = lowered("def f(x):\n    assert x > 0\n    if x > 9:\n        raise ValueError('big')\n    return x\n")
    assert "return x" in text


def test_calls_on_attributes_are_attribute_calls() -> None:
    module = lower_python("def f(h, x):\n    heapq.heappush(h, x)\n")
    calls = [n for n in walk(module.functions[0]) if isinstance(n, Call)]
    assert len(calls) == 1
    assert render_call(calls[0]) == "heapq.heappush(h, x)"


def render_call(call: Call) -> str:
    from analysis.nodes import render_expr

    return render_expr(call)


def test_loop_kinds() -> None:
    module = lower_python("def f(xs):\n    for x in xs:\n        pass\n    while xs:\n        xs.pop()\n")
    kinds = [n.kind for n in walk(module.functions[0]) if isinstance(n, Loop)]
    assert kinds == ["for_each", "while"]


def test_a_syntax_error_raises_lowering_error() -> None:
    with pytest.raises(LoweringError):
        lower_python("def f(:\n")


def test_unsupported_expression_becomes_unknown_not_an_exception() -> None:
    # `match` and the walrus are rare in solutions; they must degrade, never crash the engine.
    text = lowered("def f(xs):\n    if (n := len(xs)) > 2:\n        return n\n")
    assert "return n" in text
