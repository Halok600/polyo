"""IR -> tabular feature tests (plan §8 rung-0: `features/tabular.py`)."""
from __future__ import annotations

from features.tabular import extract_features
from parsing.normalize import normalize_source


def test_constant_time_function_has_zero_loop_depth():
    ir = normalize_source("def f(x):\n    return x + 1\n", "python")
    assert extract_features(ir).max_loop_nesting_depth == 0


def test_single_loop_has_depth_one():
    ir = normalize_source("def f(xs):\n    for x in xs:\n        print(x)\n", "python")
    assert extract_features(ir).max_loop_nesting_depth == 1


def test_nested_loops_have_depth_matching_nesting():
    src = "def f(n):\n    for i in range(n):\n        for j in range(n):\n            pass\n"
    ir = normalize_source(src, "python")
    assert extract_features(ir).max_loop_nesting_depth == 2


def test_sequential_loops_do_not_add_up_depth():
    src = "def f(n):\n    for i in range(n):\n        pass\n    for j in range(n):\n        pass\n"
    ir = normalize_source(src, "python")
    assert extract_features(ir).max_loop_nesting_depth == 1


def test_no_allocation_has_zero_alloc_depth():
    ir = normalize_source("def f(x):\n    return x + 1\n", "python")
    assert extract_features(ir).max_alloc_nesting_depth == 0


def test_allocation_outside_any_loop_has_alloc_depth_zero():
    ir = normalize_source("def f(n):\n    xs = [0] * n\n    return xs\n", "python")
    assert extract_features(ir).max_alloc_nesting_depth == 0


def test_allocation_inside_one_loop_has_alloc_depth_one():
    src = (
        "def f(n):\n"
        "    out = []\n"
        "    for i in range(n):\n"
        "        out.append([i])\n"
        "    return out\n"
    )
    ir = normalize_source(src, "python")
    assert extract_features(ir).max_alloc_nesting_depth == 1


def test_allocation_inside_nested_loops_has_alloc_depth_two():
    src = (
        "def f(n):\n"
        "    grid = []\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            grid.append({})\n"
        "    return grid\n"
    )
    ir = normalize_source(src, "python")
    assert extract_features(ir).max_alloc_nesting_depth == 2


def test_cpp_new_expression_inside_loop_has_alloc_depth_one():
    src = (
        "void f(int n) {\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        int* row = new int[n];\n"
        "    }\n"
        "}\n"
    )
    ir = normalize_source(src, "cpp")
    assert extract_features(ir).max_alloc_nesting_depth == 1
