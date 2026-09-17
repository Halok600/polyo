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


# -- rung-2 feature families (plan §8) --------------------------------------


def test_loop_count_by_depth_buckets_nested_loops_correctly():
    src = (
        "def f(n):\n"
        "    for i in range(n):\n"
        "        pass\n"
        "    for j in range(n):\n"
        "        for k in range(n):\n"
        "            pass\n"
    )
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.loop_count == 3
    assert features.loop_count_by_depth == (2, 1, 0, 0)


def test_loop_count_by_depth_caps_deep_nesting_in_the_last_bucket():
    depth = 5
    header = "def f(n):\n" + "".join(
        f"{'    ' * (i + 1)}for i{i} in range(n):\n" for i in range(depth)
    )
    body = "    " * (depth + 1) + "pass\n"
    ir = normalize_source(header + body, "python")
    features = extract_features(ir)
    assert sum(features.loop_count_by_depth) == depth
    assert features.loop_count_by_depth[3] == 2  # depths 3 and 4 both land in "3+"


def test_alloc_inside_loop_count_only_counts_allocs_under_a_loop():
    src = (
        "def f(n):\n"
        "    a = []\n"
        "    for i in range(n):\n"
        "        b = []\n"
        "    return a, b\n"
    )
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.alloc_count == 2
    assert features.alloc_inside_loop_count == 1


def test_recursion_shape_none_for_non_recursive_function():
    ir = normalize_source("def f(x):\n    return x + 1\n", "python")
    features = extract_features(ir)
    assert features.recursion_call_count == 0
    assert features.recursion_shape == "none"


def test_recursion_shape_single_for_one_recursive_call():
    src = "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n"
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.recursion_call_count == 1
    assert features.recursion_shape == "single"


def test_recursion_shape_multiple_for_branching_recursion():
    src = "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.recursion_call_count == 2
    assert features.recursion_shape == "multiple"


def test_library_call_counts():
    src = (
        "def f(arr):\n"
        "    arr = sorted(arr)\n"
        "    bisect.bisect_left(arr, 1)\n"
        "    heapq.heappush(arr, 1)\n"
        "    heapq.heappop(arr)\n"
        "    return max(arr)\n"
    )
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.sort_call_count == 1
    assert features.binary_search_call_count == 1
    assert features.heap_op_count == 2
    assert features.math_op_count == 1


def test_branch_break_continue_counts():
    src = (
        "def f(xs):\n"
        "    for x in xs:\n"
        "        if x < 0:\n"
        "            continue\n"
        "        if x > 100:\n"
        "            break\n"
    )
    ir = normalize_source(src, "python")
    features = extract_features(ir)
    assert features.branch_count == 2
    assert features.break_count == 1
    assert features.continue_count == 1


def test_as_dict_one_hot_encodes_recursion_shape():
    ir = normalize_source("def f(x):\n    return x + 1\n", "python")
    d = extract_features(ir).as_dict()
    assert d["recursion_shape_none"] == 1.0
    assert d["recursion_shape_single"] == 0.0
    assert d["recursion_shape_multiple"] == 0.0
    assert "recursion_shape" not in d


def test_as_dict_flattens_loop_count_by_depth():
    ir = normalize_source("def f(n):\n    for i in range(n):\n        pass\n", "python")
    d = extract_features(ir).as_dict()
    assert d["loop_count_depth_0"] == 1.0
    assert d["loop_count_depth_1"] == 0.0
