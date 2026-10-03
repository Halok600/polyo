"""Parameter roles (`analysis/roles.py`): which parameters are sizes, collections or linked nodes.

Python and JavaScript declare nothing, so the role is inferred from use; C, C++, Java and Go
declare types, which are read directly. A wrong role makes the whole analysis answer a different
question, so the evidence rules are pinned here."""

from __future__ import annotations

import pytest

from analysis.lower import lower_source
from analysis.nodes import TypeRef
from analysis.poly import Poly
from analysis.roles import infer_roles, params_to_values, value_for_type
from analysis.values import ContV, IntV, NodeV, ScalarV


def func_of(source: str, language: str = "python"):  # noqa: ANN201
    return lower_source(source, language).functions[0]


def role_names(source: str, language: str = "python") -> dict[str, str]:
    return {name: info.role for name, info in infer_roles(func_of(source, language)).items()}


# --------------------------------------------------------------------------- inference
def test_subscripted_and_iterated_and_len_params_are_collections() -> None:
    roles = role_names(
        "def f(nums, grid, text):\n    a = nums[0]\n    for row in grid:\n        pass\n    return len(text)\n"
    )
    assert roles == {"nums": "container", "grid": "container", "text": "container"}


def test_arithmetic_and_range_params_are_integers() -> None:
    roles = role_names("def f(n, k):\n    for i in range(n):\n        x = k + 1\n    return x\n")
    assert roles == {"n": "int", "k": "int"}


def test_a_param_only_compared_or_added_is_an_integer() -> None:
    assert role_names("def f(nums, target):\n    return nums[0] + target\n")["target"] == "int"


def test_attribute_access_to_next_left_right_marks_a_linked_node() -> None:
    roles = role_names("def f(head, root):\n    a = head.next\n    return root.left\n")
    assert roles == {"head": "node", "root": "node"}


def test_collection_methods_mark_a_collection() -> None:
    assert role_names("def f(items):\n    items.append(1)\n")["items"] == "container"


def test_string_methods_mark_a_string() -> None:
    infos = infer_roles(func_of("def f(s):\n    return s.lower().strip()\n"))
    assert infos["s"].role == "container" and infos["s"].kind == "str"


def test_nested_subscripts_give_a_two_dimensional_collection() -> None:
    infos = infer_roles(func_of("def f(grid):\n    return grid[0][1]\n"))
    assert infos["grid"].depth == 2


def test_iterating_a_row_of_a_collection_counts_as_a_second_dimension() -> None:
    infos = infer_roles(
        func_of("def f(grid):\n    for row in grid:\n        for x in row:\n            pass\n")
    )
    assert infos["grid"].depth == 2


def test_len_of_a_row_counts_as_a_second_dimension() -> None:
    infos = infer_roles(func_of("def f(grid):\n    return len(grid[0])\n"))
    assert infos["grid"].depth == 2


def test_an_unused_or_ambiguous_param_defaults_to_an_integer() -> None:
    assert role_names("def f(x):\n    return 1\n")["x"] == "int"


def test_javascript_params_use_the_same_inference() -> None:
    roles = role_names("function f(nums, k) {\n  return nums.length + k;\n}\n", "javascript")
    assert roles["nums"] == "container" and roles["k"] == "int"


# ------------------------------------------------------------------ values from roles
def test_params_to_values_makes_size_variables() -> None:
    func = func_of("def f(nums, n):\n    return len(nums) + n\n")
    values = params_to_values(func, "python")
    assert isinstance(values["nums"], ContV) and values["nums"].length.vars()
    assert isinstance(values["n"], IntV) and values["n"].mag is not None
    assert values["nums"].length != values["n"].mag  # two different variables


def test_two_dimensional_param_gets_an_outer_and_an_inner_length() -> None:
    func = func_of("def f(grid):\n    return grid[0][0]\n")
    grid = params_to_values(func, "python")["grid"]
    assert isinstance(grid, ContV) and isinstance(grid.elem, ContV)
    assert grid.length != grid.elem.length


def test_node_param_becomes_a_node_value() -> None:
    func = func_of("def f(head):\n    return head.next\n")
    assert isinstance(params_to_values(func, "python")["head"], NodeV)


# ------------------------------------------------------------------- declared types
@pytest.mark.parametrize(
    ("type_ref", "kind"),
    [
        (TypeRef("vector", (TypeRef("int"),)), "list"),
        (TypeRef("List", (TypeRef("Integer"),)), "list"),
        (TypeRef("int", dims=1), "array"),
        (TypeRef("string"), "str"),
        (TypeRef("String"), "str"),
        (TypeRef("unordered_map", (TypeRef("int"), TypeRef("int"))), "dict"),
        (TypeRef("HashSet", (TypeRef("Integer"),)), "set"),
        (TypeRef("priority_queue", (TypeRef("int"),)), "heap"),
        (TypeRef("TreeMap", (TypeRef("Integer"), TypeRef("Integer"))), "treemap"),
        (TypeRef("ArrayDeque", (TypeRef("Integer"),)), "deque"),
    ],
)
def test_declared_container_types(type_ref: TypeRef, kind: str) -> None:
    value = value_for_type(type_ref, "p")
    assert isinstance(value, ContV) and value.kind == kind


def test_declared_integer_and_scalar_types() -> None:
    assert isinstance(value_for_type(TypeRef("int"), "n"), IntV)
    assert isinstance(value_for_type(TypeRef("long"), "n"), IntV)
    assert isinstance(value_for_type(TypeRef("boolean"), "b"), ScalarV)
    assert isinstance(value_for_type(TypeRef("double"), "d"), ScalarV)


def test_declared_user_type_is_a_node() -> None:
    assert isinstance(value_for_type(TypeRef("ListNode"), "head"), NodeV)
    assert isinstance(value_for_type(TypeRef("TreeNode"), "root"), NodeV)


def test_declared_nested_container_has_a_row_with_its_own_length() -> None:
    value = value_for_type(TypeRef("vector", (TypeRef("vector", (TypeRef("int"),)),)), "g")
    assert isinstance(value, ContV) and isinstance(value.elem, ContV)
    assert value.length != value.elem.length


def test_two_dimensional_array_type() -> None:
    value = value_for_type(TypeRef("int", dims=2), "m")
    assert isinstance(value, ContV) and isinstance(value.elem, ContV)


def test_typed_params_in_cpp_java_go() -> None:
    cpp = params_to_values(func_of("int f(vector<int>& a, int n) { return n; }", "cpp"), "cpp")
    assert isinstance(cpp["a"], ContV) and isinstance(cpp["n"], IntV)
    java = params_to_values(
        func_of("class S { int f(int[] a, int n) { return n; } }", "java"), "java"
    )
    assert isinstance(java["a"], ContV) and isinstance(java["n"], IntV)
    go = params_to_values(func_of("func f(a []int, n int) int { return n }", "go"), "go")
    assert isinstance(go["a"], ContV) and isinstance(go["n"], IntV)


def test_integer_param_magnitude_is_a_polynomial_in_its_own_variable() -> None:
    func = func_of("def f(n):\n    for i in range(n):\n        pass\n")
    mag = params_to_values(func, "python")["n"].mag  # type: ignore[union-attr]
    assert isinstance(mag, Poly) and len(mag.vars()) == 1


def test_max_of_two_values_is_not_evidence_that_they_are_collections() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    source = "def f(nums):\n    best = 0\n    for i in range(len(nums)):\n        best = max(best, nums[i])\n    return best\n"
    func = lower_source(source, "python").functions[0]
    assert infer_roles(func)["nums"].depth == 1


def test_a_spread_marks_its_operand_as_a_collection() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source("function f(nums) { return [...nums]; }", "javascript").functions[0]
    assert infer_roles(func)["nums"].role == "container"


def test_new_set_of_a_parameter_marks_it_as_a_collection() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source("function f(nums) { return new Set(nums).size; }", "javascript").functions[
        0
    ]
    assert infer_roles(func)["nums"].role == "container"


def test_a_parameter_named_like_a_graph_has_ragged_rows() -> None:
    from analysis.lower import lower_source
    from analysis.roles import params_to_values
    from analysis.values import ContV

    func = lower_source(
        "def f(graph, grid):\n    return graph[0][0] + grid[0][0]\n", "python"
    ).functions[0]
    values = params_to_values(func, "python")
    assert isinstance(values["graph"], ContV) and values["graph"].ragged
    assert isinstance(values["grid"], ContV) and not values["grid"].ragged


def test_a_typed_nested_vector_named_adj_has_ragged_rows() -> None:
    from analysis.lower import lower_source
    from analysis.roles import params_to_values
    from analysis.values import ContV

    func = lower_source(
        "int f(vector<vector<int>>& adj, vector<vector<int>>& board) { return 0; }", "cpp"
    ).functions[0]
    values = params_to_values(func, "cpp")
    assert isinstance(values["adj"], ContV) and values["adj"].ragged
    assert isinstance(values["board"], ContV) and not values["board"].ragged


def test_a_comprehension_variable_is_one_level_below_the_collection_it_walks() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source("def f(grid):\n    return [row[:] for row in grid]\n", "python").functions[
        0
    ]
    assert infer_roles(func)["grid"].depth == 2


def test_a_module_call_that_consumes_a_collection_marks_its_argument() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source(
        "import copy\n\n\ndef f(grid):\n    return copy.deepcopy(grid)\n", "python"
    ).functions[0]
    assert infer_roles(func)["grid"].role == "container"


def test_only_the_first_argument_of_a_module_call_is_the_collection() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    source = (
        "import heapq\n\n\ndef f(nums):\n    heap = []\n    for x in nums:\n        heapq.heappush(heap, x)\n"
        "    return heap\n"
    )
    assert infer_roles(lower_source(source, "python").functions[0])["nums"].depth == 1


def test_math_max_of_two_values_does_not_make_them_collections() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source(
        "function f(nums) { let b = 0; for (let i = 0; i < nums.length; i++) b = Math.max(b, nums[i]); return b; }",
        "javascript",
    ).functions[0]
    assert infer_roles(func)["nums"].depth == 1


def test_a_callback_parameter_is_one_level_below_the_collection_it_is_mapped_over() -> None:
    from analysis.lower import lower_source
    from analysis.roles import infer_roles

    func = lower_source(
        "function f(grid) { return grid.map((row) => [...row]); }", "javascript"
    ).functions[0]
    assert infer_roles(func)["grid"].depth == 2


def test_rows_of_a_parameter_named_like_a_list_of_pairs_have_constant_length() -> None:
    from analysis.lower import lower_source
    from analysis.roles import params_to_values
    from analysis.values import ContV

    func = lower_source(
        "int f(vector<vector<int>>& intervals, vector<vector<int>>& grid) { return 0; }", "cpp"
    ).functions[0]
    values = params_to_values(func, "cpp")
    intervals, grid = values["intervals"], values["grid"]
    assert isinstance(intervals, ContV) and isinstance(intervals.elem, ContV)
    assert intervals.elem.length.const_value() == 2
    assert isinstance(grid, ContV) and isinstance(grid.elem, ContV)
    assert grid.elem.length.const_value() is None
