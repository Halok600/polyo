"""Engine gaps found by the first-contact run on the blind corpus (tests/data/blind_v1_cases.json).

Each test names the construct, uses a small program that isolates it, and (where the blind program
itself is the clearest statement) the blind program too. They were written BEFORE the fix and
failed first; the first-contact record in tests/data/blind_v1_first_contact.json is never edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.engine import analyze

_BLIND = {
    case["id"]: case
    for case in json.loads(
        Path(__file__).with_name("data").joinpath("blind_v1_cases.json").read_text("utf-8")
    )
}


def _answer(source: str, language: str = "python") -> tuple[str, str, str]:
    result = analyze(source, language)
    return result.time.cls, result.space.cls, result.certainty


def _blind(case_id: str) -> tuple[str, str]:
    case = _BLIND[case_id]
    result = analyze(case["source"], case["language"])
    return result.time.cls, result.space.cls


# ------------------------------------------------------------------ defaultdict and dict values
def test_a_defaultdict_gains_a_key_when_it_is_subscripted() -> None:
    source = (
        "def f(nums):\n"
        "    groups = defaultdict(list)\n"
        "    for x in nums:\n"
        "        groups[x].append(x)\n"
        "    return len(groups)\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_iterating_the_rows_of_a_dict_of_lists_visits_every_entry_once() -> None:
    source = (
        "def f(nums):\n"
        "    groups = defaultdict(list)\n"
        "    for x in nums:\n"
        "        groups[x].append(x)\n"
        "    t = 0\n"
        "    for rows in groups.values():\n"
        "        for v in rows:\n"
        "            t += v\n"
        "    return t\n"
    )
    assert _answer(source) == ("O(n)", "O(n)", "certain")


def test_setdefault_rows_are_summed_the_same_way() -> None:
    source = (
        "def f(nums):\n"
        "    groups = {}\n"
        "    for x in nums:\n"
        "        groups.setdefault(x, []).append(x)\n"
        "    t = 0\n"
        "    for rows in groups.values():\n"
        "        for v in rows:\n"
        "            t += v\n"
        "    return t\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_the_blind_dict_of_lists_program() -> None:
    assert _blind("sum_of_distances_dict_of_lists_python") == ("O(n)", "O(n)")


# ------------------------------------------------------------------ never silently zero
def test_a_container_the_engine_cannot_track_is_assumed_not_empty() -> None:
    """Growth hidden behind a call the engine cannot see must not turn a loop over the container
    into O(1) with full certainty: that is an under-estimate presented as a proof."""
    source = (
        "def f(items):\n"
        "    seen = []\n"
        "    register(seen, items)\n"
        "    t = 0\n"
        "    for x in seen:\n"
        "        t += x\n"
        "    return t\n"
    )
    time_cls, _space, certainty = _answer(source)
    assert time_cls == "O(n)" and certainty != "certain"


def test_a_lower_order_term_survives_a_container_that_resolves_to_nothing() -> None:
    """`n + n * k` was reduced to `n * k` before `k` (a container that never grew) was resolved to
    zero, which erased the whole cost."""
    source = (
        "def f(nums):\n"
        "    groups = defaultdict(list)\n"
        "    for i, x in enumerate(nums):\n"
        "        groups[x].append(i)\n"
        "    t = 0\n"
        "    for idxs in groups.values():\n"
        "        for k, i in enumerate(idxs):\n"
        "            t += i\n"
        "    return t\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


# ------------------------------------------------------------------ keys, method references, seeds
def test_a_java_method_reference_is_not_an_assumption() -> None:
    source = (
        "import java.util.*;\n"
        "class Solution {\n"
        "    int f(int[] nums) {\n"
        "        Map<Integer, Integer> counts = new HashMap<>();\n"
        "        for (int x : nums) {\n"
        "            counts.merge(x, 1, Integer::sum);\n"
        "        }\n"
        "        return counts.size();\n"
        "    }\n"
        "}\n"
    )
    time_cls, space_cls, certainty = _answer(source, "java")
    assert (time_cls, space_cls, certainty) == ("O(n)", "O(n)", "certain")


def test_merge_with_a_data_key_grows_the_map_even_after_a_constant_put() -> None:
    assert _blind("subarray_sum_equals_k_prefix_map_java") == ("O(n)", "O(n)")


def test_an_insert_whose_key_is_not_seen_makes_the_distinct_count_unknown() -> None:
    source = (
        "def f(nums):\n"
        "    seen = set()\n"
        "    seen.add(0)\n"
        "    for x in nums:\n"
        "        seen.update([x])\n"
        "    return len(seen)\n"
    )
    assert _answer(source)[1] == "O(n)"


def test_a_visited_set_of_grid_cells_is_as_big_as_the_grid() -> None:
    assert _blind("num_islands_bfs_python") == ("O(n^2)", "O(n^2)")


# ------------------------------------------------------------------ what a loop test costs
def test_strlen_in_a_c_loop_condition_is_paid_on_every_iteration() -> None:
    assert _blind("count_uppercase_strlen_condition_c") == ("O(n^2)", "O(1)")


def test_a_constant_time_loop_condition_adds_nothing() -> None:
    source = (
        "int f(const char *s, int n) {\n"
        "    int c = 0;\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        if (s[i] == 'a') c++;\n"
        "    }\n"
        "    return c;\n"
        "}\n"
    )
    assert _answer(source, "c")[:2] == ("O(n)", "O(1)")


def test_len_in_a_python_while_condition_is_constant_time() -> None:
    source = (
        "def f(s):\n    i = 0\n    c = 0\n    while i < len(s):\n        c += 1\n        i += 1\n"
        "    return c\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(1)")


def test_a_sizeable_call_in_a_while_condition_is_paid_each_time() -> None:
    source = (
        "def f(items):\n"
        "    i = 0\n"
        "    while i < len(items) and items[i:].count(0) > 0:\n"
        "        i += 1\n"
        "    return i\n"
    )
    assert _answer(source)[0] == "O(n^2)"


def test_rows_stored_into_a_dict_one_by_one_are_summed_the_same_way() -> None:
    source = (
        "def f(nums):\n"
        "    groups = {}\n"
        "    for x in nums:\n"
        "        if x not in groups:\n"
        "            groups[x] = []\n"
        "        groups[x].append(x)\n"
        "    t = 0\n"
        "    for rows in groups.values():\n"
        "        for v in rows:\n"
        "            t += v\n"
        "    return t\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


# ------------------------------------------------------------------ growth through helpers
_CLOSURE_HELPER = (
    "def f(nums):\n    out = []\n\n    def add(x):\n        out.append(x)\n\n"
    "    for x in nums:\n        add(x)\n    return out\n"
)
_PARAM_HELPER = (
    "def add(res, x):\n    res.append(x)\n\n\ndef f(nums):\n    out = []\n"
    "    for x in nums:\n        add(out, x)\n    return out\n"
)
_PARAM_CHAIN = (
    "def inner(res):\n    res.append(1)\n\n\ndef outer(res):\n    inner(res)\n\n\n"
    "def f(nums):\n    out = []\n    for x in nums:\n        outer(out)\n    return out\n"
)
_READ_ONLY_HELPER = (
    "def total(res):\n    t = 0\n    for x in res:\n        t += x\n    return t\n\n\n"
    "def f(nums):\n    return total(nums)\n"
)


@pytest.mark.parametrize("source", [_CLOSURE_HELPER, _PARAM_HELPER, _PARAM_CHAIN])
def test_appending_through_a_helper_grows_the_callers_container(source: str) -> None:
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_a_helper_that_only_reads_its_container_adds_no_space() -> None:
    assert _answer(_READ_ONLY_HELPER)[1] == "O(1)"


def test_a_cpp_reference_parameter_is_the_callers_container() -> None:
    source = (
        "void add(vector<int>& r, int x) { r.push_back(x); }\n"
        "vector<int> f(vector<int>& nums) {\n"
        "    vector<int> out;\n"
        "    for (int x : nums) add(out, x);\n"
        "    return out;\n"
        "}\n"
    )
    assert _answer(source, "cpp")[:2] == ("O(n)", "O(n)")


def test_a_cpp_by_value_parameter_is_a_copy_so_it_has_no_growth_variable() -> None:
    from analysis.interp import Interp
    from analysis.lower import lower_source

    source = (
        "int by_value(vector<int> r, int x) { r.push_back(x); return r.size(); }\n"
        "int by_ref(vector<int>& r, int x) { r.push_back(x); return r.size(); }\n"
    )
    module = lower_source(source, "cpp")
    interp = Interp(module, "cpp")
    by_value, by_ref = (interp.summary_for(f) for f in module.functions)
    assert by_value.param_caps == {}
    assert set(by_ref.param_caps) == {"r"}


def test_a_java_helper_adding_to_a_list_parameter() -> None:
    source = (
        "import java.util.*;\n"
        "class Solution {\n"
        "    void add(List<Integer> res, int x) { res.add(x); }\n"
        "    List<Integer> f(int[] nums) {\n"
        "        List<Integer> out = new ArrayList<>();\n"
        "        for (int x : nums) add(out, x);\n"
        "        return out;\n"
        "    }\n"
        "}\n"
    )
    assert _answer(source, "java")[:2] == ("O(n)", "O(n)")


def test_a_recursion_collecting_into_a_closure_list() -> None:
    source = (
        "def f(nums):\n    out = []\n\n    def go(i):\n        if i == len(nums):\n"
        "            return\n        out.append(nums[i])\n        go(i + 1)\n\n    go(0)\n"
        "    return out\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_a_recursion_collecting_into_a_list_parameter() -> None:
    source = (
        "def go(nums, i, out):\n    if i == len(nums):\n        return\n"
        "    out.append(nums[i])\n    go(nums, i + 1, out)\n\n\n"
        "def f(nums):\n    out = []\n    go(nums, 0, out)\n    return out\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_backtracking_that_copies_its_path_into_the_result() -> None:
    assert _blind("subsets_python")[1] == "O(n^2)"  # n * 2^n entries, above n^2
    assert _blind("permutations_cpp")[1] == "O(n^2)"  # n * n! entries


def test_a_path_that_is_pushed_and_popped_does_not_grow_without_bound() -> None:
    """`path` goes up and down with the recursion, so it is never longer than the depth: the
    results are what grow (one number per leaf), not the path."""
    source = (
        "def f(nums):\n    result = []\n    path = []\n\n    def go(i):\n"
        "        if i == len(nums):\n            result.append(len(path))\n            return\n"
        "        path.append(nums[i])\n        go(i + 1)\n        path.pop()\n"
        "        go(i + 1)\n\n    go(0)\n    return result\n"
    )
    assert _answer(source)[1] == "O(n^2)"  # 2^n results: above the n^2 ceiling of the taxonomy


def test_the_blind_tree_paths_program() -> None:
    assert _blind("root_to_leaf_paths_python") == ("O(n^2)", "O(n^2)")


# ------------------------------------------------------------------ worklists over adjacency lists
def test_a_breadth_first_walk_over_rows_is_vertices_plus_edges() -> None:
    assert _blind("is_bipartite_java")[0] == "O(n)"


def test_a_walk_over_a_locally_built_adjacency_list_is_vertices_plus_edges() -> None:
    assert _blind("count_components_go")[0] == "O(n)"


def test_the_same_walk_in_python_with_a_set_is_unchanged() -> None:
    source = (
        "def f(graph):\n"
        "    seen = set()\n"
        "    count = 0\n"
        "    for s in range(len(graph)):\n"
        "        if s in seen:\n"
        "            continue\n"
        "        count += 1\n"
        "        stack = [s]\n"
        "        seen.add(s)\n"
        "        while stack:\n"
        "            u = stack.pop()\n"
        "            for v in graph[u]:\n"
        "                if v not in seen:\n"
        "                    seen.add(v)\n"
        "                    stack.append(v)\n"
        "    return count\n"
    )
    assert _answer(source)[:2] == ("O(n)", "O(n)")


# ------------------------------------------------------------------ pointers that only move one way
def test_a_staircase_walk_with_a_decreasing_column_pointer_is_rows_plus_columns() -> None:
    assert _blind("count_negatives_sorted_grid_javascript")[0] == "O(n)"


def test_a_data_dependent_conjunct_cannot_hide_a_numeric_bound() -> None:
    source = """
def f(grid, m):
    col = m
    count = 0
    for row in grid:
        while col > 0 and row[col - 1] < 0:
            col -= 1
        count += m - col
    return count
"""
    assert _answer(source)[0] == "O(n)"


def test_a_loop_bound_read_from_data_alone_is_still_not_provable() -> None:
    source = """
def f(a, weights):
    w = 100
    t = 0
    while w >= weights[0]:
        w -= 1
        t += 1
    return t
"""
    assert _answer(source)[2] != "certain"


# ------------------------------------------------------------------ one pointer handed to another
_RUNS_PY = """
def f(s):
    i = 0
    runs = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        runs += 1
        i = j
    return runs
"""
_RUNS_NEXT_PY = """
def f(s):
    n = len(s)
    i = 0
    count = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        count += 1
        i = j
    return count
"""
_RUNS_C = """
int f(const char *s, int n) {
    int i = 0, runs = 0;
    while (i < n) {
        int j = i;
        while (j < n && s[j] == s[i]) j++;
        runs++;
        i = j;
    }
    return runs;
}
"""
_RESCAN_FROM_I = """
def f(a):
    n = len(a)
    i = 0
    best = 0
    while i < n:
        j = i
        while j < n and a[j] != 0:
            j += 1
        best = max(best, j - i)
        i += 1
    return best
"""
_RESET_TO_ZERO = """
def f(a):
    n = len(a)
    i = 0
    while i < n:
        j = 0
        while j < n and a[j] != a[i]:
            j += 1
        i += 1
    return i
"""


@pytest.mark.parametrize("source", [_RUNS_PY, _RUNS_NEXT_PY])
def test_scanning_runs_with_a_hand_off_pointer_is_linear(source: str) -> None:
    assert _answer(source)[:2] == ("O(n)", "O(1)")


def test_the_same_scan_in_c_is_linear() -> None:
    assert _answer(_RUNS_C, "c")[:2] == ("O(n)", "O(1)")


def test_a_pointer_that_restarts_at_the_outer_one_without_catching_up_is_quadratic() -> None:
    """`j = i` but `i` only moves by one: every inner scan can run to the end."""
    assert _answer(_RESCAN_FROM_I)[0] == "O(n^2)"


def test_a_pointer_reset_to_a_constant_is_quadratic() -> None:
    assert _answer(_RESET_TO_ZERO)[0] == "O(n^2)"


def test_the_leaked_stride_variable_never_shows_in_the_answer() -> None:
    result = analyze(_RUNS_PY, "python")
    assert "^-" not in result.time.text and "^-" not in result.space.text


# ------------------------------------------------------------------ nodes the function links itself
def test_walking_a_list_that_this_function_linked_together_costs_its_length() -> None:
    assert _blind("insertion_sort_list_javascript")[0] == "O(n^2)"


def test_a_list_built_from_fresh_nodes_is_as_long_as_the_loop_that_built_it() -> None:
    source = """
class Solution {
    int f(int[] nums) {
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        for (int x : nums) {
            tail.next = new ListNode(x);
            tail = tail.next;
        }
        int count = 0;
        ListNode cur = dummy.next;
        while (cur != null) {
            count++;
            cur = cur.next;
        }
        return count;
    }
}
"""
    assert _answer(source, "java")[0] == "O(n)"


def test_a_single_fresh_node_that_is_never_linked_is_one_node() -> None:
    source = """
function f(n) {
    let node = new ListNode(n);
    let total = 0;
    while (node !== null) {
        total += node.val;
        node = node.next;
    }
    return total;
}
"""
    assert _answer(source, "javascript")[0] == "O(1)"


# ------------------------------------------------------------------ memoised interval recursion
def test_a_two_dimensional_memo_table_is_keyed_by_both_indices() -> None:
    assert _blind("matrix_chain_memo_cpp") == ("O(n^3)", "O(n^2)")


def test_a_memoised_interval_recursion_in_python_is_cubic() -> None:
    source = """
def best(dims):
    n = len(dims)
    memo = [[-1] * n for _ in range(n)]

    def solve(i, j):
        if i >= j:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]
        result = float("inf")
        for k in range(i, j):
            cost = solve(i, k) + solve(k + 1, j) + dims[i - 1] * dims[k] * dims[j]
            result = min(result, cost)
        memo[i][j] = result
        return result

    return solve(1, n - 1)
"""
    assert _answer(source)[:2] == ("O(n^3)", "O(n^2)")


def test_a_memoised_recursion_with_constant_work_per_state_stays_quadratic() -> None:
    source = """
def paths(n, m):
    memo = [[-1] * m for _ in range(n)]

    def go(r, c):
        if r == n - 1 or c == m - 1:
            return 1
        if memo[r][c] != -1:
            return memo[r][c]
        memo[r][c] = go(r + 1, c) + go(r, c + 1)
        return memo[r][c]

    return go(0, 0)
"""
    assert _answer(source)[0] == "O(n^2)"


# ------------------------------------------------------------------ names that stand for a type
def test_a_cpp_typedef_of_a_vector_is_a_vector() -> None:
    source = """
typedef vector<int> vi;
vi squares(int n) {
    vi out;
    for (int i = 0; i < n; i++) out.push_back(i * i);
    return out;
}
"""
    assert _answer(source, "cpp")[:2] == ("O(n)", "O(n)")


def test_a_cpp_using_alias_of_a_nested_vector_is_a_grid() -> None:
    source = """
using Grid = vector<vector<int>>;
Grid make(int n, int m) {
    Grid g(n, vector<int>(m, 0));
    return g;
}
"""
    assert _answer(source, "cpp")[:2] == ("O(n^2)", "O(n^2)")


def test_a_typedef_of_a_number_or_a_struct_is_left_alone() -> None:
    source = """
typedef long long ll;
struct Node { int val; Node *next; };
ll total(int n) {
    ll t = 0;
    for (int i = 0; i < n; i++) t += i;
    return t;
}
"""
    assert _answer(source, "cpp")[:2] == ("O(n)", "O(1)")


def test_a_go_named_slice_type_is_a_slice() -> None:
    source = """
package main

type stack []int

func squares(n int) stack {
    s := stack{}
    for i := 0; i < n; i++ {
        s = append(s, i*i)
    }
    return s
}
"""
    assert _answer(source, "go")[:2] == ("O(n)", "O(n)")


def test_a_go_heap_over_a_named_slice_type_grows_with_its_pushes() -> None:
    assert _blind("meeting_rooms_heap_go") == ("O(n log n)", "O(n)")


# ------------------------------------------------------------------ recursion on the two sides of a split
def test_quicksort_on_an_index_range_is_quadratic_with_a_linear_stack() -> None:
    assert _blind("quicksort_lomuto_java") == ("O(n^2)", "O(n)")


def test_the_range_split_is_an_assumption_not_a_proof() -> None:
    result = analyze(_BLIND["quicksort_lomuto_java"]["source"], "java")
    assert result.certainty == "assumed"
    assert any("split" in a.reason for a in result.assumptions)


def test_quicksort_in_python_with_a_helper_that_returns_the_pivot() -> None:
    source = """
def quicksort(a, lo, hi):
    if lo < hi:
        p = partition(a, lo, hi)
        quicksort(a, lo, p - 1)
        quicksort(a, p + 1, hi)


def partition(a, lo, hi):
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i
"""
    assert _answer(source)[0] == "O(n^2)"


def test_two_calls_that_each_take_one_off_are_still_exponential() -> None:
    source = """
def f(lo, hi):
    if lo >= hi:
        return 0
    return f(lo, hi - 1) + f(lo + 1, hi)
"""
    assert _answer(source)[0] == "O(2^n)"


def test_ranges_that_both_keep_the_split_point_are_not_a_proven_split() -> None:
    """f(lo, p) and f(p, hi) overlap at p: nothing says the ranges ever shrink."""
    source = """
def f(a, lo, hi):
    if lo >= hi:
        return
    p = (lo + hi) // 2 if a[lo] else hi
    f(a, lo, p)
    f(a, p, hi)
"""
    assert _answer(source)[2] == "unknown"


# ------------------------------------------------------------------ a container that feeds itself
_DOUBLING_IN_PLACE = """
def f(nums):
    res = [[]]
    for x in nums:
        for r in list(res):
            res.append(r + [x])
    return res
"""
_DOUBLING_EXTEND = """
def f(nums):
    res = [[]]
    for x in nums:
        res += [r + [x] for r in res]
    return res
"""
_LETTERS = """
def f(digits):
    mapping = {'2': 'abc', '3': 'def', '4': 'ghi'}
    res = ['']
    for d in digits:
        res = [p + c for p in res for c in mapping[d]]
    return res
"""
_LEVEL_LISTS = """
def f(n):
    cur = [1]
    for _ in range(n):
        nxt = []
        for x in cur:
            nxt.append(x)
            nxt.append(x + 1)
        cur = nxt
    return cur
"""
_ONE_PASS_ONLY = """
def f(m):
    res = [0]
    for r in list(res):
        for j in range(m):
            res.append(r)
    return res
"""
_FIXED_ROUNDS = """
def f(a):
    res = list(a)
    for _ in range(3):
        res = res + res
    return res
"""


@pytest.mark.parametrize("source", [_DOUBLING_IN_PLACE, _DOUBLING_EXTEND, _LETTERS])
def test_a_list_that_multiplies_every_round_is_exponential_in_the_rounds(source: str) -> None:
    assert _answer(source)[0] == "O(2^n)"


def test_the_blind_doubling_program() -> None:
    assert _blind("all_subsequences_strings_javascript") == ("O(2^n)", "O(n^2)")


def test_one_pass_over_a_snapshot_is_never_called_exponential() -> None:
    """Over-estimated (the snapshot is taken to be as long as the final list), but not 2^n."""
    assert _answer(_ONE_PASS_ONLY)[0] != "O(2^n)"


def test_a_fixed_number_of_doublings_is_a_constant_factor() -> None:
    assert _answer(_FIXED_ROUNDS)[:2] == ("O(n)", "O(n)")


_TREE_LEVELS_BY_COMPREHENSION = """
def levels(root):
    out = []
    level = [root] if root else []
    while level:
        out.append([n.val for n in level])
        level = [c for n in level for c in (n.left, n.right) if c]
    return out
"""
_NARY_LEVELS = """
def levels(root):
    out = []
    level = [root] if root else []
    while level:
        out.append([n.val for n in level])
        level = [c for n in level for c in n.children]
    return out
"""


@pytest.mark.parametrize("source", [_TREE_LEVELS_BY_COMPREHENSION, _NARY_LEVELS])
def test_a_tree_walked_level_by_level_is_not_exponential(source: str) -> None:
    """The next level is built out of the children of this one: bounded by the nodes in the tree,
    not multiplied by a constant each round."""
    assert _answer(source)[0] != "O(2^n)"


# ------------------------------------------------------------------ reading standard input
def test_a_line_unpacked_into_names_has_exactly_that_many_tokens() -> None:
    source = """
a, b, x = map(int, input().split())
print((b // x) - ((a - 1) // x))
"""
    assert _answer(source)[:2] == ("O(1)", "O(1)")


def test_reading_pairs_in_a_loop_is_linear_in_the_number_of_pairs() -> None:
    source = """
n = int(input())
best = 0
for _ in range(n):
    x, y = map(int, input().split())
    best = max(best, x + y)
print(best)
"""
    assert _answer(source)[:2] == ("O(n)", "O(1)")


def test_counting_pairs_read_one_per_line_stores_at_most_one_key_per_line() -> None:
    source = """
n = int(input())
count = {}
for _ in range(n):
    a, b = input().split()
    count[(a, b)] = count.get((a, b), 0) + 1
print(max(count.values()))
"""
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_a_whole_line_kept_as_a_list_is_as_long_as_the_input() -> None:
    source = """
n = int(input())
xs = list(map(int, input().split()))
print(sum(xs))
"""
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_a_grid_read_row_by_row_is_rows_times_columns() -> None:
    source = """
n, m = map(int, input().split())
grid = [input().split() for _ in range(n)]
print(len(grid))
"""
    assert _answer(source)[:2] == ("O(n^2)", "O(n^2)")


def test_tokens_used_only_by_constant_index_are_a_constant_number() -> None:
    source = """
import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String[] s = in.readLine().split(" ");
        int n = Integer.parseInt(s[0]);
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            s = in.readLine().split(" ");
            a[i] = Integer.parseInt(s[0]);
        }
        Arrays.sort(a);
        System.out.println(a[0]);
    }
}
"""
    assert _answer(source, "java")[:2] == ("O(n log n)", "O(n)")


# ------------------------------------------------------------------ a call on something unknown
def test_splitting_an_unknown_string_is_an_assumed_input_sized_list_not_one_token() -> None:
    """The separator used to be taken for the subject, so `line.split(" ")` on a value the engine
    knows nothing about was a list of ONE token and every loop over it free."""
    source = """
class A {
    void f() {
        String line = mystery();
        String[] s = line.split(" ");
        int x = 0;
        for (int i = 0; i < s.length; i++) { x += i; }
    }
}
"""
    time_cls, _space, certainty = _answer(source, "java")
    assert time_cls == "O(n)" and certainty != "certain"


def test_a_free_function_still_takes_its_subject_from_the_arguments() -> None:
    assert _answer("def f(xs):\n    return sorted(xs)\n")[0] == "O(n log n)"


def test_a_user_defined_line_reader_returns_a_line_of_input() -> None:
    source = """
import java.io.*;
import java.util.*;

public class Main {
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    String[] canned = { "4", "1 1 2" };
    boolean test = false;
    int at = -1;

    String readLine() throws IOException {
        at++;
        if (test) return canned[at];
        return in.readLine();
    }

    void solve() throws IOException {
        readLine();
        String[] parts = readLine().split(" ");
        List<Integer> values = new ArrayList<>();
        for (int i = 0; i < parts.length; i++) {
            values.add(Integer.valueOf(parts[i]));
        }
        Collections.sort(values);
        System.out.println(values.get(0));
    }

    public static void main(String[] args) throws IOException {
        new Main().solve();
    }
}
"""
    assert _answer(source, "java")[:2] == ("O(n log n)", "O(n)")


# ------------------------------------------------------------------ itertools
def test_every_triple_of_a_list_is_a_cubic_walk() -> None:
    source = """
from itertools import combinations


def f(xs):
    best = 0
    for a, b, c in combinations(xs, 3):
        best = max(best, a + b + c)
    return best
"""
    assert _answer(source)[:2] == ("O(n^3)", "O(1)")


def test_every_ordering_of_a_list_is_factorial() -> None:
    source = """
import itertools


def f(xs):
    best = 0
    for p in itertools.permutations(xs):
        best = max(best, p[0])
    return best
"""
    assert _answer(source)[0] == "O(2^n)"  # n!, rounded up to the largest legacy class


def test_pairs_of_orderings_with_a_fixed_length_are_polynomial() -> None:
    source = """
from itertools import permutations


def f(xs):
    t = 0
    for a, b in permutations(xs, 2):
        t += a * b
    return t
"""
    assert _answer(source)[0] == "O(n^2)"


def test_the_product_of_two_lists_visits_every_pair() -> None:
    source = """
import itertools


def f(a, b):
    t = 0
    for x, y in itertools.product(a, b):
        t += x * y
    return t
"""
    assert _answer(source)[0] == "O(n^2)"


def test_counting_pairs_in_a_generator_costs_the_pairs() -> None:
    source = """
from itertools import combinations


def f(xs):
    return sum(1 for a, b in combinations(xs, 2) if a < b)
"""
    assert _answer(source)[0] == "O(n^2)"


def test_materialising_all_pairs_stores_all_pairs() -> None:
    source = """
from itertools import combinations


def f(xs):
    return list(combinations(xs, 2))
"""
    assert _answer(source)[:2] == ("O(n^2)", "O(n^2)")


def test_a_comprehension_over_one_line_of_input_unpacked_into_names() -> None:
    source = """
n = int(input())
seen = {}
for _ in range(n):
    h, m = (int(x) for x in input().split())
    seen[h * 60 + m] = seen.get(h * 60 + m, 0) + 1
print(max(seen.values()))
"""
    assert _answer(source)[:2] == ("O(n)", "O(n)")


def test_a_list_comprehension_over_one_line_of_input_unpacked_into_names() -> None:
    source = """
n = int(input())
total = 0
for i in range(n):
    a, b = [int(x) for x in input().split(" ")]
    total += a * b
print(total)
"""
    assert _answer(source)[:2] == ("O(n)", "O(1)")
