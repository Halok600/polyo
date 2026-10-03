"""Recursion: how many times a function calls itself, on what, and what each call costs.

Each test is one recurrence shape, small enough to solve by hand; the golden suite covers the same
shapes across languages."""

from __future__ import annotations

import pytest

from analysis.engine import analyze


def answer(source: str, language: str = "python") -> tuple[str, str]:
    result = analyze(source, language)
    return result.time.text, result.space.text


# ------------------------------------------------------------------ one call, subtractive
def test_factorial_is_linear_in_time_and_stack() -> None:
    source = (
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_sum_by_index_walks_the_remaining_elements() -> None:
    source = (
        "def sum_from(nums, i):\n    if i == len(nums):\n        return 0\n"
        "    return nums[i] + sum_from(nums, i + 1)\n\n\ndef total(nums):\n    return sum_from(nums, 0)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_slice_copied_at_every_level_is_quadratic_in_time_and_space() -> None:
    source = "def f(nums):\n    if not nums:\n        return 0\n    return nums[0] + f(nums[1:])\n"
    assert answer(source) == ("O(n^2)", "O(n^2)")


def test_a_countdown_by_two_is_still_linear() -> None:
    source = "def f(n):\n    if n <= 0:\n        return 0\n    return 1 + f(n - 2)\n"
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------------------------ several calls, subtractive
def test_two_calls_that_each_take_one_off_are_exponential() -> None:
    source = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    assert answer(source) == ("O(2^n)", "O(n)")


def test_towers_of_hanoi_doubles_each_level() -> None:
    source = (
        "def hanoi(n, a, b, c):\n    if n == 0:\n        return 0\n    moves = hanoi(n - 1, a, c, b)\n"
        "    moves += 1\n    moves += hanoi(n - 1, c, b, a)\n    return moves\n"
    )
    assert answer(source) == ("O(2^n)", "O(n)")


def test_include_exclude_over_an_index_is_exponential_in_the_remaining_length() -> None:
    source = (
        "def count_from(nums, i, target):\n    if i == len(nums):\n        return 1 if target == 0 else 0\n"
        "    return count_from(nums, i + 1, target - nums[i]) + count_from(nums, i + 1, target)\n\n\n"
        "def count_subsets(nums, target):\n    return count_from(nums, 0, target)\n"
    )
    assert answer(source) == ("O(2^n)", "O(n)")


# ------------------------------------------------------------------------- a call in a loop
def test_a_loop_of_calls_on_the_loop_index_gives_two_to_the_n() -> None:
    source = (
        "def dfs(nums, start):\n    count = 1\n    for i in range(start, len(nums)):\n"
        "        count += dfs(nums, i + 1)\n    return count\n\n\ndef count_subsets(nums):\n    return dfs(nums, 0)\n"
    )
    assert answer(source) == ("O(2^n)", "O(n)")


def test_a_loop_of_calls_that_each_use_one_fewer_gives_a_factorial() -> None:
    source = (
        "def permute(nums, i):\n    if i == len(nums):\n        return 1\n    count = 0\n"
        "    for j in range(i, len(nums)):\n        nums[i], nums[j] = nums[j], nums[i]\n"
        "        count += permute(nums, i + 1)\n        nums[i], nums[j] = nums[j], nums[i]\n    return count\n\n\n"
        "def count_permutations(nums):\n    return permute(nums, 0)\n"
    )
    assert answer(source) == ("O(n!)", "O(n)")


# ------------------------------------------------------------------------------ divisive
def test_halving_the_argument_once_is_logarithmic() -> None:
    source = "def bits(n):\n    if n == 0:\n        return 0\n    return 1 + bits(n // 2)\n"
    assert answer(source) == ("O(log n)", "O(log n)")


def test_euclid_by_recursion_is_logarithmic() -> None:
    source = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n"
    assert answer(source) == ("O(log n)", "O(log n)")


def test_power_by_squaring_calls_itself_once_on_half() -> None:
    source = (
        "def power(x, n):\n    if n == 0:\n        return 1\n    half = power(x, n // 2)\n"
        "    if n % 2 == 0:\n        return half * half\n    return half * half * x\n"
    )
    assert answer(source) == ("O(log n)", "O(log n)")


def test_recursive_binary_search_is_logarithmic() -> None:
    source = (
        "def bsearch(nums, target, lo, hi):\n    if lo > hi:\n        return -1\n    mid = (lo + hi) // 2\n"
        "    if nums[mid] == target:\n        return mid\n    if nums[mid] < target:\n"
        "        return bsearch(nums, target, mid + 1, hi)\n    return bsearch(nums, target, lo, mid - 1)\n\n\n"
        "def search(nums, target):\n    return bsearch(nums, target, 0, len(nums) - 1)\n"
    )
    assert answer(source) == ("O(log n)", "O(log n)")


@pytest.mark.parametrize(
    ("calls", "combine", "expected"),
    [
        (4, "for i in range(n):\n        total += 1\n", "O(n^2)"),  # a > b^d: n^(log2 4)
        (2, "for i in range(n):\n        total += 1\n", "O(n log n)"),  # a = b^d: n log n
        (
            2,
            "for i in range(n):\n        for j in range(n):\n            total += 1\n",
            "O(n^2)",
        ),  # a < b^d
        (2, "total += 1\n", "O(n)"),  # a > b^0: n^(log2 2)
        (1, "total += 1\n", "O(log n)"),  # a = b^0: log n
    ],
)
def test_the_master_theorem_cases(calls: int, combine: str, expected: str) -> None:
    recursion = "".join("    total += work(n // 2)\n" for _ in range(calls))
    source = (
        "def work(n):\n    if n <= 1:\n        return 1\n    total = 0\n    "
        + combine
        + recursion
        + "    return total\n"
    )
    assert answer(source)[0] == expected


def test_merge_sort_is_n_log_n_with_linear_space() -> None:
    source = (
        "def merge_sort(nums):\n    if len(nums) <= 1:\n        return nums\n    mid = len(nums) // 2\n"
        "    left = merge_sort(nums[:mid])\n    right = merge_sort(nums[mid:])\n    result = []\n    i = 0\n    j = 0\n"
        "    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n"
        "            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n"
        "    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n"
    )
    assert answer(source) == ("O(n log n)", "O(n)")


def test_summing_a_range_by_halving_it_is_linear_with_logarithmic_stack() -> None:
    source = (
        "def range_sum(nums, lo, hi):\n    if lo == hi:\n        return nums[lo]\n    mid = (lo + hi) // 2\n"
        "    return range_sum(nums, lo, mid) + range_sum(nums, mid + 1, hi)\n\n\n"
        "def total(nums):\n    return range_sum(nums, 0, len(nums) - 1)\n"
    )
    assert answer(source) == ("O(n)", "O(log n)")


# ---------------------------------------------------------------------------- structural
def test_visiting_every_node_of_a_tree_once_is_linear() -> None:
    source = (
        "def count_nodes(root):\n    if root is None:\n        return 0\n"
        "    return 1 + count_nodes(root.left) + count_nodes(root.right)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_the_depth_of_a_tree_is_linear_with_a_stack_as_deep_as_the_tree() -> None:
    source = (
        "def max_depth(root):\n    if root is None:\n        return 0\n"
        "    return 1 + max(max_depth(root.left), max_depth(root.right))\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_walking_a_linked_list_by_recursion_is_linear() -> None:
    source = "def length(head):\n    if head is None:\n        return 0\n    return 1 + length(head.next)\n"
    assert answer(source) == ("O(n)", "O(n)")


# ----------------------------------------------------------------- memoised and visited-guarded
def test_memoised_fibonacci_computes_each_value_once() -> None:
    source = (
        "def fib(n, memo):\n    if n < 2:\n        return n\n    if n in memo:\n        return memo[n]\n"
        "    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n    return memo[n]\n\n\n"
        "def fibonacci(n):\n    return fib(n, {})\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_depth_first_search_with_a_visited_array_is_vertices_plus_edges() -> None:
    source = (
        "def dfs(graph, node, visited):\n    visited[node] = True\n    count = 1\n    for nxt in graph[node]:\n"
        "        if not visited[nxt]:\n            count += dfs(graph, nxt, visited)\n    return count\n\n\n"
        "def count_reachable(graph, start):\n    visited = [False] * len(graph)\n    return dfs(graph, start, visited)\n"
    )
    assert answer(source) == ("O(n + m)", "O(n)")


def test_flood_fill_that_clears_the_cells_it_visits_is_the_grid_size() -> None:
    source = (
        "def sink(grid, r, c):\n    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] != 1:\n        return\n"
        "    grid[r][c] = 0\n    sink(grid, r + 1, c)\n    sink(grid, r - 1, c)\n    sink(grid, r, c + 1)\n    sink(grid, r, c - 1)\n\n\n"
        "def num_islands(grid):\n    count = 0\n    for r in range(len(grid)):\n        for c in range(len(grid[0])):\n"
        "            if grid[r][c] == 1:\n                count += 1\n                sink(grid, r, c)\n    return count\n"
    )
    assert answer(source) == ("O(n * m)", "O(n * m)")


def test_a_grid_of_memoised_paths_is_rows_times_columns() -> None:
    source = (
        "def go(r, c, m, n, memo):\n    if r == m - 1 or c == n - 1:\n        return 1\n    if memo[r][c] != -1:\n"
        "        return memo[r][c]\n    memo[r][c] = go(r + 1, c, m, n, memo) + go(r, c + 1, m, n, memo)\n    return memo[r][c]\n\n\n"
        "def unique_paths(m, n):\n    memo = [[-1] * n for _ in range(m)]\n    return go(0, 0, m, n, memo)\n"
    )
    assert answer(source) == ("O(n * m)", "O(n * m)")


# ---------------------------------------------------------------------------------- mutual
def test_mutual_recursion_counts_the_whole_cycle() -> None:
    source = (
        "def is_even(n):\n    if n == 0:\n        return True\n    return is_odd(n - 1)\n\n\n"
        "def is_odd(n):\n    if n == 0:\n        return False\n    return is_even(n - 1)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------------------------------- not provable
def test_a_call_that_does_not_shrink_anything_is_reported_unknown() -> None:
    source = "def f(n):\n    return f(n)\n"
    assert analyze(source, "python").certainty == "unknown"


def test_recursion_nested_inside_a_function_is_solved_too() -> None:
    source = (
        "def solve(n):\n    def go(k):\n        if k == 0:\n            return 1\n        return go(k - 1) + 1\n"
        "    return go(n)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


# ---------------------------------------------------- sub-ranges in every language's own spelling
def test_cpp_iterator_range_constructor_shrinks_by_the_offset() -> None:
    source = (
        "int sumSlices(vector<int> nums) {\n    if (nums.empty()) return 0;\n"
        "    return nums[0] + sumSlices(vector<int>(nums.begin() + 1, nums.end()));\n}\n"
    )
    assert answer(source, "cpp") == ("O(n^2)", "O(n^2)")


def test_java_copy_of_range_shrinks_by_the_offset() -> None:
    source = (
        "class Solution {\n    public int sumSlices(int[] nums) {\n        if (nums.length == 0) return 0;\n"
        "        return nums[0] + sumSlices(Arrays.copyOfRange(nums, 1, nums.length));\n    }\n}\n"
    )
    assert answer(source, "java") == ("O(n^2)", "O(n^2)")


def test_javascript_slice_with_one_argument_shrinks_by_the_offset() -> None:
    source = (
        "function sumSlices(nums) {\n  if (nums.length === 0) return 0;\n"
        "  return nums[0] + sumSlices(nums.slice(1));\n}\n"
    )
    assert answer(source, "javascript") == ("O(n^2)", "O(n^2)")


def test_a_java_sublist_is_a_view_and_costs_nothing_to_take() -> None:
    source = (
        "class Solution {\n    public int sumList(List<Integer> nums) {\n        if (nums.isEmpty()) return 0;\n"
        "        return nums.get(0) + sumList(nums.subList(1, nums.size()));\n    }\n}\n"
    )
    assert answer(source, "java") == ("O(n)", "O(n)")


def test_inserting_at_the_front_of_a_cpp_vector_shifts_everything() -> None:
    source = "void f(vector<int>& v, int n) {\n    for (int i = 0; i < n; i++) {\n        v.insert(v.begin(), i);\n    }\n}\n"
    assert answer(source, "cpp")[0] == "O(n * m)"


def test_inserting_at_the_end_of_a_cpp_vector_is_constant() -> None:
    source = "void f(vector<int>& v, int n) {\n    for (int i = 0; i < n; i++) {\n        v.insert(v.end(), i);\n    }\n}\n"
    assert answer(source, "cpp")[0] == "O(n)"


def test_the_distance_between_two_iterators_is_the_number_of_elements_between_them() -> None:
    source = "int f(vector<int>& v) {\n    return distance(v.begin() + 2, v.end());\n}\n"
    result = analyze(source, "cpp")
    assert result.time.text == "O(1)"


def test_a_comparison_of_array_elements_is_not_a_visited_guard() -> None:
    # mergeSort(nums, tmp, lo, hi) reads and writes nums, but never to skip work it already did
    source = (
        "def merge_sort(nums, tmp, lo, hi):\n    if hi - lo <= 1:\n        return\n    mid = (lo + hi) // 2\n"
        "    merge_sort(nums, tmp, lo, mid)\n    merge_sort(nums, tmp, mid, hi)\n    i, j, k = lo, mid, lo\n"
        "    while i < mid and j < hi:\n        if nums[i] <= nums[j]:\n            tmp[k] = nums[i]\n            i += 1\n"
        "        else:\n            tmp[k] = nums[j]\n            j += 1\n        k += 1\n"
        "    while i < mid:\n        tmp[k] = nums[i]\n        i += 1\n        k += 1\n"
        "    while j < hi:\n        tmp[k] = nums[j]\n        j += 1\n        k += 1\n"
        "    for k in range(lo, hi):\n        nums[k] = tmp[k]\n"
    )
    assert answer(source)[0] == "O(n log n)"


def test_go_comma_ok_lookup_is_a_memo_guard() -> None:
    source = (
        "func fib(n int, memo map[int]int) int {\n    if n < 2 {\n        return n\n    }\n"
        "    if v, ok := memo[n]; ok {\n        return v\n    }\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n"
        "    return memo[n]\n}\n\nfunc fibonacci(n int) int {\n    return fib(n, make(map[int]int))\n}\n"
    )
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_a_value_read_from_the_table_and_tested_is_a_memo_guard() -> None:
    source = (
        "def fib(n, memo):\n    if n < 2:\n        return n\n    cached = memo.get(n)\n    if cached is not None:\n"
        "        return cached\n    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n    return memo[n]\n\n\n"
        "def fibonacci(n):\n    return fib(n, {})\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------- memo tables that are not parameters
def test_a_decorated_cache_makes_fibonacci_linear() -> None:
    source = (
        "from functools import lru_cache\n\n\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2:\n        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_global_memo_dictionary_makes_fibonacci_linear() -> None:
    source = (
        "memo = {}\n\n\ndef fib(n):\n    if n < 2:\n        return n\n    if n in memo:\n        return memo[n]\n"
        "    memo[n] = fib(n - 1) + fib(n - 2)\n    return memo[n]\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_memo_captured_by_a_nested_function_makes_it_linear() -> None:
    source = (
        "def solve(n):\n    memo = {}\n\n    def dp(i):\n        if i < 2:\n            return i\n"
        "        if i in memo:\n            return memo[i]\n        memo[i] = dp(i - 1) + dp(i - 2)\n"
        "        return memo[i]\n\n    return dp(n)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_memo_stored_on_self_makes_a_method_linear() -> None:
    source = (
        "class Solution:\n    def __init__(self):\n        self.memo = {}\n\n    def fib(self, n):\n"
        "        if n < 2:\n            return n\n        if n in self.memo:\n            return self.memo[n]\n"
        "        self.memo[n] = self.fib(n - 1) + self.fib(n - 2)\n        return self.memo[n]\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_marking_and_unmarking_is_backtracking_not_memoisation() -> None:
    # permutations with a `used` array: every state is explored again, so this is NOT O(n)
    source = (
        "def permute(nums, path, used):\n    if len(path) == len(nums):\n        return 1\n    count = 0\n"
        "    for i in range(len(nums)):\n        if used[i]:\n            continue\n        used[i] = True\n"
        "        path.append(nums[i])\n        count += permute(nums, path, used)\n        path.pop()\n"
        "        used[i] = False\n    return count\n"
    )
    result = analyze(source, "python")
    assert result.time.cls != "O(n)" or result.certainty == "unknown"


def test_quick_sort_with_unbalanced_splits_is_never_claimed_exponential() -> None:
    # two calls on parts that together are smaller than the whole: the engine cannot split the
    # sizes, so it must say it does not know rather than answer 2^n
    source = (
        "def qsort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[0]\n"
        "    left = [x for x in a[1:] if x < pivot]\n    right = [x for x in a[1:] if x >= pivot]\n"
        "    return qsort(left) + [pivot] + qsort(right)\n"
    )
    result = analyze(source, "python")
    assert result.time.cls != "O(2^n)" or result.certainty == "unknown"


# ------------------------------------------------------------ found by held-out programs
def test_a_branch_that_grows_the_argument_makes_the_whole_recursion_unprovable() -> None:
    # collatz: one call halves n, the other makes it 3n + 1; the halving path must not be taken
    # as the answer
    source = (
        "def collatz(n):\n    if n == 1:\n        return 0\n    if n % 2 == 0:\n        return 1 + collatz(n // 2)\n"
        "    return 1 + collatz(3 * n + 1)\n"
    )
    assert analyze(source, "python").certainty == "unknown"


def test_a_slice_with_a_negative_end_drops_that_many_elements() -> None:
    source = (
        "def is_pal(s):\n    if len(s) <= 1:\n        return True\n    if s[0] != s[-1]:\n        return False\n"
        "    return is_pal(s[1:-1])\n"
    )
    assert answer(source) == ("O(n^2)", "O(n^2)")


def test_recursing_over_the_children_of_every_node_is_linear_in_the_nodes() -> None:
    source = (
        "def depth(node):\n    if node is None:\n        return 0\n    best = 0\n    for child in node.children:\n"
        "        best = max(best, depth(child))\n    return best + 1\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_nested_depth_first_search_with_a_visited_set_is_vertices_plus_edges() -> None:
    source = (
        "def count_components(graph):\n    seen = set()\n\n    def dfs(u):\n        seen.add(u)\n        for v in graph[u]:\n"
        "            if v not in seen:\n                dfs(v)\n\n    count = 0\n    for u in range(len(graph)):\n"
        "        if u not in seen:\n            count += 1\n            dfs(u)\n    return count\n"
    )
    assert answer(source) == ("O(n + m)", "O(n)")


def test_an_unused_helper_does_not_hide_a_mutually_recursive_pair() -> None:
    source = (
        "def helper(x):\n    return x + 1\n\n\ndef is_even(n):\n    if n == 0:\n        return True\n"
        "    return is_odd(n - 1)\n\n\ndef is_odd(n):\n    if n == 0:\n        return False\n    return is_even(n - 1)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")
