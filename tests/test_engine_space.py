"""Space conventions and the space-specific mistakes a time-first engine makes.

Space is the memory the function allocates, including what it returns (the input itself is not
counted), plus the recursion stack. These pin the cases where that is easy to get wrong."""

from __future__ import annotations

from analysis.engine import analyze


def answer(source: str, language: str = "python") -> tuple[str, str]:
    result = analyze(source, language)
    return result.time.text, result.space.text


# ------------------------------------------------------------ sets and dicts of loop indices
def test_a_set_of_row_indices_holds_at_most_one_entry_per_row() -> None:
    source = (
        "def f(matrix):\n    rows = set()\n    for i in range(len(matrix)):\n"
        "        for j in range(len(matrix[0])):\n            if matrix[i][j] == 0:\n                rows.add(i)\n"
        "    return rows\n"
    )
    assert answer(source) == ("O(n * m)", "O(n)")


def test_sets_of_row_and_column_indices_are_rows_plus_columns() -> None:
    source = (
        "def f(matrix):\n    rows, cols = set(), set()\n    for i in range(len(matrix)):\n"
        "        for j in range(len(matrix[0])):\n            if matrix[i][j] == 0:\n                rows.add(i)\n"
        "                cols.add(j)\n    return len(rows) + len(cols)\n"
    )
    assert answer(source) == ("O(n * m)", "O(n + m)")


def test_a_dictionary_keyed_by_the_loop_index_holds_at_most_one_entry_per_index() -> None:
    source = (
        "def f(nums):\n    last = {}\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n"
        "            last[i] = j\n    return last\n"
    )
    assert answer(source) == ("O(n^2)", "O(n)")


def test_a_set_of_pairs_of_indices_is_bounded_by_their_ranges() -> None:
    source = (
        "def f(grid):\n    seen = set()\n    for r in range(len(grid)):\n        for c in range(len(grid[0])):\n"
        "            seen.add((r, c))\n    return seen\n"
    )
    assert answer(source) == ("O(n * m)", "O(n * m)")


def test_a_set_of_data_values_can_hold_one_entry_per_element() -> None:
    source = (
        "def f(nums):\n    seen = set()\n    for x in nums:\n        for y in nums:\n            seen.add(x + y)\n"
        "    return seen\n"
    )
    assert answer(source)[1] == "O(n^2)"


def test_a_set_that_is_added_to_in_a_nested_loop_with_a_data_key_is_not_bounded_by_the_loop() -> (
    None
):
    source = "def f(nums):\n    seen = set()\n    for i in range(len(nums)):\n        seen.add(nums[i] * 2)\n    return seen\n"
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------------------------- element values are not known
def test_a_counter_array_is_not_assumed_to_stay_zero() -> None:
    # counts[v] starts at 0 but is incremented; the extend below really produces n elements
    source = (
        "def counting_sort(nums):\n    counts = [0] * 101\n    for x in nums:\n        counts[x] += 1\n    out = []\n"
        "    for v in range(101):\n        out.extend([v] * counts[v])\n    return out\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_filled_table_does_not_fix_the_size_of_a_list_built_from_it() -> None:
    source = (
        "def f(nums):\n    sizes = [0] * len(nums)\n    for i in range(len(nums)):\n        sizes[i] = nums[i]\n"
        "    out = []\n    for i in range(len(nums)):\n        out.extend([i] * sizes[i])\n    return out\n"
    )
    result = analyze(source, "python")
    assert result.space.cls != "O(1)"


# ----------------------------------------------------------------------------- number to text
def test_the_text_of_a_number_has_as_many_characters_as_it_has_digits() -> None:
    source = "def digit_sum(n):\n    return sum(int(c) for c in str(n))\n"
    assert answer(source) == ("O(log n)", "O(log n)")


def test_the_text_of_a_loop_index_is_not_charged_a_logarithm() -> None:
    # a machine-word integer converts in constant time; only a PARAMETER treated as a size counts
    source = "def f(n):\n    parts = []\n    for i in range(n):\n        parts.append(str(i))\n    return ','.join(parts)\n"
    assert answer(source) == ("O(n)", "O(n)")


# ------------------------------------------------------------------- C++ by-value parameters
def test_a_cpp_container_taken_by_value_is_copied_on_every_call() -> None:
    source = "int f(vector<int> nums) {\n    return nums.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_a_cpp_container_taken_by_reference_is_not_copied() -> None:
    source = "int f(vector<int>& nums) {\n    return nums.size();\n}\n"
    assert answer(source, "cpp") == ("O(1)", "O(1)")


def test_a_cpp_const_reference_is_not_copied() -> None:
    source = "int f(const vector<int>& nums) {\n    return nums.size();\n}\n"
    assert answer(source, "cpp") == ("O(1)", "O(1)")


def test_a_cpp_string_taken_by_value_is_copied() -> None:
    source = (
        "bool isPal(string s) {\n    int i = 0, j = s.size() - 1;\n    while (i < j) {\n"
        "        if (s[i] != s[j]) return false;\n        i++;\n        j--;\n    }\n    return true;\n}\n"
    )
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_a_recursive_cpp_function_that_takes_a_vector_by_value_copies_it_in_every_frame() -> None:
    source = "int g(vector<int> v, int i) {\n    if (i == v.size()) return 0;\n    return v[i] + g(v, i + 1);\n}\n"
    assert answer(source, "cpp") == ("O(n^2)", "O(n^2)")


def test_a_cpp_pointer_parameter_is_not_a_copy_of_what_it_points_to() -> None:
    source = "int f(int* nums, int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) s += nums[i];\n    return s;\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(1)")


def test_java_and_python_collections_are_references_and_never_copied() -> None:
    assert answer("def f(nums):\n    return len(nums)\n") == ("O(1)", "O(1)")
    assert answer("class S { int f(List<Integer> nums) { return nums.size(); } }", "java") == (
        "O(1)",
        "O(1)",
    )


# ----------------------------------------------------------- found by the second held-out batch
def test_comparing_a_string_with_a_literal_costs_the_literal() -> None:
    source = (
        "int count(vector<string>& tokens) {\n    int c = 0;\n    for (auto& t : tokens) {\n"
        '        if (t == "+" || t == "-") c++;\n    }\n    return c;\n}\n'
    )
    assert answer(source, "cpp") == ("O(n)", "O(1)")


def test_a_dictionary_of_lists_grows_by_every_append_into_its_lists() -> None:
    source = (
        "from collections import defaultdict\n\n\ndef group(words):\n    groups = defaultdict(list)\n"
        "    for w in words:\n        groups[w[0]].append(w)\n    return groups\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_setdefault_with_a_list_makes_a_dictionary_of_lists() -> None:
    source = (
        "def group(words):\n    groups = {}\n    for w in words:\n        groups.setdefault(w[0], []).append(w)\n"
        "    return groups\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_a_dictionary_that_stores_a_fresh_list_per_key_then_appends_to_it_is_a_dictionary_of_lists() -> (
    None
):
    source = (
        "def group(words):\n    groups = {}\n    for w in words:\n        if w[0] not in groups:\n"
        "            groups[w[0]] = []\n        groups[w[0]].append(w)\n    return groups\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


LEVELS_PY = (
    "from collections import deque\n\n\ndef levels(root):\n    out = []\n    queue = deque([root]) if root else deque()\n"
    "    while queue:\n        level = []\n        for _ in range(len(queue)):\n            node = queue.popleft()\n"
    "            level.append(node.val)\n            if node.left:\n                queue.append(node.left)\n"
    "            if node.right:\n                queue.append(node.right)\n        out.append(level)\n    return out\n"
)


def test_level_order_traversal_of_a_tree_visits_every_node_once() -> None:
    assert answer(LEVELS_PY) == ("O(n)", "O(n)")


def test_level_order_traversal_in_java_with_a_size_snapshot() -> None:
    source = (
        "class Solution {\n    public List<List<Integer>> levels(TreeNode root) {\n"
        "        List<List<Integer>> out = new ArrayList<>();\n        if (root == null) return out;\n"
        "        Queue<TreeNode> q = new ArrayDeque<>();\n        q.add(root);\n        while (!q.isEmpty()) {\n"
        "            int size = q.size();\n            List<Integer> level = new ArrayList<>();\n"
        "            for (int i = 0; i < size; i++) {\n                TreeNode node = q.poll();\n"
        "                level.add(node.val);\n                if (node.left != null) q.add(node.left);\n"
        "                if (node.right != null) q.add(node.right);\n            }\n            out.add(level);\n        }\n"
        "        return out;\n    }\n}\n"
    )
    assert answer(source, "java") == ("O(n)", "O(n)")


def test_a_depth_first_search_with_an_explicit_stack_over_a_grid_visits_each_cell_once() -> None:
    source = (
        "def count_land(grid):\n    n, m = len(grid), len(grid[0])\n    visited = [[False] * m for _ in range(n)]\n"
        "    count = 0\n    for i in range(n):\n        for j in range(m):\n"
        "            if grid[i][j] == 1 and not visited[i][j]:\n                count += 1\n"
        "                stack = [(i, j)]\n                visited[i][j] = True\n                while stack:\n"
        "                    r, c = stack.pop()\n                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n"
        "                        nr, nc = r + dr, c + dc\n"
        "                        if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == 1 and not visited[nr][nc]:\n"
        "                            visited[nr][nc] = True\n                            stack.append((nr, nc))\n"
        "    return count\n"
    )
    assert answer(source) == ("O(n * m)", "O(n * m)")


def test_a_worklist_refilled_each_round_with_a_visited_marker_that_is_reset_is_not_amortised() -> (
    None
):
    # the visited array is rebuilt for every start, so the walks do not share their work
    source = (
        "def f(grid):\n    n = len(grid)\n    total = 0\n    for s in range(n):\n        visited = [False] * n\n"
        "        stack = [s]\n        visited[s] = True\n        while stack:\n            u = stack.pop()\n"
        "            total += 1\n            for v in grid[u]:\n                if not visited[v]:\n"
        "                    visited[v] = True\n                    stack.append(v)\n    return total\n"
    )
    assert answer(source)[0] == "O(n^2 + n * m)"  # n walks, each paying for its own visited array
