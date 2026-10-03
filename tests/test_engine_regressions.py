"""Regressions found by running the engine over the golden suite.

Each test is the smallest snippet that reproduced one real bug, in the language it showed up in."""

from __future__ import annotations

from analysis.engine import analyze


def answer(source: str, language: str = "python") -> tuple[str, str]:
    result = analyze(source, language)
    return result.time.text, result.space.text


# ------------------------------------------------------------------------------ C++ specifics
def test_cpp_nested_vector_constructor_allocates_rows_by_columns() -> None:
    source = "void f(int n, int m) {\n vector<vector<int>> t(n, vector<int>(m, 0));\n}\n"
    assert answer(source, "cpp") == ("O(n * m)", "O(n * m)")


def test_cpp_comparing_iterators_is_free() -> None:
    source = (
        "int f(vector<int>& nums) {\n unordered_map<int,int> seen;\n"
        " for (int i = 0; i < nums.size(); i++) {\n"
        "  auto it = seen.find(nums[i]);\n  if (it != seen.end()) return 1;\n  seen[nums[i]] = i;\n }\n return 0;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n)"


def test_pushing_a_named_string_into_a_vector_adds_one_element() -> None:
    source = (
        "int f(string& s) {\n vector<string> words;\n string cur;\n"
        " for (char ch : s) {\n  if (ch == ' ') { words.push_back(cur); cur.clear(); } else { cur += ch; }\n }\n"
        " int c = 0;\n for (string& w : words) { c++; }\n return c;\n}\n"
    )
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_iterator_offset_constructor_copies_the_tail() -> None:
    source = (
        "int f(vector<int>& nums) {\n int t = 0;\n for (int i = 0; i < nums.size(); i++) {\n"
        "  vector<int> tail(nums.begin() + i, nums.end());\n  t += tail.size();\n }\n return t;\n}\n"
    )
    assert answer(source, "cpp") == ("O(n^2)", "O(n)")


def test_cpp_copy_initialisation_of_a_vector_is_a_copy() -> None:
    source = "int f(vector<int>& nums) {\n vector<int> items = nums;\n return items.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_reference_initialisation_is_not_a_copy() -> None:
    source = "int f(vector<int>& nums) {\n vector<int>& items = nums;\n return items.size();\n}\n"
    assert answer(source, "cpp") == ("O(1)", "O(1)")


# ---------------------------------------------------------------------------- loop bounds
def test_a_bound_read_from_an_element_counts_down_to_zero_at_worst() -> None:
    source = (
        "int f(vector<int>& weights, int capacity) {\n int s = 0;\n"
        " for (int i = 0; i < weights.size(); i++) {\n"
        "  for (int w = capacity; w >= weights[i]; w--) { s += w; }\n }\n return s;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n * m)"


def test_euclid_is_logarithmic_in_the_smaller_operand() -> None:
    source = "def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a\n"
    assert answer(source)[0] == "O(log n)"


def test_shifting_until_zero_with_not_equal_is_logarithmic() -> None:
    source = (
        "int f(int n) {\n int c = 0;\n while (n != 0) { c += n & 1; n >>= 1; }\n return c;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(log n)"


def test_a_pop_loop_written_with_length_greater_than_zero_is_a_drain() -> None:
    source = (
        "function f(nums) {\n const result = new Array(nums.length).fill(-1);\n const stack = [];\n"
        " for (let i = 0; i < nums.length; i++) {\n"
        "  while (stack.length > 0 && nums[stack[stack.length - 1]] < nums[i]) {\n"
        "   result[stack.pop()] = nums[i];\n  }\n  stack.push(i);\n }\n return result;\n}\n"
    )
    assert answer(source, "javascript") == ("O(n)", "O(n)")


def test_a_pointer_used_through_charat_cannot_pass_the_end_of_the_string() -> None:
    source = (
        "class S {\n int f(String s) {\n  Set<Character> window = new HashSet<>();\n  int left = 0;\n  int best = 0;\n"
        "  for (int right = 0; right < s.length(); right++) {\n"
        "   while (window.contains(s.charAt(right))) { window.remove(s.charAt(left)); left++; }\n"
        "   window.add(s.charAt(right));\n   best = Math.max(best, right - left + 1);\n  }\n  return best;\n }\n}\n"
    )
    assert answer(source, "java") == ("O(n)", "O(n)")


def test_a_reset_pointer_inside_a_for_c_loop_blocks_amortisation() -> None:
    # the inner loop re-initialises w each outer iteration, so its cost is NOT shared
    source = (
        "int f(vector<int>& a, int cap) {\n int s = 0;\n for (int i = 0; i < a.size(); i++) {\n"
        "  for (int w = cap; w > 0; w--) { s += a[w]; }\n }\n return s;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n * m)"


# --------------------------------------------------------------------------- strings, copies
def test_string_concatenation_in_a_loop_is_quadratic_with_linear_space() -> None:
    source = 'def f(chars):\n    result = ""\n    for ch in chars:\n        result += ch\n    return result\n'
    assert answer(source) == ("O(n^2)", "O(n)")


def test_cpp_string_append_is_amortised_constant() -> None:
    source = "string f(vector<char>& chars) {\n string result;\n for (char ch : chars) { result += ch; }\n return result;\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_spreading_a_collection_copies_it() -> None:
    source = "function f(nums) { const c = [...nums].sort((a, b) => a - b); return c; }\n"
    assert answer(source, "javascript") == ("O(n log n)", "O(n)")


def test_a_primitive_java_array_sort_needs_only_a_logarithmic_stack() -> None:
    source = "class S { void f(int[] nums) { Arrays.sort(nums); } }\n"
    assert answer(source, "java") == ("O(n log n)", "O(log n)")


def test_c_allocation_functions_allocate_their_size() -> None:
    source = "int* f(int n) {\n int* dp = calloc(n + 1, sizeof(int));\n for (int i = 0; i < n; i++) dp[i] = i;\n return dp;\n}\n"
    assert answer(source, "c") == ("O(n)", "O(n)")


def test_reading_a_number_does_not_retain_the_line_it_was_read_from() -> None:
    source = (
        "n = int(input())\ntotal = 0\nfor _ in range(n):\n    total += int(input())\nprint(total)\n"
    )
    assert answer(source) == ("O(n)", "O(1)")


# ----------------------------------------------------------------------------- pairs and rows
def test_rows_only_indexed_by_constants_are_constant_size() -> None:
    source = (
        "class S {\n List<int[]> f(int[][] intervals) {\n  int[][] ordered = intervals.clone();\n"
        "  Arrays.sort(ordered, (a, b) -> a[0] - b[0]);\n  List<int[]> merged = new ArrayList<>();\n"
        "  for (int[] iv : ordered) { merged.add(iv); }\n  return merged;\n }\n}\n"
    )
    assert answer(source, "java")[1] == "O(n)"


def test_rows_of_a_local_adjacency_list_total_the_pushes_into_them() -> None:
    source = (
        "def f(n, edges):\n    graph = [[] for _ in range(n)]\n    for u, v in edges:\n"
        "        graph[u].append(v)\n    total = 0\n    for u in range(n):\n        for v in graph[u]:\n"
        "            total += 1\n    return total\n"
    )
    assert answer(source)[0] == "O(n + m)"


def test_a_scan_with_a_running_max_is_linear() -> None:
    source = (
        "def f(nums):\n    best = nums[0]\n    cur = nums[0]\n    for i in range(1, len(nums)):\n"
        "        cur = max(nums[i], cur + nums[i])\n        best = max(best, cur)\n    return best\n"
    )
    assert answer(source) == ("O(n)", "O(1)")


# ------------------------------------------------------------------- counted for-loops, C family
def test_a_triangular_loop_written_with_c_style_for_is_quadratic() -> None:
    source = (
        "int f(vector<int>& nums) {\n int count = 0;\n for (int i = 0; i < nums.size(); i++) {\n"
        "  for (int j = 0; j < i; j++) { if (nums[j] < nums[i]) count++; }\n }\n return count;\n}\n"
    )
    assert answer(source, "cpp") == ("O(n^2)", "O(1)")


def test_a_triangular_loop_in_go_is_quadratic() -> None:
    source = (
        "func f(nums []int) int {\n count := 0\n for i := 0; i < len(nums); i++ {\n"
        "  for j := 0; j < i; j++ { if nums[j] < nums[i] { count++ } }\n }\n return count\n}\n"
    )
    assert answer(source, "go") == ("O(n^2)", "O(1)")


# ----------------------------------------------------------------------------- Euclid's algorithm
def test_euclid_with_a_temporary_in_cpp_is_logarithmic() -> None:
    source = "int gcd(int a, int b) {\n while (b != 0) {\n  int t = b;\n  b = a % b;\n  a = t;\n }\n return a;\n}\n"
    assert answer(source, "cpp") == ("O(log n)", "O(1)")


def test_euclid_with_a_truthiness_test_is_logarithmic() -> None:
    source = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    assert answer(source) == ("O(log n)", "O(1)")


def test_a_remainder_loop_that_does_not_swap_is_not_assumed_logarithmic() -> None:
    # b = a % b with `a` fixed does not provably shrink geometrically
    source = "def f(a, b):\n    while b:\n        b = a % b\n    return b\n"
    assert answer(source)[0] != "O(log n)"


# ------------------------------------------------------------------------------------- graphs
BFS_PYTHON = (
    "from collections import deque\n\n\ndef bfs_order(graph, start):\n    visited = {start}\n"
    "    queue = deque([start])\n    order = []\n    while queue:\n        node = queue.popleft()\n"
    "        order.append(node)\n        for nxt in graph[node]:\n            if nxt not in visited:\n"
    "                visited.add(nxt)\n                queue.append(nxt)\n    return order\n"
)


def test_bfs_over_an_adjacency_list_parameter_is_vertices_plus_edges() -> None:
    assert answer(BFS_PYTHON) == ("O(n + m)", "O(n)")


def test_a_worklist_tested_with_len_greater_than_zero_is_still_a_drain() -> None:
    source = BFS_PYTHON.replace("while queue:", "while len(queue) > 0:")
    assert answer(source) == ("O(n + m)", "O(n)")


def test_a_worklist_tested_with_len_not_equal_zero_is_still_a_drain() -> None:
    source = BFS_PYTHON.replace("while queue:", "while len(queue) != 0:")
    assert answer(source) == ("O(n + m)", "O(n)")


def test_java_list_get_of_a_row_is_summable_like_a_subscript() -> None:
    source = (
        "class S {\n int f(List<List<Integer>> graph, int start) {\n  boolean[] seen = new boolean[graph.size()];\n"
        "  Queue<Integer> q = new ArrayDeque<>();\n  q.add(start);\n  seen[start] = true;\n  int count = 0;\n"
        "  while (!q.isEmpty()) {\n   int u = q.poll();\n   count++;\n"
        "   for (int v : graph.get(u)) {\n    if (!seen[v]) { seen[v] = true; q.add(v); }\n   }\n  }\n  return count;\n }\n}\n"
    )
    assert answer(source, "java")[0] == "O(n + m)"


def test_go_append_returns_the_grown_slice_so_it_stays_trackable_after_a_loop() -> None:
    source = (
        "func f(n int, indeg []int) int {\n queue := []int{}\n for i := 0; i < n; i++ {\n"
        "  if indeg[i] == 0 {\n   queue = append(queue, i)\n  }\n }\n total := 0\n"
        " for _, x := range queue {\n  total += x\n }\n return total\n}\n"
    )
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_go_append_in_a_loop_leaves_a_container_not_an_unknown_value() -> None:
    # once the slice is lost, the drain loop below is "assumed linear" with a note
    source = (
        "func f(n int, indeg []int) int {\n queue := []int{}\n for i := 0; i < n; i++ {\n"
        "  if indeg[i] == 0 {\n   queue = append(queue, i)\n  }\n }\n head := 0\n"
        " for head < len(queue) {\n  head++\n }\n return head\n}\n"
    )
    result = analyze(source, "go")
    assert result.certainty == "certain"


KAHN_CPP = (
    "vector<int> topoOrder(int n, vector<vector<int>>& edges) {\n vector<vector<int>> graph(n);\n"
    " vector<int> indeg(n, 0);\n for (auto& e : edges) {\n  graph[e[0]].push_back(e[1]);\n  indeg[e[1]]++;\n }\n"
    " vector<int> order;\n queue<int> q;\n for (int i = 0; i < n; i++) {\n  if (indeg[i] == 0) q.push(i);\n }\n"
    " while (!q.empty()) {\n  int u = q.front();\n  q.pop();\n  order.push_back(u);\n"
    "  for (int v : graph[u]) {\n   indeg[v]--;\n   if (indeg[v] == 0) q.push(v);\n  }\n }\n return order;\n}\n"
)
KAHN_JAVA = (
    "class Solution {\n public List<Integer> topoOrder(int n, int[][] edges) {\n"
    "  List<List<Integer>> graph = new ArrayList<>();\n  for (int i = 0; i < n; i++) graph.add(new ArrayList<>());\n"
    "  int[] indeg = new int[n];\n  for (int[] e : edges) {\n   graph.get(e[0]).add(e[1]);\n   indeg[e[1]]++;\n  }\n"
    "  Queue<Integer> queue = new ArrayDeque<>();\n  for (int i = 0; i < n; i++) {\n   if (indeg[i] == 0) queue.add(i);\n  }\n"
    "  List<Integer> order = new ArrayList<>();\n  while (!queue.isEmpty()) {\n   int u = queue.poll();\n   order.add(u);\n"
    "   for (int v : graph.get(u)) {\n    indeg[v]--;\n    if (indeg[v] == 0) queue.add(v);\n   }\n  }\n  return order;\n }\n}\n"
)
KAHN_GO = (
    "func topoOrder(n int, edges [][]int) []int {\n graph := make([][]int, n)\n indeg := make([]int, n)\n"
    " for _, e := range edges {\n  graph[e[0]] = append(graph[e[0]], e[1])\n  indeg[e[1]]++\n }\n"
    " queue := []int{}\n for i := 0; i < n; i++ {\n  if indeg[i] == 0 {\n   queue = append(queue, i)\n  }\n }\n"
    " order := []int{}\n head := 0\n for head < len(queue) {\n  u := queue[head]\n  head++\n  order = append(order, u)\n"
    "  for _, v := range graph[u] {\n   indeg[v]--\n   if indeg[v] == 0 {\n    queue = append(queue, v)\n   }\n  }\n }\n return order\n}\n"
)


def test_kahn_topological_sort_in_cpp_is_linear_in_vertices_and_edges() -> None:
    result = analyze(KAHN_CPP, "cpp")
    assert (result.time.cls, result.space.cls, result.certainty) == ("O(n)", "O(n)", "certain")


def test_kahn_topological_sort_in_java_is_linear_in_vertices_and_edges() -> None:
    result = analyze(KAHN_JAVA, "java")
    assert (result.time.cls, result.space.cls, result.certainty) == ("O(n)", "O(n)", "certain")


def test_kahn_topological_sort_in_go_is_linear_in_vertices_and_edges() -> None:
    result = analyze(KAHN_GO, "go")
    assert (result.time.cls, result.space.cls, result.certainty) == ("O(n)", "O(n)", "certain")


def test_go_rows_assigned_after_make_take_the_assigned_length() -> None:
    source = (
        "func f(n int, m int) int {\n grid := make([][]int, n)\n for i := range grid {\n  grid[i] = make([]int, m)\n }\n"
        " t := 0\n for i := 0; i < n; i++ {\n  for j := 0; j < m; j++ {\n   t += grid[i][j]\n  }\n }\n return t\n}\n"
    )
    assert answer(source, "go") == ("O(n * m)", "O(n * m)")


def test_java_rows_added_one_by_one_are_found_again_by_get() -> None:
    source = (
        "class S {\n int f(int n) {\n  List<List<Integer>> g = new ArrayList<>();\n"
        "  for (int i = 0; i < n; i++) g.add(new ArrayList<>());\n  int c = 0;\n"
        "  for (int i = 0; i < n; i++) {\n   for (int x : g.get(i)) { c += x; }\n  }\n  return c;\n }\n}\n"
    )
    result = analyze(source, "java")
    assert result.certainty == "certain"


# ------------------------------------------------------------- C++ "vexing parse" declarations
def test_cpp_local_vector_sized_by_a_single_identifier_is_allocated() -> None:
    # tree-sitter reads `vector<int> a(n);` as a function declarator; locally it is a constructor
    source = "int f(int n) {\n vector<int> a(n);\n return a.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_local_vector_copy_constructed_from_a_name_is_a_copy() -> None:
    source = "int f(vector<int>& nums) {\n vector<int> b(nums);\n return b.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_top_level_prototype_is_not_mistaken_for_a_variable() -> None:
    source = "int helper(Node);\nint f(int n) {\n return n;\n}\n"
    assert answer(source, "cpp") == ("O(1)", "O(1)")


def test_cpp_string_filled_by_count_is_a_string_of_that_length() -> None:
    source = "int f(int n) {\n string s(n, 'x');\n return s.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_string_copy_constructed_from_a_name_is_a_copy() -> None:
    source = "int f(string& s) {\n string t(s);\n return t.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_a_pointer_initialised_by_the_inner_for_is_reset_on_every_entry() -> None:
    # w restarts at `capacity` for each item, so its walk is not shared between items
    source = (
        "int f(vector<int>& weights, vector<int>& dp, int capacity) {\n int s = 0;\n"
        " for (int i = 0; i < weights.size(); i++) {\n"
        "  for (int w = capacity; w >= 0; w--) { s += dp[w]; }\n }\n return s;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n * m)"


def test_a_pointer_initialised_by_the_inner_for_over_an_indexed_container_is_reset() -> None:
    source = (
        "int f(vector<int>& weights, vector<int>& dp) {\n int s = 0;\n"
        " for (int i = 0; i < weights.size(); i++) {\n"
        "  for (int w = dp.size() - 1; w >= weights[i]; w--) { s += dp[w]; }\n }\n return s;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n * m)"


def test_a_countdown_range_to_a_data_dependent_floor_is_bounded_by_its_start() -> None:
    # knapsack: range(capacity, weights[i] - 1, -1) runs at most `capacity` times
    source = (
        "def f(weights, capacity):\n    dp = [0] * (capacity + 1)\n    for i in range(len(weights)):\n"
        "        for w in range(capacity, weights[i] - 1, -1):\n            dp[w] = dp[w] + 1\n    return dp[capacity]\n"
    )
    assert answer(source) == ("O(n * m)", "O(m)")


def test_an_ascending_range_from_a_data_dependent_start_is_bounded_by_its_end() -> None:
    source = (
        "def f(nums):\n    t = 0\n    for i in range(len(nums)):\n        for j in range(nums[i], len(nums)):\n"
        "            t += 1\n    return t\n"
    )
    assert answer(source)[0] == "O(n^2)"


def test_an_ascending_range_to_a_data_dependent_end_is_assumed_linear() -> None:
    source = "def f(nums):\n    t = 0\n    for j in range(nums[0]):\n        t += 1\n    return t\n"
    result = analyze(source, "python")
    assert result.certainty == "assumed"


# ------------------------------------------------------------------------------ symbolic strides
def test_a_stride_equal_to_the_outer_index_gives_a_harmonic_sum_in_cpp() -> None:
    source = (
        "vector<int> f(int n) {\n vector<int> counts(n + 1, 0);\n for (int i = 1; i <= n; i++) {\n"
        "  for (int j = i; j <= n; j += i) { counts[j]++; }\n }\n return counts;\n}\n"
    )
    assert answer(source, "cpp") == ("O(n log n)", "O(n)")


def test_a_stride_equal_to_the_outer_index_gives_a_harmonic_sum_in_go() -> None:
    source = (
        "func f(n int) []int {\n counts := make([]int, n+1)\n for i := 1; i <= n; i++ {\n"
        "  for j := i; j <= n; j += i {\n   counts[j]++\n  }\n }\n return counts\n}\n"
    )
    assert answer(source, "go") == ("O(n log n)", "O(n)")


def test_a_while_loop_with_a_variable_stride_is_a_harmonic_sum() -> None:
    source = (
        "def f(n):\n    c = 0\n    for i in range(1, n + 1):\n        j = i\n        while j <= n:\n"
        "            c += 1\n            j += i\n    return c\n"
    )
    assert answer(source)[0] == "O(n log n)"


def test_a_sieve_marking_multiples_from_a_square_is_n_log_log_n_bounded_by_n_log_n() -> None:
    source = (
        "def f(n):\n    mark = [False] * (n + 1)\n    for i in range(2, n + 1):\n"
        "        if not mark[i]:\n            for j in range(i * i, n + 1, i):\n                mark[j] = True\n"
        "    return mark\n"
    )
    assert answer(source)[0] in ("O(n log n)", "O(n)")


def test_c_malloc_of_a_sized_array_allocates_it() -> None:
    source = (
        "int* f(int n) {\n int* a = (int*)malloc(n * sizeof(int));\n"
        " for (int i = 0; i < n; i++) a[i] = i;\n return a;\n}\n"
    )
    assert answer(source, "c") == ("O(n)", "O(n)")


def test_c_realloc_grows_to_the_requested_size() -> None:
    source = "int* f(int* a, int n) {\n int* b = realloc(a, n * sizeof(int));\n return b;\n}\n"
    assert answer(source, "c") == ("O(n)", "O(n)")


# ------------------------------------------------------------------ dense grids vs ragged lists
def test_iterating_the_rows_of_a_grid_parameter_costs_rows_times_columns() -> None:
    source = "def deep_copy(grid):\n    return [row[:] for row in grid]\n"
    assert answer(source) == ("O(n * m)", "O(n * m)")


def test_iterating_the_rows_of_an_adjacency_list_parameter_costs_vertices_plus_edges() -> None:
    source = (
        "def count(graph):\n    t = 0\n    for nbrs in graph:\n        for v in nbrs:\n"
        "            t += 1\n    return t\n"
    )
    assert answer(source)[0] == "O(n + m)"


def test_java_row_clone_loop_over_a_grid_is_rows_times_columns() -> None:
    source = (
        "class S {\n int[][] f(int[][] grid) {\n  int[][] copy = new int[grid.length][];\n"
        "  for (int i = 0; i < grid.length; i++) {\n   copy[i] = grid[i].clone();\n  }\n  return copy;\n }\n}\n"
    )
    assert answer(source, "java") == ("O(n * m)", "O(n * m)")


def test_a_grid_walked_by_two_indices_is_rows_times_columns() -> None:
    source = (
        "def f(grid):\n    t = 0\n    for i in range(len(grid)):\n        for j in range(len(grid[0])):\n"
        "            t += grid[i][j]\n    return t\n"
    )
    assert answer(source) == ("O(n * m)", "O(1)")


def test_a_dense_table_built_with_fixed_length_rows_is_rows_times_columns() -> None:
    source = (
        "def f(n, m):\n    t = [[0] * m for _ in range(n)]\n    s = 0\n    for row in t:\n"
        "        for x in row:\n            s += x\n    return s\n"
    )
    assert answer(source) == ("O(n * m)", "O(n * m)")


# ------------------------------------------------------------ shallow copies and stored references
def test_a_shallow_copy_of_a_list_of_rows_copies_only_the_row_references() -> None:
    source = "def f(grid):\n    c = grid.copy()\n    return len(c)\n"
    assert answer(source) == ("O(n)", "O(n)")


def test_a_deep_copy_copies_every_element() -> None:
    source = "import copy\n\n\ndef f(grid):\n    c = copy.deepcopy(grid)\n    return len(c) + len(grid[0])\n"
    assert answer(source) == ("O(n * m)", "O(n * m)")


def test_a_full_slice_of_a_list_of_rows_shares_the_rows() -> None:
    source = "def f(grid):\n    c = grid[:]\n    return len(c)\n"
    assert answer(source) == ("O(n)", "O(n)")


def test_appending_an_existing_row_stores_a_reference_not_a_copy() -> None:
    source = "def f(grid):\n    out = []\n    for row in grid:\n        out.append(row)\n    return out\n"
    assert answer(source) == ("O(n)", "O(n)")


def test_appending_a_freshly_built_row_retains_that_row() -> None:
    source = "def f(n, m):\n    out = []\n    for i in range(n):\n        out.append([0] * m)\n    return out\n"
    assert answer(source) == ("O(n * m)", "O(n * m)")


# ---------------------------------------------------------------------- Go slices and conversions
def test_go_copy_by_appending_to_a_nil_slice_allocates_the_copy() -> None:
    source = (
        "func f(nums []int) int {\n items := append([]int(nil), nums...)\n return len(items)\n}\n"
    )
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_go_pop_front_by_shifting_the_slice_down_is_quadratic() -> None:
    source = (
        "func f(nums []int) int {\n items := append([]int(nil), nums...)\n total := 0\n"
        " for len(items) > 0 {\n  total += items[0]\n  items = append(items[:0], items[1:]...)\n }\n return total\n}\n"
    )
    assert answer(source, "go") == ("O(n^2)", "O(n)")


def test_go_byte_conversion_copies_the_string() -> None:
    source = "func f(s string) int {\n b := []byte(s)\n return len(b)\n}\n"
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_go_conversion_between_slice_types_is_not_a_copy() -> None:
    source = "func f(nums []int) int {\n m := []int(nums)\n return len(m)\n}\n"
    assert answer(source, "go") == ("O(1)", "O(1)")


def test_js_spread_into_push_adds_every_element() -> None:
    source = "function f(a, b) {\n const out = [];\n out.push(...a);\n out.push(...b);\n return out;\n}\n"
    assert answer(source, "javascript") == ("O(n + m)", "O(n + m)")


# --------------------------------------------------------------------- callbacks (map/filter/...)
def test_js_map_with_a_copying_callback_is_rows_times_columns() -> None:
    source = "function deepCopy(grid) {\n  return grid.map((row) => [...row]);\n}\n"
    assert answer(source, "javascript") == ("O(n * m)", "O(n * m)")


def test_js_map_with_a_constant_callback_is_linear() -> None:
    source = "function f(nums) {\n  return nums.map((x) => x * 2);\n}\n"
    assert answer(source, "javascript") == ("O(n)", "O(n)")


def test_js_filter_then_for_each_is_linear() -> None:
    source = "function f(nums) {\n  let t = 0;\n  nums.filter((x) => x > 0).forEach((x) => { t += x; });\n  return t;\n}\n"
    assert answer(source, "javascript")[0] == "O(n)"


def test_js_reduce_with_an_inner_scan_is_quadratic() -> None:
    source = "function f(nums) {\n  return nums.reduce((acc, x) => acc + nums.indexOf(x), 0);\n}\n"
    assert answer(source, "javascript")[0] == "O(n^2)"


def test_java_stream_map_collect_is_linear() -> None:
    source = (
        "class S {\n List<Integer> f(List<Integer> nums) {\n"
        "  return nums.stream().map(x -> x * 2).collect(Collectors.toList());\n }\n}\n"
    )
    assert answer(source, "java")[0] == "O(n)"


def test_js_map_that_returns_its_rows_unchanged_shares_them() -> None:
    source = "function f(grid) {\n  return grid.map((row) => row);\n}\n"
    assert answer(source, "javascript") == ("O(n)", "O(n)")


def test_a_go_map_is_a_hash_map_so_delete_is_constant_time() -> None:
    source = (
        "func f(keys []int) int {\n seen := make(map[int]bool)\n for _, k := range keys {\n  seen[k] = true\n }\n"
        " for _, k := range keys {\n  delete(seen, k)\n }\n return len(seen)\n}\n"
    )
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_a_cpp_ordered_map_costs_a_logarithm_per_operation() -> None:
    source = (
        "int f(vector<int>& keys) {\n map<int,int> seen;\n for (int k : keys) {\n  seen[k] = 1;\n }\n"
        " for (int k : keys) {\n  seen.erase(k);\n }\n return seen.size();\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n log n)"


def test_inserting_into_a_cpp_ordered_map_by_subscript_costs_a_logarithm() -> None:
    source = "int f(vector<int>& keys) {\n map<int,int> m;\n for (int k : keys) {\n  m[k] = 1;\n }\n return m.size();\n}\n"
    assert answer(source, "cpp")[0] == "O(n log n)"


def test_cpp_copy_of_a_list_of_intervals_is_linear_because_rows_are_pairs() -> None:
    source = "int f(vector<vector<int>>& intervals) {\n vector<vector<int>> ordered = intervals;\n return ordered.size();\n}\n"
    assert answer(source, "cpp") == ("O(n)", "O(n)")


def test_cpp_copy_of_a_grid_is_rows_times_columns() -> None:
    source = (
        "int f(vector<vector<int>>& grid) {\n vector<vector<int>> c = grid;\n return c.size();\n}\n"
    )
    assert answer(source, "cpp") == ("O(n * m)", "O(n * m)")


# --------------------------------------------------------- a buffer that is flushed and reset
def test_python_buffer_joined_into_a_word_and_reset_is_linear() -> None:
    source = (
        "def f(s):\n    words = []\n    cur = []\n    for ch in s:\n        if ch == ' ':\n"
        "            words.append(''.join(cur))\n            cur = []\n        else:\n            cur.append(ch)\n"
        "    words.append(''.join(cur))\n    return len(words)\n"
    )
    assert answer(source) == ("O(n)", "O(n)")


def test_java_string_builder_flushed_and_cleared_is_linear() -> None:
    source = (
        "class S {\n int f(String s) {\n  List<String> words = new ArrayList<>();\n  StringBuilder cur = new StringBuilder();\n"
        "  for (int i = 0; i < s.length(); i++) {\n   char ch = s.charAt(i);\n"
        "   if (ch == ' ') { words.add(cur.toString()); cur.setLength(0); } else { cur.append(ch); }\n  }\n"
        "  words.add(cur.toString());\n  return words.size();\n }\n}\n"
    )
    assert answer(source, "java") == ("O(n)", "O(n)")


def test_go_buffer_resliced_to_zero_after_each_word_is_linear() -> None:
    source = (
        "func f(s string) int {\n words := []string{}\n cur := []byte{}\n for i := 0; i < len(s); i++ {\n"
        "  if s[i] == ' ' {\n   words = append(words, string(cur))\n   cur = cur[:0]\n  } else {\n"
        "   cur = append(cur, s[i])\n  }\n }\n words = append(words, string(cur))\n return len(words)\n}\n"
    )
    assert answer(source, "go") == ("O(n)", "O(n)")


def test_a_buffer_that_is_copied_every_iteration_without_a_reset_stays_quadratic() -> None:
    source = (
        "def f(s):\n    words = []\n    cur = []\n    for ch in s:\n        cur.append(ch)\n"
        "        words.append(''.join(cur))\n    return len(words)\n"
    )
    assert answer(source)[0] == "O(n^2)"


def test_a_buffer_copied_before_the_reset_and_again_after_it_is_not_amortised() -> None:
    # the second copy happens on a buffer that was not reset first on this path
    source = (
        "def f(s):\n    words = []\n    cur = []\n    for ch in s:\n        cur.append(ch)\n"
        "        words.append(''.join(cur))\n        if ch == ' ':\n            cur = []\n    return len(words)\n"
    )
    assert answer(source)[0] == "O(n^2)"


# -------------------------------------------------------------------- which function is the answer
PROGRAM_CPP = (
    "int main() {\n int n;\n cin >> n;\n long long total = 0;\n for (int i = 0; i < n; i++) {\n"
    "  int x;\n  cin >> x;\n  total += x;\n }\n cout << total << endl;\n return 0;\n}\n"
)


def test_a_program_that_reads_input_is_the_answer_even_with_an_unused_helper() -> None:
    helper = "int unused_helper(int x) {\n return x + 1;\n}\n\n"
    assert answer(helper + PROGRAM_CPP, "cpp") == answer(PROGRAM_CPP, "cpp") == ("O(n)", "O(1)")


def test_a_program_that_reads_input_includes_the_cost_of_the_functions_it_calls() -> None:
    source = (
        "int scan(int n) {\n int s = 0;\n for (int i = 0; i < n; i++) { s += i; }\n return s;\n}\n\n"
        "int main() {\n int n;\n cin >> n;\n cout << scan(n) << endl;\n return 0;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n)"


def test_a_main_that_only_calls_the_solution_on_literals_is_a_harness() -> None:
    source = (
        "int quadratic(vector<int>& a) {\n int c = 0;\n for (int i = 0; i < a.size(); i++) {\n"
        "  for (int j = 0; j < a.size(); j++) { c++; }\n }\n return c;\n}\n\n"
        "int main() {\n vector<int> v = {1, 2, 3};\n cout << quadratic(v) << endl;\n return 0;\n}\n"
    )
    assert answer(source, "cpp")[0] == "O(n^2)"


def test_an_unused_helper_next_to_a_java_main_program_does_not_change_the_answer() -> None:
    source = (
        "import java.util.Scanner;\n\npublic class Main {\n private int helper(int x) { return x + 1; }\n\n"
        " public static void main(String[] args) {\n  Scanner sc = new Scanner(System.in);\n"
        "  int n = sc.nextInt();\n  long total = 0;\n  for (int i = 0; i < n; i++) { total += sc.nextInt(); }\n"
        "  System.out.println(total);\n }\n}\n"
    )
    assert answer(source, "java") == ("O(n)", "O(1)")


# ------------------------------------------------------ loop indices in temporaries and callers
def test_a_slice_taken_between_two_loop_indices_costs_at_most_the_input() -> None:
    source = (
        "def f(s):\n    count = 0\n    for i in range(len(s)):\n        for j in range(i + 1, len(s) + 1):\n"
        "            sub = s[i:j]\n            if sub == sub[::-1]:\n                count += 1\n    return count\n"
    )
    assert answer(source) == ("O(n^3)", "O(n)")


def test_a_temporary_sized_by_a_loop_index_is_bounded_by_the_loop_range() -> None:
    source = "def f(n):\n    best = 0\n    for i in range(n):\n        tmp = [0] * i\n        best += len(tmp)\n    return best\n"
    assert answer(source) == ("O(n^2)", "O(n)")


ROW_SUMS_PY = (
    "def row_sum(grid, i):\n    total = 0\n    for x in grid[i]:\n        total += x\n    return total\n\n\n"
    "def all_row_sums(grid):\n    out = []\n    for i in range(len(grid)):\n        out.append(row_sum(grid, i))\n"
    "    return out\n"
)


def test_a_grid_passed_to_a_helper_that_walks_its_rows_costs_rows_times_columns() -> None:
    assert answer(ROW_SUMS_PY) == ("O(n * m)", "O(n)")


def test_the_same_in_javascript() -> None:
    source = (
        "function rowSum(grid, i) {\n  let total = 0;\n  for (const x of grid[i]) total += x;\n  return total;\n}\n\n"
        "function allRowSums(grid) {\n  const out = [];\n  for (let i = 0; i < grid.length; i++) out.push(rowSum(grid, i));\n"
        "  return out;\n}\n"
    )
    assert answer(source, "javascript") == ("O(n * m)", "O(n)")


def test_a_row_passed_to_a_helper_that_iterates_it_is_a_collection() -> None:
    source = (
        "def total(row):\n    s = 0\n    for x in row:\n        s += x\n    return s\n\n\n"
        "def f(grid):\n    t = 0\n    for r in grid:\n        t += total(r)\n    return t\n"
    )
    assert answer(source)[0] == "O(n * m)"


def test_mutual_calls_do_not_loop_the_role_inference() -> None:
    source = "def a(xs):\n    return b(xs) + len(xs)\n\n\ndef b(xs):\n    return a(xs)\n"
    result = analyze(source, "python")  # must terminate; the answer itself is not the point
    assert result.entry


# -------------------------------------------------- a loop whose test is not an order comparison
def test_halving_under_an_equality_test_is_logarithmic() -> None:
    source = "def f(n):\n    while n % 2 == 0:\n        n //= 2\n    return n == 1\n"
    assert answer(source) == ("O(log n)", "O(1)")


def test_dividing_out_a_factor_under_a_modulo_test_is_logarithmic_in_java() -> None:
    source = "class S {\n int f(int n) {\n  int c = 0;\n  while (n % 10 == 0) {\n   n /= 10;\n   c++;\n  }\n  return c;\n }\n}\n"
    assert answer(source, "java") == ("O(log n)", "O(1)")


def test_a_bit_test_loop_that_shifts_is_logarithmic() -> None:
    source = "int f(int n) {\n int c = 0;\n while ((n & 1) == 0) {\n  n >>= 1;\n  c++;\n }\n return c;\n}\n"
    assert answer(source, "cpp") == ("O(log n)", "O(1)")


def test_an_equality_test_loop_that_only_subtracts_is_not_assumed_logarithmic() -> None:
    # `n -= 2` under `n % 2 == 0` runs forever through zero and the negatives; no bound is proven
    source = "def f(n):\n    while n % 2 == 0:\n        n -= 2\n    return n\n"
    assert answer(source)[0] != "O(log n)"


def test_halving_while_another_variable_decides_the_exit_is_not_logarithmic() -> None:
    source = (
        "def f(n, m):\n    i = 0\n    while i < m:\n        n //= 2\n        i += 1\n    return n\n"
    )
    # the only size that appears is m, which the result names n
    assert answer(source)[0] == "O(n)"
