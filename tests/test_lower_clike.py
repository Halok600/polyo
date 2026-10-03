"""C, C++, Java and JavaScript -> engine AST (`analysis/lower_clike.py`), via tree-sitter.

Every expectation was written against a real parse tree of the construct (see the dumps in the
Phase 1 notes), not a guess at the grammar. Expected text is `render_module`'s canonical form."""
from __future__ import annotations

import pytest

from analysis.lower_clike import lower_clike
from analysis.nodes import FuncDef, Loop, walk
from analysis.nodes import render_module as render


def all_nodes(module):  # noqa: ANN001, ANN201
    for func in module.functions:
        yield from walk(func)
    for stmt in module.toplevel:
        yield from walk(stmt)


def lowered(source: str, language: str) -> list[str]:
    return render(lower_clike(source, language)).splitlines()


# ------------------------------------------------------------------------------------------ C++
def test_cpp_class_method_two_sum() -> None:
    source = """
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] + nums[j] == target) return {i, j};
            }
        }
        return {};
    }
};
"""
    assert lowered(source, "cpp") == [
        "def Solution.twoSum(nums: vector<int>, target: int):",
        "    n: int = nums.size()",
        "    for i: int = 0; i < n; i += 1:",
        "        for j: int = i + 1; j < n; j += 1:",
        "            if nums[i] + nums[j] == target:",
        "                return [i, j]",
        "    return []",
    ]


def test_cpp_constructor_style_declaration_is_an_allocation() -> None:
    text = lowered("void f(int n) {\n    vector<int> dp(n + 1, 0);\n}\n", "cpp")
    assert "    dp: vector<int> = new vector<int>(n + 1, 0)" in text


def test_cpp_array_declaration_keeps_its_size() -> None:
    module = lower_clike("void f() {\n    int counts[26] = {0};\n}\n", "cpp")
    decl = next(n for n in all_nodes(module) if hasattr(n, "decl") and getattr(n, "decl", None))
    assert decl.decl.dims == 1 and len(decl.decl.sizes) == 1


def test_cpp_declaration_without_initialiser_keeps_the_type() -> None:
    text = lowered("void f() {\n    unordered_map<int, int> seen;\n}\n", "cpp")
    assert "    seen: unordered_map<int, int>" in text


def test_cpp_range_for_and_reference_parameters() -> None:
    text = lowered("int f(vector<int>& nums) {\n    int s = 0;\n    for (int x : nums) s += x;\n    return s;\n}\n", "cpp")
    assert text[2:4] == ["    for x in nums:", "        s += x"]
    assert text[0] == "def f(nums: vector<int>):"


def test_cpp_qualified_and_member_calls_are_attribute_calls() -> None:
    text = lowered(
        "void f(vector<int>& v) {\n    std::sort(v.begin(), v.end());\n    v.push_back(1);\n}\n",
        "cpp",
    )
    assert "    std.sort(v.begin(), v.end())" in text
    assert "    v.push_back(1)" in text


def test_cpp_arrow_member_access() -> None:
    text = lowered("int f(ListNode* cur) {\n    return cur->next->val;\n}\n", "cpp")
    assert "    return cur.next.val" in text


def test_cpp_if_else_if_else_chain_and_while_and_do_while() -> None:
    text = lowered(
        "void f(int a, int b) {\n    if (a > b) { x = 1; } else if (a < b) { x = 2; } else { x = 3; }\n"
        "    while (lo <= hi) { lo = mid + 1; }\n    do { i++; } while (i < n);\n}\n",
        "cpp",
    )
    assert "    if a > b:" in text and "    elif a < b:" in text and "    else:" in text
    assert "    while lo <= hi:" in text
    assert "    do:" in text and "    while i < n" in text


def test_cpp_postfix_increments_inside_expressions_are_hoisted_after_the_statement() -> None:
    text = lowered("void f() {\n    tmp[k++] = nums[i++];\n}\n", "cpp")
    assert text[1] == "    tmp[k] = nums[i]"
    assert set(text[2:]) == {"    k += 1", "    i += 1"}


def test_cpp_compound_assignment_and_prefix_forms() -> None:
    text = lowered("void f() {\n    s += x;\n    --j;\n    i <<= 1;\n}\n", "cpp")
    assert text[1:] == ["    s += x", "    j -= 1", "    i <<= 1"]


def test_cpp_ternary_cast_new_and_lambda() -> None:
    text = lowered(
        "void f(vector<int>& nums, int n) {\n    int y = a > b ? a : b;\n"
        "    int s = (int)nums.size() - 1;\n    int* p = new int[n];\n"
        "    auto g = [&](int a, int b) { return a < b; };\n}\n",
        "cpp",
    )
    assert "    y: int = a if a > b else b" in text
    assert "    s: int = nums.size() - 1" in text
    assert "    p: int[] = new int[n]" in text
    assert "    g: auto = lambda a, b: <block>" in text


def test_cpp_logical_operators_become_and_or_not() -> None:
    text = lowered("bool f(int x) {\n    return !seen.count(x) && st.empty() || x > 3;\n}\n", "cpp")
    assert "    return not seen.count(x) and st.empty() or x > 3" in text


def test_cpp_namespace_and_template_wrappers_are_looked_through() -> None:
    source = "namespace ns {\n template <typename T>\n T id(T x) { return x; }\n}\n"
    module = lower_clike(source, "cpp")
    assert [f.name for f in module.functions] == ["id"]


def test_cpp_main_is_kept_as_a_function() -> None:
    module = lower_clike("int main() {\n    return 0;\n}\n", "cpp")
    assert [f.name for f in module.functions] == ["main"]


def test_cpp_out_of_class_method_definition_records_its_class() -> None:
    module = lower_clike("int Solution::count(int n) {\n    return n;\n}\n", "cpp")
    (func,) = module.functions
    assert func.cls == "Solution" and func.name == "count"


def test_cpp_pointer_returning_function_is_found_through_the_declarator() -> None:
    module = lower_clike("ListNode* rev(ListNode* head) {\n    return head;\n}\n", "cpp")
    (func,) = module.functions
    assert func.name == "rev" and func.params[0].name == "head"


# ------------------------------------------------------------------------------------------ C
def test_c_null_terminated_scan_and_allocation() -> None:
    source = """
int f(char* s, int n) {
    int* dp = calloc(n + 1, sizeof(int));
    for (int i = 0; s[i] != '\\0'; i++) { total += s[i]; }
    free(dp);
    return dp[n];
}
"""
    text = lowered(source, "c")
    assert "    dp: int[] = calloc(n + 1, 1)" in text
    assert "    for i: int = 0; s[i] != 0; i += 1:" in text


def test_c_struct_pointer_declaration() -> None:
    text = lowered("void f(struct ListNode* head) {\n    struct ListNode* cur = head;\n    while (cur) { cur = cur->next; }\n}\n", "c")
    assert "    cur: ListNode[] = head" in text
    assert "    while cur:" in text


# ---------------------------------------------------------------------------------------- Java
def test_java_array_and_generic_declarations() -> None:
    source = """
class Solution {
    public int[] f(int n) {
        int[] out = new int[n + 1];
        List<List<Integer>> g = new ArrayList<>();
        int[][] memo = new int[m][n];
        return out;
    }
}
"""
    text = lowered(source, "java")
    assert "    out: int[] = new int[n + 1]" in text
    assert "    g: List<List<Integer>> = new ArrayList()" in text
    assert "    memo: int[][] = new int[m][n]" in text


def test_java_loops_and_enhanced_for() -> None:
    source = """
class S {
    int f(int[] nums, int n) {
        for (int i = 0; i < n; i++) { total += nums[i]; }
        for (int x : nums) { s += x; }
        while (lo <= hi) { lo = mid + 1; }
        return 0;
    }
}
"""
    text = lowered(source, "java")
    assert "    for i: int = 0; i < n; i += 1:" in text
    assert "    for x in nums:" in text
    assert "    while lo <= hi:" in text


def test_java_method_calls_and_length_field() -> None:
    source = "class S {\n int f(int[] nums) {\n  seen.put(nums[i], i);\n  Arrays.sort(nums);\n  return nums.length;\n }\n}\n"
    text = lowered(source, "java")
    assert "    seen.put(nums[i], i)" in text
    assert "    Arrays.sort(nums)" in text
    assert "    return nums.length" in text


def test_java_array_literal_return_and_ternary() -> None:
    text = lowered("class S {\n int[] f(int i, int j) {\n  boolean b = !x ? true : false;\n  return new int[]{i, j};\n }\n}\n", "java")
    assert "    return [i, j]" in text
    assert "    b: boolean = False if not x else True" in text or "    b: boolean = True if not x else False" in text


def test_java_lambda_argument() -> None:
    text = lowered("class S {\n void f(List<Integer> list) {\n  Collections.sort(list, (p, q) -> p - q);\n }\n}\n", "java")
    assert "    Collections.sort(list, lambda p, q: p - q)" in text


def test_java_nested_classes_and_constructors_do_not_crash() -> None:
    source = "public class Main {\n static class Node { int v; Node(int v) { this.v = v; } }\n public static void main(String[] a) { }\n}\n"
    module = lower_clike(source, "java")
    assert "main" in [f.name for f in module.functions]


# ---------------------------------------------------------------------------------- JavaScript
def test_js_function_expression_assigned_to_var_becomes_a_named_function() -> None:
    source = """
var twoSum = function(nums, target) {
  const n = nums.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (nums[i] + nums[j] === target) return [i, j];
    }
  }
  return [];
};
"""
    module = lower_clike(source, "javascript")
    assert [f.name for f in module.functions] == ["twoSum"]
    text = render(module).splitlines()
    assert text[0] == "def twoSum(nums, target):"
    assert "    n = nums.length" in text
    assert "            if nums[i] + nums[j] == target:" in text  # === becomes ==


def test_js_arrow_functions_with_expression_bodies_return_their_value() -> None:
    module = lower_clike("const add = (a, b) => a + b;\n", "javascript")
    (func,) = module.functions
    assert func.name == "add"
    assert render(module).splitlines()[1] == "    return a + b"


def test_js_for_of_destructuring_and_postfix_in_subscript() -> None:
    text = lowered(
        "function f(queue, counts) {\n  for (const [k, v] of counts) { s += v; }\n"
        "  const [r, c] = queue[head++];\n}\n",
        "javascript",
    )
    assert "    for (k, v) in counts:" in text
    assert "    (r, c) = queue[head]" in text
    assert "    head += 1" in text


def test_js_new_expressions_and_method_chains() -> None:
    text = lowered(
        "function f(n) {\n  const seen = new Map();\n  const out = new Array(n + 1).fill(0);\n}\n",
        "javascript",
    )
    assert "    seen = new Map()" in text
    assert "    out = new Array(n + 1).fill(0)" in text


def test_js_script_level_code_becomes_toplevel() -> None:
    module = lower_clike('const n = parseInt(lines[0]);\nlet total = 0;\nfor (let i = 0; i < n; i++) { total += 1; }\n', "javascript")
    assert module.functions == ()
    assert len(module.toplevel) == 3


def test_js_class_methods_are_flattened() -> None:
    module = lower_clike("class Solution { method(x) { return x; } }\n", "javascript")
    (func,) = module.functions
    assert func.cls == "Solution" and func.name == "method"


def test_js_object_literal_in_call() -> None:
    text = lowered("function f(n) {\n  const g = Array.from({ length: n }, () => []);\n}\n", "javascript")
    assert '    g = Array.from({"length": n}, lambda : [])' in text


# ------------------------------------------------------------------------------ robustness
@pytest.mark.parametrize("language", ["c", "cpp", "java", "javascript"])
def test_empty_and_comment_only_sources_lower_to_an_empty_module(language: str) -> None:
    module = lower_clike("// nothing here\n", language)
    assert module.functions == () and module.toplevel == ()


def test_loop_kinds_in_cpp() -> None:
    module = lower_clike(
        "void f(vector<int>& a) {\n for (int i = 0; i < 3; i++) {}\n for (int x : a) {}\n while (a.size()) {}\n do {} while (0);\n}\n",
        "cpp",
    )
    kinds = [n.kind for n in all_nodes(module) if isinstance(n, Loop)]
    assert kinds == ["for_c", "for_each", "while", "do_while"]
    assert all(isinstance(f, FuncDef) for f in module.functions)


# ------------------------------------------------------------- gaps found by scanning the goldens
def test_cpp_dereference_of_a_call_result_is_an_index_zero() -> None:
    text = lowered(
        "int f(vector<int>& a) {\n    int hi = *max_element(a.begin(), a.end());\n    return hi;\n}\n",
        "cpp",
    )
    assert "    hi: int = max_element(a.begin(), a.end())[0]" in text


def test_js_in_operator_is_a_membership_comparison() -> None:
    text = lowered("function f(ch, pairs) {\n  if (ch in pairs) { return 1; }\n}\n", "javascript")
    assert "    if ch in pairs:" in text


def test_java_null_literal_is_none() -> None:
    text = lowered("class S {\n boolean f(Node n) {\n  return n == null;\n }\n}\n", "java")
    assert "    return n == None" in text


def test_java_declaration_initialiser_side_effects_are_hoisted_not_lost() -> None:
    text = lowered("class S {\n int f(int[] arr, int i) {\n  int x = arr[i++];\n  return x;\n }\n}\n", "java")
    assert "    x: int = arr[i]" in text
    assert "    i += 1" in text
