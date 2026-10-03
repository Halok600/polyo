"""Go -> engine AST (`analysis/lower_go.py`), via tree-sitter.

Go's grammar is shaped differently from the C family (statement lists, expression lists,
`:=`, one `for` keyword for every loop, slices and maps as built-in types), hence its own lowerer.
Expectations were written against real parse trees."""
from __future__ import annotations

from analysis.lower_go import lower_go
from analysis.nodes import Loop, walk
from analysis.nodes import render_module as render


def lowered(source: str) -> list[str]:
    return render(lower_go(source)).splitlines()


def all_nodes(module):  # noqa: ANN001, ANN201
    for func in module.functions:
        yield from walk(func)
    for stmt in module.toplevel:
        yield from walk(stmt)


def test_two_sum() -> None:
    source = """
func twoSum(nums []int, target int) []int {
    n := len(nums)
    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            if nums[i]+nums[j] == target {
                return []int{i, j}
            }
        }
    }
    return []int{}
}
"""
    assert lowered(source) == [
        "def twoSum(nums: int[], target: int):",
        "    n = len(nums)",
        "    for i = 0; i < n; i += 1:",
        "        for j = i + 1; j < n; j += 1:",
        "            if nums[i] + nums[j] == target:",
        "                return [i, j]",
        "    return []",
    ]


def test_range_forms() -> None:
    text = lowered(
        "func f(nums []int) {\n    for _, x := range nums { s += x }\n    for i := range nums { s += i }\n}\n"
    )
    assert "    for (_, x) in nums:" in text
    assert "    for i in nums:" in text


def test_for_as_while_and_forever() -> None:
    text = lowered("func f(lo int, hi int) {\n    for lo <= hi { lo = mid + 1 }\n    for { break }\n}\n")
    assert "    while lo <= hi:" in text
    assert "    while True:" in text


def test_if_with_initialiser_and_else() -> None:
    text = lowered(
        "func f(seen map[int]int, target int, x int) []int {\n"
        "    if j, ok := seen[target-x]; ok { return []int{j, 1} } else { x = 2 }\n    return nil\n}\n"
    )
    assert "    (j, ok) = seen[target - x]" in text
    assert "    if ok:" in text and "    else:" in text


def test_make_allocations() -> None:
    text = lowered(
        "func f(n int) {\n    seen := make(map[int]int)\n    out := make([]int, n+1)\n"
        "    var counts [26]int\n}\n"
    )
    assert "    seen = new map<int, int>()" in text
    assert "    out = new int[](n + 1)" in text
    assert "    counts: int[]" in text


def test_array_declaration_keeps_its_size() -> None:
    module = lower_go("func f() {\n    var counts [26]int\n}\n")
    decl = next(n for n in all_nodes(module) if getattr(n, "decl", None))
    assert decl.decl.dims == 1 and len(decl.decl.sizes) == 1


def test_tuple_assignment_and_inc_dec_and_compound_ops() -> None:
    text = lowered("func f(a int, b int) {\n    a, b = b, a+b\n    i++\n    j--\n    x += y\n    n >>= 1\n}\n")
    assert "    (a, b) = (b, a + b)" in text
    assert "    i += 1" in text and "    j -= 1" in text
    assert "    x += y" in text and "    n >>= 1" in text


def test_slices_and_append() -> None:
    text = lowered("func f(s string, q []int, i int, j int) {\n    sub := s[i:j]\n    q = append(q, 1)\n    t := s[i:]\n}\n")
    assert "    sub = s[i:j]" in text
    assert "    q = append(q, 1)" in text
    assert "    t = s[i:]" in text


def test_methods_record_their_receiver_type() -> None:
    module = lower_go("func (p PQ) Len() int { return len(p) }\nfunc (h *IntHeap) Push(x any) { *h = append(*h, x.(int)) }\n")
    assert [(f.cls, f.name) for f in module.functions] == [("PQ", "Len"), ("IntHeap", "Push")]
    assert [p.name for p in module.functions[0].params] == []


def test_package_imports_types_and_main_are_handled() -> None:
    source = """
package main

import "fmt"

type Item struct { dist int; node int }

func main() {
    fmt.Println(twoSum([]int{2, 7}, 9))
}
"""
    module = lower_go(source)
    assert [f.name for f in module.functions] == ["main"]


def test_composite_literals_and_closures() -> None:
    text = lowered(
        "func f(a [][]int) {\n    dirs := [][2]int{{1, 0}, {-1, 0}}\n"
        "    sort.Slice(a, func(i, j int) bool { return a[i][0] < a[j][0] })\n}\n"
    )
    assert "    dirs = [[1, 0], [-1, 0]]" in text
    assert "    sort.Slice(a, lambda i, j: <block>)" in text


def test_switch_and_defer_do_not_crash() -> None:
    module = lower_go(
        "func f(x int) {\n    switch x { case 1: y = 1 default: y = 2 }\n    defer cleanup()\n}\n"
    )
    assert [f.name for f in module.functions] == ["f"]


def test_loop_kinds() -> None:
    module = lower_go(
        "func f(a []int) {\n    for i := 0; i < 3; i++ {}\n    for _, x := range a {}\n    for len(a) > 0 {}\n}\n"
    )
    assert [n.kind for n in all_nodes(module) if isinstance(n, Loop)] == ["for_c", "for_each", "while"]


def test_multiple_return_values_and_named_results() -> None:
    text = lowered("func f(a int, b int) (int, int) {\n    return b, a\n}\n")
    assert "    return (b, a)" in text


def test_grouped_parameters_share_a_type() -> None:
    module = lower_go("func f(a, b int, s string) {}\n")
    params = module.functions[0].params
    assert [(p.name, p.type.render() if p.type else None) for p in params] == [
        ("a", "int"), ("b", "int"), ("s", "string"),
    ]


def test_empty_source() -> None:
    module = lower_go("// nothing\n")
    assert module.functions == () and module.toplevel == ()
