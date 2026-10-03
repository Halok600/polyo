"""The engine explains itself: a line-numbered derivation next to every answer.

`analyze(...).steps` lists what the answer rests on (each loop's trip count and total cost, costly
library calls, how a recursion was solved, sizeable allocations); `.assumptions` lists the bounds
the engine had to assume. The API serves both, so a user can check the reasoning instead of trusting
a bare class.
"""

from __future__ import annotations

from analysis.engine import analyze

NESTED = """\
def has_pair(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return True
    return False
"""

FIB = """\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"""

MERGE_SORT = """\
def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
"""


def _texts(source: str, language: str = "python", kind: str | None = None) -> list[str]:
    steps = analyze(source, language).steps
    return [s.text for s in steps if kind is None or s.kind == kind]


def test_a_nested_loop_reports_both_loops_by_line_and_cost() -> None:
    result = analyze(NESTED, "python")
    loops = [s for s in result.steps if s.kind == "loop"]
    assert [s.line for s in loops] == [2, 3]
    outer, inner = loops
    assert "O(n)" in outer.text and "O(n^2)" in outer.text
    assert "O(n)" in inner.text


def test_steps_are_ordered_by_line_and_end_with_the_totals() -> None:
    steps = analyze(NESTED, "python").steps
    body = [s for s in steps if s.kind != "total"]
    assert [s.line for s in body] == sorted(s.line for s in body)
    assert [s.kind for s in steps[-2:]] == ["total", "total"]
    assert "time" in steps[-2].text and "O(n^2)" in steps[-2].text
    assert "space" in steps[-1].text and "O(1)" in steps[-1].text


def test_every_language_explains_the_same_nested_loop() -> None:
    java = """\
class Solution {
    boolean hasPair(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) return true;
            }
        }
        return false;
    }
}
"""
    loops = [s for s in analyze(java, "java").steps if s.kind == "loop"]
    assert [s.line for s in loops] == [3, 4]
    assert "O(n^2)" in loops[0].text


def test_a_costly_library_call_is_listed() -> None:
    source = "def f(nums):\n    nums.sort()\n    return nums[0]\n"
    calls = _texts(source, kind="call")
    assert any("sort" in text and "O(n log n)" in text for text in calls)


def test_a_recursion_says_how_it_was_solved() -> None:
    fib = [s for s in analyze(FIB, "python").steps if s.kind == "recursion"]
    assert len(fib) == 1 and fib[0].line == 1
    assert "O(2^n)" in fib[0].text
    merge = [s for s in analyze(MERGE_SORT, "python").steps if s.kind == "recursion"]
    assert len(merge) == 1 and "O(n log n)" in merge[0].text


def test_an_amortised_loop_says_so() -> None:
    source = """\
def longest(s):
    seen = {}
    left = 0
    best = 0
    for right in range(len(s)):
        while s[right] in seen and left < right:
            seen.pop(s[left], None)
            left += 1
        seen[s[right]] = right
        best = max(best, right - left + 1)
    return best
"""
    loops = [s for s in analyze(source, "python").steps if s.kind == "loop"]
    assert any("amortised" in s.text for s in loops)


def test_a_large_allocation_is_listed() -> None:
    source = "def f(n, m):\n    grid = [[0] * m for _ in range(n)]\n    return grid\n"
    allocations = _texts(source, kind="alloc")
    assert any("O(n * m)" in text for text in allocations)


def test_assumptions_carry_the_line_and_the_reason() -> None:
    source = """\
def f(nums):
    i = 0
    while mystery(i):
        i += 1
    return i
"""
    result = analyze(source, "python")
    assert result.certainty in ("assumed", "unknown")
    assert result.assumptions, "an assumed bound must be reported"
    first = result.assumptions[0]
    assert first.line == 3 and first.reason


def test_a_certain_answer_has_no_assumptions() -> None:
    assert analyze(NESTED, "python").assumptions == []


def test_variables_use_the_same_letters_as_the_answer() -> None:
    source = "def f(a, b):\n    for x in a:\n        for y in b:\n            pass\n"
    result = analyze(source, "python")
    assert result.time.text == "O(n * m)"
    loops = [s for s in result.steps if s.kind == "loop"]
    assert "O(n * m)" in loops[0].text and "O(m)" in loops[1].text


def test_the_derivation_is_deterministic() -> None:
    assert _texts(NESTED) == _texts(NESTED)
    assert _texts(MERGE_SORT) == _texts(MERGE_SORT)


def test_garbage_has_an_empty_or_total_only_derivation() -> None:
    result = analyze("def f():\n    return 1\n", "python")
    assert all(s.kind == "total" for s in result.steps)
