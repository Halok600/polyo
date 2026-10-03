"""Semantics-preserving source perturbations (`eval/perturb.py`).

Each transform must change the TEXT of a program without changing what it computes or how costly
it is. They exist to answer one question about any predictor: does its answer move when the code
changes only superficially? (The shipped GNN's did: 40 dead statements flipped O(n^2) to
O(n log n).)
"""
from __future__ import annotations

import ast

import pytest

from eval.perturb import (
    PERTURBATIONS,
    add_comments,
    add_dead_code,
    apply_all,
    insert_unused_helper,
    reformat_whitespace,
    rename_identifiers,
    reorder_functions,
)
from parsing.parse import parse_source

_PY = '''def count_pairs(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count
'''

_CPP = '''int countPairs(vector<int>& nums, int target) {
    int count = 0;
    for (int i = 0; i < nums.size(); i++) {
        for (int j = i + 1; j < nums.size(); j++) {
            if (nums[i] + nums[j] == target) count++;
        }
    }
    return count;
}
'''

_JAVA = '''class Solution {
    public int countPairs(int[] nums, int target) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) count++;
            }
        }
        return count;
    }
}
'''

_JS = '''function countPairs(nums, target) {
  let count = 0;
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) count++;
    }
  }
  return count;
}
'''

_GO = '''func countPairs(nums []int, target int) int {
    count := 0
    for i := 0; i < len(nums); i++ {
        for j := i + 1; j < len(nums); j++ {
            if nums[i]+nums[j] == target {
                count++
            }
        }
    }
    return count
}
'''

_C = '''int countPairs(int* nums, int n, int target) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (nums[i] + nums[j] == target) count++;
        }
    }
    return count;
}
'''

_BY_LANGUAGE = {
    "python": _PY,
    "cpp": _CPP,
    "java": _JAVA,
    "javascript": _JS,
    "go": _GO,
    "c": _C,
}


def _leaf_tokens(source: str, language: str) -> list[tuple[str, bytes]]:
    """Comment-free (type, text) leaf sequence: equal sequences mean the same program text modulo
    whitespace and comments."""
    tree = parse_source(source, language)
    out: list[tuple[str, bytes]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if "comment" in node.type:
            continue
        if node.child_count == 0:
            out.append((node.type, node.text or b""))
        else:
            stack.extend(reversed(node.children))
    return out


def _assert_parses(source: str, language: str) -> None:
    if language == "python":
        ast.parse(source)
    else:
        assert not parse_source(source, language).root_node.has_error, source


@pytest.mark.parametrize("language", list(_BY_LANGUAGE))
def test_dead_code_adds_statements_and_still_parses(language: str) -> None:
    source = _BY_LANGUAGE[language]
    out = add_dead_code(source, language, k=12)
    assert out is not None and out != source
    _assert_parses(out, language)
    assert out.count("pad") >= 12
    # The original program is still there, in order, after the padding.
    for line in source.splitlines():
        assert line.strip() in out


def test_dead_code_python_keeps_ast_of_original_statements() -> None:
    out = add_dead_code(_PY, "python", k=5)
    assert out is not None
    original = ast.parse(_PY).body[0]
    padded = ast.parse(out).body[0]
    assert isinstance(original, ast.FunctionDef) and isinstance(padded, ast.FunctionDef)
    assert [ast.dump(s) for s in padded.body[-len(original.body):]] == [
        ast.dump(s) for s in original.body
    ]


def test_rename_changes_variables_but_not_literals_members_or_function_names() -> None:
    source = '''def find(nums):
    count = 0
    for ch in "i and n":
        if ch == 'i':
            count += nums.count(1)
    return count
'''
    out = rename_identifiers(source, "python")
    assert out is not None
    ast.parse(out)
    assert '"i and n"' in out and "'i'" in out  # literals untouched
    assert ".count(1)" in out  # a member access is not a local variable
    assert "def find(" in out  # function names are untouched
    assert "nums_r" in out and "count_r" in out


def test_rename_skips_names_that_are_called_as_functions() -> None:
    source = (
        "def total(nums):\n    return sum(nums)\n\n\ndef run(nums):\n    return total(nums)\n"
    )
    out = rename_identifiers(source, "python")
    assert out is not None
    assert "def total(" in out and "return total(" in out  # the function name is untouched ...
    assert "nums_r" in out  # ... while an ordinary variable in the same file is renamed


@pytest.mark.parametrize("language", list(_BY_LANGUAGE))
def test_rename_preserves_structure(language: str) -> None:
    source = _BY_LANGUAGE[language]
    out = rename_identifiers(source, language)
    assert out is not None and out != source
    _assert_parses(out, language)
    before = _leaf_tokens(source, language)
    after = _leaf_tokens(out, language)
    assert [t for t, _ in before] == [t for t, _ in after]


@pytest.mark.parametrize("language", list(_BY_LANGUAGE))
def test_comments_do_not_change_the_program(language: str) -> None:
    source = _BY_LANGUAGE[language]
    out = add_comments(source, language)
    assert out is not None and out != source
    _assert_parses(out, language)
    assert _leaf_tokens(out, language) == _leaf_tokens(source, language)


@pytest.mark.parametrize("language", list(_BY_LANGUAGE))
def test_reformatting_does_not_change_the_program(language: str) -> None:
    source = _BY_LANGUAGE[language]
    out = reformat_whitespace(source, language)
    assert out is not None and out != source
    _assert_parses(out, language)
    if language == "python":
        assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(source))
    else:
        assert _leaf_tokens(out, language) == _leaf_tokens(source, language)


@pytest.mark.parametrize("language", list(_BY_LANGUAGE))
def test_unused_helper_is_added_and_never_called(language: str) -> None:
    source = _BY_LANGUAGE[language]
    out = insert_unused_helper(source, language)
    assert out is not None and out != source
    _assert_parses(out, language)
    assert out.count("unused_helper_zz") == 1  # defined once, referenced nowhere else
    assert source.strip().splitlines()[0].strip() in out


def test_reorder_functions_reverses_python_top_level_defs() -> None:
    source = "def a(x):\n    return x\n\n\ndef b(x):\n    return a(x)\n"
    out = reorder_functions(source, "python")
    assert out is not None
    names = [n.name for n in ast.parse(out).body if isinstance(n, ast.FunctionDef)]
    assert names == ["b", "a"]


def test_reorder_is_not_applicable_to_a_single_function() -> None:
    assert reorder_functions(_PY, "python") is None
    assert reorder_functions(_CPP, "cpp") is None  # C++ needs declaration before use


def test_apply_all_returns_only_applicable_changed_variants() -> None:
    variants = apply_all(_PY, "python")
    names = [name for name, _ in variants]
    assert len(names) == len(set(names))
    assert {"dead_code", "rename", "comments", "reformat", "unused_helper"} <= set(names)
    assert "reorder_functions" not in names  # only one function
    assert all(text != _PY for _, text in variants)


def test_every_registered_perturbation_has_a_unique_name() -> None:
    names = [p.name for p in PERTURBATIONS]
    assert len(names) == len(set(names))
