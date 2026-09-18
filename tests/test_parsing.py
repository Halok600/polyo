"""Tests for the tree-sitter parsing driver and language detection (plan
§5/§12 `parsing/parse.py`). Phase 1 shipped Python + C++; Phase 4 added
Java, JavaScript, C and Go (Tier 1/2); Phase 7 added TypeScript, Rust, C#
and Kotlin (Tier 3, plan §3: IR mapping file only, no oracle)."""
from __future__ import annotations

import pytest

from parsing.parse import (
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
    detect_language,
    detect_language_from_source,
    parse_source,
)


def test_supported_languages_cover_every_tier_1_2_and_3_language():
    assert SUPPORTED_LANGUAGES == {
        "python",
        "cpp",
        "java",
        "javascript",
        "c",
        "go",
        "typescript",
        "rust",
        "csharp",
        "kotlin",
    }


def test_detect_language_from_python_extension():
    assert detect_language("solution.py") == "python"


def test_detect_language_from_cpp_extensions():
    assert detect_language("solution.cpp") == "cpp"
    assert detect_language("solution.cc") == "cpp"
    assert detect_language("solution.hpp") == "cpp"


def test_detect_language_from_phase_4_extensions():
    assert detect_language("Solution.java") == "java"
    assert detect_language("solution.js") == "javascript"
    assert detect_language("solution.c") == "c"
    assert detect_language("solution.go") == "go"


def test_detect_language_from_tier_3_extensions():
    assert detect_language("solution.ts") == "typescript"
    assert detect_language("solution.rs") == "rust"
    assert detect_language("Solution.cs") == "csharp"
    assert detect_language("solution.kt") == "kotlin"
    assert detect_language("solution.kts") == "kotlin"


def test_detect_language_rejects_unknown_extension():
    with pytest.raises(UnsupportedLanguageError):
        detect_language("solution.rb")


def test_parse_source_returns_root_node_with_expected_type_for_python():
    tree = parse_source("def f():\n    pass\n", "python")
    assert tree.root_node.type == "module"


def test_parse_source_returns_root_node_with_expected_type_for_cpp():
    tree = parse_source("int f() { return 0; }", "cpp")
    assert tree.root_node.type == "translation_unit"


def test_parse_source_returns_root_node_with_expected_type_for_java():
    tree = parse_source("class Sol { int f() { return 0; } }", "java")
    assert tree.root_node.type == "program"


def test_parse_source_returns_root_node_with_expected_type_for_javascript():
    tree = parse_source("function f() { return 0; }", "javascript")
    assert tree.root_node.type == "program"


def test_parse_source_returns_root_node_with_expected_type_for_c():
    tree = parse_source("int f() { return 0; }", "c")
    assert tree.root_node.type == "translation_unit"


def test_parse_source_returns_root_node_with_expected_type_for_go():
    tree = parse_source("func f() int { return 0 }", "go")
    assert tree.root_node.type == "source_file"


def test_parse_source_returns_root_node_with_expected_type_for_typescript():
    tree = parse_source("function f(): number { return 0; }", "typescript")
    assert tree.root_node.type == "program"


def test_parse_source_returns_root_node_with_expected_type_for_rust():
    tree = parse_source("fn f() -> i32 { return 0; }", "rust")
    assert tree.root_node.type == "source_file"


def test_parse_source_returns_root_node_with_expected_type_for_csharp():
    tree = parse_source("class C { int f() { return 0; } }", "csharp")
    assert tree.root_node.type == "compilation_unit"


def test_parse_source_returns_root_node_with_expected_type_for_kotlin():
    tree = parse_source("fun f(): Int { return 0 }", "kotlin")
    assert tree.root_node.type == "source_file"


def test_parse_source_rejects_unsupported_language():
    with pytest.raises(UnsupportedLanguageError):
        parse_source("puts 'hi'", "ruby")


def test_parse_source_marks_syntax_errors():
    tree = parse_source("def f(:\n", "python")
    assert tree.root_node.has_error


_SOURCE_BY_LANGUAGE = {
    "python": (
        "def linear_search(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    ),
    "cpp": (
        "int f(std::vector<int>& xs) {\n"
        "    int total = 0;\n"
        "    for (int x : xs) {\n"
        "        total += x;\n"
        "    }\n"
        "    return total;\n"
        "}\n"
    ),
    "java": (
        "class Solution {\n"
        "    int f(int[] xs) {\n"
        "        int total = 0;\n"
        "        for (int x : xs) {\n"
        "            total += x;\n"
        "        }\n"
        "        return total;\n"
        "    }\n"
        "}\n"
    ),
    "javascript": (
        "function f(xs) {\n"
        "    let total = 0;\n"
        "    for (const x of xs) {\n"
        "        total += x;\n"
        "    }\n"
        "    return total;\n"
        "}\n"
    ),
    "c": (
        "int f(int* xs, int n) {\n"
        "    int total = 0;\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        total += xs[i];\n"
        "    }\n"
        "    return total;\n"
        "}\n"
    ),
    "go": (
        "func f(xs []int) int {\n"
        "    total := 0\n"
        "    for _, x := range xs {\n"
        "        total += x\n"
        "    }\n"
        "    return total\n"
        "}\n"
    ),
}


@pytest.mark.parametrize("language", sorted(_SOURCE_BY_LANGUAGE))
def test_detect_language_from_source_identifies_each_supported_language(language):
    assert detect_language_from_source(_SOURCE_BY_LANGUAGE[language]) == language


def test_detect_language_from_source_returns_some_supported_language_for_garbage_input():
    # No crash, no exception -- the heuristic picks the *closest* grammar
    # even for nonsense input; there is no "reject" path by design (plan
    # §10's auto-detect must always return something the rest of the
    # pipeline can act on).
    assert detect_language_from_source("!!! not code at all ???") in SUPPORTED_LANGUAGES
