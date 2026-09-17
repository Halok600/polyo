"""Tests for the tree-sitter parsing driver and language detection (plan
§5/§12 `parsing/parse.py`). Phase 1 shipped Python + C++; Phase 4 added
Java, JavaScript, C and Go."""
from __future__ import annotations

import pytest

from parsing.parse import (
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
    detect_language,
    parse_source,
)


def test_supported_languages_cover_all_six_target_languages():
    assert SUPPORTED_LANGUAGES == {"python", "cpp", "java", "javascript", "c", "go"}


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


def test_detect_language_rejects_unknown_extension():
    with pytest.raises(UnsupportedLanguageError):
        detect_language("solution.rs")


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


def test_parse_source_rejects_unsupported_language():
    with pytest.raises(UnsupportedLanguageError):
        parse_source("fn f() {}", "rust")


def test_parse_source_marks_syntax_errors():
    tree = parse_source("def f(:\n", "python")
    assert tree.root_node.has_error
