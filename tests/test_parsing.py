"""Tests for the tree-sitter parsing driver and language detection (Phase 1,
plan §5/§12 `parsing/parse.py`)."""
from __future__ import annotations

import pytest

from parsing.parse import (
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
    detect_language,
    parse_source,
)


def test_supported_languages_are_python_and_cpp_in_phase_1():
    assert SUPPORTED_LANGUAGES == {"python", "cpp"}


def test_detect_language_from_python_extension():
    assert detect_language("solution.py") == "python"


def test_detect_language_from_cpp_extensions():
    assert detect_language("solution.cpp") == "cpp"
    assert detect_language("solution.cc") == "cpp"
    assert detect_language("solution.hpp") == "cpp"


def test_detect_language_rejects_unknown_extension():
    with pytest.raises(UnsupportedLanguageError):
        detect_language("solution.rs")


def test_parse_source_returns_root_node_with_expected_type_for_python():
    tree = parse_source("def f():\n    pass\n", "python")
    assert tree.root_node.type == "module"


def test_parse_source_returns_root_node_with_expected_type_for_cpp():
    tree = parse_source("int f() { return 0; }", "cpp")
    assert tree.root_node.type == "translation_unit"


def test_parse_source_rejects_unsupported_language():
    with pytest.raises(UnsupportedLanguageError):
        parse_source("fn f() {}", "rust")


def test_parse_source_marks_syntax_errors():
    tree = parse_source("def f(:\n", "python")
    assert tree.root_node.has_error
