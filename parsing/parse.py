"""tree-sitter parsing driver and language detection.

Phase 1 shipped Python and C++; Phase 4 (plan §14) adds Java, JavaScript, C
and Go. Adding a language here is a new grammar dependency plus one entry in
each of the three dicts below -- the per-language semantics live in
`parsing/lang/<lang>.toml`, consumed by `parsing/normalize.py`, not here.

`.h` is ambiguous between C and C++ in the wild; it stayed mapped to "cpp"
(Phase 1's choice) rather than being reassigned, since a C++ header is by far
the more common case in this project's corpora and repointing it would be a
silent behaviour change for existing C++ inputs.
"""
from __future__ import annotations

from pathlib import PurePath

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Parser, Tree

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"python", "cpp", "java", "javascript", "c", "go"}
)

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".mjs": "javascript",
    ".c": "c",
    ".go": "go",
}

_LANGUAGE_CAPSULES: dict[str, Language] = {
    "python": Language(tree_sitter_python.language()),
    "cpp": Language(tree_sitter_cpp.language()),
    "java": Language(tree_sitter_java.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "c": Language(tree_sitter_c.language()),
    "go": Language(tree_sitter_go.language()),
}


class UnsupportedLanguageError(ValueError):
    """Raised for a language or extension Phase 1 does not yet support."""


def detect_language(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    try:
        return _EXTENSION_TO_LANGUAGE[ext]
    except KeyError as e:
        raise UnsupportedLanguageError(f"no known language for extension {ext!r}") from e


def parse_source(source: str, language: str) -> Tree:
    try:
        ts_language = _LANGUAGE_CAPSULES[language]
    except KeyError as e:
        raise UnsupportedLanguageError(f"unsupported language: {language!r}") from e
    parser = Parser(ts_language)
    return parser.parse(source.encode("utf-8"))
