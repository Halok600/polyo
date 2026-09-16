"""tree-sitter parsing driver and language detection.

Phase 1 supports Python and C++ only (plan §14). Adding a language here is a
new grammar dependency plus one entry in each of the two dicts below -- the
per-language semantics live in `parsing/lang/<lang>.toml`, consumed by
`parsing/normalize.py`, not here.
"""
from __future__ import annotations

from pathlib import PurePath

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language, Parser, Tree

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"python", "cpp"})

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
}

_LANGUAGE_CAPSULES: dict[str, Language] = {
    "python": Language(tree_sitter_python.language()),
    "cpp": Language(tree_sitter_cpp.language()),
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
