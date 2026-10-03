"""tree-sitter parsing driver and language detection.

Phase 1 shipped Python and C++; Phase 4 (plan §14) adds Java, JavaScript, C
and Go (Tier 1/2, plan §3 -- parse + IR, and for Tier 1 an oracle driver
too). Phase 7 adds TypeScript, Rust, C#, Kotlin (Tier 3, plan §3: IR
mapping file only, no oracle -- these never get a `codegen/<lang>.j2` or a
labelled corpus, but they parse and predict through the exact same
generic walker, which is the whole point: "adding a language is a config
file, not a rewrite"). Adding a language here is a new grammar dependency
plus one entry in each of the three dicts below -- the per-language
semantics live in `parsing/lang/<lang>.toml`, consumed by
`parsing/normalize.py`, not here.

`.h` is ambiguous between C and C++ in the wild; it stayed mapped to "cpp"
(Phase 1's choice) rather than being reassigned, since a C++ header is by far
the more common case in this project's corpora and repointing it would be a
silent behaviour change for existing C++ inputs.
"""

from __future__ import annotations

import importlib
from pathlib import PurePath

from tree_sitter import Language, Parser, Tree

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"python", "cpp", "java", "javascript", "c", "go", "typescript", "rust", "csharp", "kotlin"}
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
    ".ts": "typescript",
    ".rs": "rust",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
}


def _grammar(module: str, factory: str = "language") -> Language:
    return Language(getattr(importlib.import_module(module), factory)())


def _optional_grammar(module: str, factory: str = "language") -> Language | None:
    """Tier 3 grammars are optional: the slim serving image installs only the six languages it
    serves (requirements-api.txt), and a grammar that is not installed is a language that is not
    available here, not an import error for everyone."""
    try:
        return _grammar(module, factory)
    except ImportError:
        return None


_LANGUAGE_CAPSULES: dict[str, Language] = {
    "python": _grammar("tree_sitter_python"),
    "cpp": _grammar("tree_sitter_cpp"),
    "java": _grammar("tree_sitter_java"),
    "javascript": _grammar("tree_sitter_javascript"),
    "c": _grammar("tree_sitter_c"),
    "go": _grammar("tree_sitter_go"),
}
for _name, _module, _factory in (
    # Not `language_tsx` -- this project's IR has no JSX-specific symbols.
    ("typescript", "tree_sitter_typescript", "language_typescript"),
    ("rust", "tree_sitter_rust", "language"),
    ("csharp", "tree_sitter_c_sharp", "language"),
    ("kotlin", "tree_sitter_kotlin", "language"),
):
    _tier3 = _optional_grammar(_module, _factory)
    if _tier3 is not None:
        _LANGUAGE_CAPSULES[_name] = _tier3


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


def _error_node_count(tree: Tree) -> int:
    # tree-sitter is error-tolerant: an ERROR node (wrong grammar) or a
    # MISSING node (the parser inserted a token to recover) is exactly what
    # a wrong-language guess produces, and what a genuine syntax error in
    # the right language also produces -- this heuristic can't tell those
    # apart, but detect_language_from_source only needs "closest grammar",
    # not "valid code".
    count = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            count += 1
        stack.extend(node.children)
    return count


def detect_language_from_source(source: str) -> str:
    """Language auto-detection from raw source text (plan §10's `"auto"`
    request field) -- `detect_language` above only works from a filename
    extension, which the API's `/v1/predict` request never has. Parses
    `source` with every supported grammar and picks the one producing the
    fewest ERROR/MISSING nodes: cheap (no keyword heuristics to maintain
    per language), and reuses the same tree-sitter grammars already
    installed for real parsing rather than a second, separate detector."""
    source_bytes = source.encode("utf-8")
    best_language: str | None = None
    best_errors = -1
    for language in sorted(_LANGUAGE_CAPSULES):  # the grammars installed here
        parser = Parser(_LANGUAGE_CAPSULES[language])
        tree = parser.parse(source_bytes)
        errors = _error_node_count(tree)
        if best_language is None or errors < best_errors:
            best_language, best_errors = language, errors
    assert best_language is not None  # SUPPORTED_LANGUAGES is never empty
    return best_language
