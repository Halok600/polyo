"""One entry point from source text to the engine's AST, for every served language."""

from __future__ import annotations

from analysis.lower_clike import lower_clike
from analysis.lower_go import lower_go
from analysis.lower_python import lower_python
from analysis.nodes import LoweringError, Module

LOWERABLE_LANGUAGES = frozenset({"python", "cpp", "java", "javascript", "c", "go"})


def lower_source(source: str, language: str) -> Module:
    """Lower `source` (written in `language`) to a `Module`.

    Raises `LoweringError` for a language the engine does not handle or for Python that the
    stdlib parser rejects. The tree-sitter languages are error-tolerant and always return a
    (possibly empty) module.
    """
    if language == "python":
        return lower_python(source)
    if language in ("c", "cpp", "java", "javascript"):
        return lower_clike(source, language)
    if language == "go":
        return lower_go(source)
    raise LoweringError(f"the complexity engine does not handle {language!r}")
