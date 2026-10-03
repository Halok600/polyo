"""Every perturbation applied to every golden source must stay a valid, structurally identical
program. If a transform silently corrupted a source, the invariance numbers in the scorecard
would measure the corruption instead of the predictor."""
from __future__ import annotations

import ast

import pytest

from eval.golden_suite import GoldenCase, load_cases
from eval.perturb import apply_all
from parsing.parse import parse_source

_CASES = load_cases()
_PARAMS = [
    pytest.param(case, language, id=f"{case.id}-{language}")
    for case in _CASES
    for language in case.languages
]


def _leaf_tokens(source: str, language: str) -> list[tuple[str, bytes]]:
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


def _parses(source: str, language: str) -> bool:
    if language == "python":
        try:
            ast.parse(source)
        except SyntaxError:
            return False
        return True
    return not parse_source(source, language).root_node.has_error


@pytest.mark.parametrize(("case", "language"), _PARAMS)
def test_perturbed_sources_stay_valid_and_structurally_equal(
    case: GoldenCase, language: str
) -> None:
    original = case.sources[language]
    for name, perturbed in apply_all(original, language):
        assert _parses(perturbed, language), f"{name} broke {case.id} [{language}]"
        if name in ("comments", "reformat"):
            if language == "python":
                assert ast.dump(ast.parse(perturbed)) == ast.dump(ast.parse(original)), name
            else:
                assert _leaf_tokens(perturbed, language) == _leaf_tokens(original, language), name
        elif name == "rename":
            before = [t for t, _ in _leaf_tokens(original, language)]
            after = [t for t, _ in _leaf_tokens(perturbed, language)]
            assert before == after, f"rename changed token kinds in {case.id} [{language}]"


def test_every_language_has_perturbations_applied_somewhere() -> None:
    applied: dict[str, set[str]] = {}
    for case in _CASES:
        for language in case.languages:
            for name, _ in apply_all(case.sources[language], language):
                applied.setdefault(language, set()).add(name)
    for language in ("python", "cpp", "java", "javascript", "go", "c"):
        assert {"dead_code", "rename", "comments", "reformat"} <= applied[language], language
