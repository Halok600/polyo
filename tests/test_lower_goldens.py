"""The lowerers against the whole golden suite.

Per-construct tests cover what I thought of; this covers what the 759 real sources contain. Three
properties, all of which a lowering bug would break silently downstream:
  * every source lowers (no crash) into a module that has code in it;
  * nothing is `Unknown` -- an unlowered construct would otherwise become an invisible O(1);
  * the same algorithm has the same loop-nest depth in every language (the engine-side version of
    the IR premise that the original bug report turned out to be innocent of).
"""
from __future__ import annotations

import pytest

from analysis.lower import lower_source
from analysis.nodes import Loop, Module, Unknown, walk
from eval.golden_suite import GoldenCase, load_cases

_CASES = load_cases()
_PARAMS = [
    pytest.param(case, language, id=f"{case.id}-{language}")
    for case in _CASES
    for language in case.languages
]
_STRUCTURAL_GROUPS = {"basics", "sublinear", "constant"}


def _nodes(module: Module):  # noqa: ANN202
    for func in module.functions:
        yield from walk(func)
    for stmt in module.toplevel:
        yield from walk(stmt)


def _max_loop_depth(module: Module) -> int:
    def depth(node: object) -> int:
        from analysis.nodes import children

        below = max((depth(c) for c in children(node)), default=0)
        return below + (1 if isinstance(node, Loop) else 0)

    return max(
        [depth(f) for f in module.functions] + [depth(s) for s in module.toplevel], default=0
    )


@pytest.mark.parametrize(("case", "language"), _PARAMS)
def test_every_golden_source_lowers_with_code_and_no_unknowns(
    case: GoldenCase, language: str
) -> None:
    module = lower_source(case.sources[language], language)
    assert module.functions or module.toplevel, "lowered to an empty module"
    unknown = sorted({n.text for n in _nodes(module) if isinstance(n, Unknown)})
    assert not unknown, f"unlowered constructs: {unknown}"


@pytest.mark.parametrize(
    "case",
    [c for c in _CASES if c.group in _STRUCTURAL_GROUPS and "known-limitation" not in c.tags],
    ids=lambda c: c.id,
)
def test_loop_nest_depth_agrees_across_languages(case: GoldenCase) -> None:
    depths = {
        lang: _max_loop_depth(lower_source(src, lang)) for lang, src in case.sources.items()
    }
    assert len(set(depths.values())) == 1, depths
