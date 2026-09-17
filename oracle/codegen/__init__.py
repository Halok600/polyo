"""Driver codegen: one test-spec + one language -> one runnable measurement
script (plan §6). Templates are declarative (`codegen/<lang>.j2`) so that
adding a language is a new template, not new Python logic here -- Phase 2
ships `python.j2` only; `cpp.j2`, `java.j2`, `javascript.j2`, `go.j2` land in
Phase 4.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from oracle.spec import TestSpec

_TEMPLATE_DIR = Path(__file__).parent
_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)

_TEMPLATE_BY_LANGUAGE: dict[str, str] = {
    "python": "python.j2",
}


class UnsupportedOracleLanguageError(ValueError):
    """No codegen template for this language yet (plan §14 phases these in)."""


def render_driver(language: str, spec: TestSpec, solution_path: Path) -> str:
    try:
        template_name = _TEMPLATE_BY_LANGUAGE[language]
    except KeyError as e:
        raise UnsupportedOracleLanguageError(
            f"no oracle codegen template for language: {language!r}"
        ) from e

    params = [
        {
            "type": p.type.value,
            "has_size": p.size == spec.n_grid.variable,
            "dist": p.dist.value,
            "const": p.const,
        }
        for p in spec.params
    ]
    template = _ENV.get_template(template_name)
    return template.render(
        solution_path=str(solution_path),
        entrypoint=spec.entrypoint,
        params=params,
        n_values=spec.n_grid.values(),
        budget_ms=spec.budget_ms,
        warmup_iters=2,
        measure_repeats=7,
        seed=0,
    )
