"""Driver codegen: one test-spec + one language -> one runnable measurement
script (plan §6). Templates are declarative (`codegen/<lang>.j2`) so that
adding a language is a new template, not new Python logic here -- Phase 2
shipped `python.j2`; Phase 4 adds `cpp.j2`, `java.j2`, `javascript.j2`,
`c.j2`, `go.j2`.
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
    "javascript": "javascript.j2",
    "go": "go.j2",
    "c": "c.j2",
    "cpp": "cpp.j2",
    "java": "java.j2",
}

# Java and JavaScript run on JIT'd VMs (HotSpot, V8) where the first
# thousand-ish calls are interpreted, not compiled -- timing them cold
# measures the interpreter, not the algorithm (plan §6: "JIT warmup is not
# optional"). This is a ONE-TIME cost paid once before the whole n-sweep,
# at the grid's smallest n (cheap, and JIT compilation doesn't need a
# realistic input size to trigger) -- not per-n, which would multiply
# thousands of iterations by however many points the sweep has, and not at
# the sweep's own (potentially huge) n, which made warmup itself the
# bottleneck (a real bug hit while building this: 3000 iterations of an
# O(n log n) sort at n=2^17 alone exceeded the driver's 30s budget). Python
# has no JIT; C/C++/Go are ahead-of-time compiled with no warm-up phase at
# all, so they get none of this (0).
_JIT_WARMUP_ITERS_BY_LANGUAGE: dict[str, int] = {
    "java": 3000,
    "javascript": 3000,
}
_DEFAULT_JIT_WARMUP_ITERS = 0
# Small, per-n warmup (every language, every n) -- just enough to fault in
# pages/populate branch predictors for THAT input size, not to reach a JIT's
# steady state (see _JIT_WARMUP_ITERS_BY_LANGUAGE above for that).
_PER_N_WARMUP_ITERS = 2


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
        warmup_iters=_PER_N_WARMUP_ITERS,
        jit_warmup_iters=_JIT_WARMUP_ITERS_BY_LANGUAGE.get(language, _DEFAULT_JIT_WARMUP_ITERS),
        measure_repeats=7,
        seed=0,
    )
