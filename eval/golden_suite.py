"""Golden suite loader: hand-labelled complexity cases, each written in every core language.

Why this exists: the served GNN predicted O(n^3) for O(n^2) C++/Java/JavaScript code (and flipped
a correct answer after 40 dead statements) because the test set was 96.6% Python and the headline
metric was pooled macro-F1. A case here is ONE algorithm written idiomatically in all five core
languages (C where it is natural), so a cross-language disagreement is directly visible.

Layout: `eval/golden/<group>.toml`, each a `[[case]]` array of tables. Sources are TOML literal
multi-line strings (`'''...'''`), so they must not contain `'''` -- golden code avoids docstrings.

Two labels per case, deliberately separate:
  - `time_class` / `space_class`: the legacy 7-class / 5-class taxonomy (`core/taxonomy.py`), which
    the existing API, conformal sets and growth chart speak.
  - `time_expr` / `space_expr`: the verbatim symbolic expression (`O(m * n)`, `O(n log n)`,
    `O(sqrt n)`, `O(V + E)`) the v2 engine reports. A case can be right on class and wrong on
    expression, and the scorecard reports both.
"""
from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from core.taxonomy import SpaceClass, TimeClass

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Order is the scorecard's column order.
CORE_LANGUAGES: tuple[str, ...] = ("python", "cpp", "java", "javascript", "go")
OPTIONAL_LANGUAGES: tuple[str, ...] = ("c",)
ALL_LANGUAGES: tuple[str, ...] = CORE_LANGUAGES + OPTIONAL_LANGUAGES

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_OVERRIDABLE = ("time_class", "space_class", "time_expr", "space_expr")
_VALID_TIME = frozenset(c.value for c in TimeClass)
_VALID_SPACE = frozenset(c.value for c in SpaceClass)


class GoldenSuiteError(ValueError):
    """A malformed golden case. Raised, never skipped: a silently dropped case silently weakens
    the net, which is the exact failure this suite exists to prevent."""


@dataclass(frozen=True, slots=True)
class Expected:
    time_class: str
    space_class: str
    time_expr: str
    space_expr: str


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    title: str
    group: str
    tags: tuple[str, ...]
    time_class: str
    space_class: str
    time_expr: str
    space_expr: str
    rationale: str
    sources: Mapping[str, str]
    overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(lang for lang in ALL_LANGUAGES if lang in self.sources)

    def expected(self, language: str) -> Expected:
        override = self.overrides.get(language, {})
        return Expected(
            time_class=override.get("time_class", self.time_class),
            space_class=override.get("space_class", self.space_class),
            time_expr=override.get("time_expr", self.time_expr),
            space_expr=override.get("space_expr", self.space_expr),
        )


def _require_str(raw: Mapping[str, object], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GoldenSuiteError(f"{where}: missing or empty string field {key!r}")
    return value


def _check_expr(value: str, key: str, where: str) -> None:
    if not (value.startswith("O(") and value.endswith(")")):
        raise GoldenSuiteError(f"{where}: {key} must look like 'O(...)', got {value!r}")


def _check_class(value: str, valid: frozenset[str], key: str, where: str) -> None:
    if value not in valid:
        raise GoldenSuiteError(f"{where}: {key} {value!r} is not in the taxonomy")


def _parse_case(raw: Mapping[str, object], group: str, origin: str) -> GoldenCase:
    case_id = _require_str(raw, "id", origin)
    where = f"{origin} case {case_id!r}"
    if not _ID_RE.match(case_id):
        raise GoldenSuiteError(f"{where}: id must match {_ID_RE.pattern}")

    time_class = _require_str(raw, "time_class", where)
    space_class = _require_str(raw, "space_class", where)
    time_expr = _require_str(raw, "time_expr", where)
    space_expr = _require_str(raw, "space_expr", where)
    _check_class(time_class, _VALID_TIME, "time_class", where)
    _check_class(space_class, _VALID_SPACE, "space_class", where)
    _check_expr(time_expr, "time_expr", where)
    _check_expr(space_expr, "space_expr", where)

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict):
        raise GoldenSuiteError(f"{where}: missing [case.sources] table")
    sources: dict[str, str] = {}
    for language, code in sources_raw.items():
        if language not in ALL_LANGUAGES:
            raise GoldenSuiteError(f"{where}: unknown language {language!r} in sources")
        if not isinstance(code, str) or not code.strip():
            raise GoldenSuiteError(f"{where}: empty source for {language!r}")
        sources[language] = code
    missing = [lang for lang in CORE_LANGUAGES if lang not in sources]
    if missing:
        raise GoldenSuiteError(f"{where}: missing core languages {missing}")

    overrides: dict[str, dict[str, str]] = {}
    overrides_raw = raw.get("overrides", {})
    if not isinstance(overrides_raw, dict):
        raise GoldenSuiteError(f"{where}: overrides must be a table")
    for language, fields in overrides_raw.items():
        if language not in sources:
            raise GoldenSuiteError(f"{where}: override for language {language!r} with no source")
        if not isinstance(fields, dict):
            raise GoldenSuiteError(f"{where}: override {language!r} must be a table")
        clean: dict[str, str] = {}
        for key, value in fields.items():
            if key not in _OVERRIDABLE or not isinstance(value, str):
                raise GoldenSuiteError(f"{where}: bad override field {key!r} for {language!r}")
            clean[key] = value
        if "time_class" in clean:
            _check_class(clean["time_class"], _VALID_TIME, "override time_class", where)
        if "space_class" in clean:
            _check_class(clean["space_class"], _VALID_SPACE, "override space_class", where)
        overrides[language] = clean

    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
        raise GoldenSuiteError(f"{where}: tags must be a list of strings")

    return GoldenCase(
        id=case_id,
        title=_require_str(raw, "title", where),
        group=group,
        tags=tuple(tags_raw),
        time_class=time_class,
        space_class=space_class,
        time_expr=time_expr,
        space_expr=space_expr,
        rationale=_require_str(raw, "rationale", where),
        sources=sources,
        overrides=overrides,
    )


def load_cases(directory: Path = GOLDEN_DIR) -> list[GoldenCase]:
    """All cases under `directory`, files in name order, cases in file order."""
    cases: list[GoldenCase] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.toml")):
        with path.open("rb") as f:
            data = tomllib.load(f)
        for raw in data.get("case", []):
            case = _parse_case(raw, group=path.stem, origin=path.name)
            if case.id in seen:
                raise GoldenSuiteError(
                    f"duplicate case id {case.id!r} in {path.name} (first seen in {seen[case.id]})"
                )
            seen[case.id] = path.name
            cases.append(case)
    return cases
