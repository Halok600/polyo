"""Golden-suite scorecard: the numbers that can actually see the failures users hit.

The old headline was a single pooled macro-F1 over a test set that was 96.6% Python (nine C++
rows), so a model that over-predicted every non-Python nested loop by one or two ranks still
"won". This report never pools that way. For each predictor it prints:

  * accuracy per LANGUAGE and per EXPECTED CLASS (a never-predicted class shows as 0% recall),
  * mean ordinal rank error (a one-rank miss is not a four-rank miss),
  * cross-language CONSISTENCY (the same algorithm in five languages must get one answer),
  * perturbation INVARIANCE (dead code, renames, comments, reformatting, an unused helper and
    function reordering must not move the answer),
  * known-hard cases SEPARATELY (tagged `known-limitation`; never part of a gate).

    python -m eval.scorecard --predictors rule,gnn --out eval/SCORECARD.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.taxonomy import SpaceClass, TimeClass, space_rank, time_rank
from eval.golden_suite import ALL_LANGUAGES, Expected, GoldenCase, load_cases
from eval.perturb import apply_all
from eval.predictors import Prediction, Predictor, available_predictors, get_predictor

KNOWN_LIMITATION = "known-limitation"
_TIME_ORDER = [c.value for c in TimeClass]
_SPACE_ORDER = [c.value for c in SpaceClass]
# A missing prediction is scored as worse than any wrong one: the full size of the taxonomy.
_MISSING_TIME_PENALTY = float(len(_TIME_ORDER))
_MISSING_SPACE_PENALTY = float(len(_SPACE_ORDER))


# --------------------------------------------------------------------------------------------
# expression normalisation
# --------------------------------------------------------------------------------------------
def _split_top(text: str, separators: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in separators:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _norm_factor(factor: str) -> str:
    if factor.startswith("(") and factor.endswith(")"):
        return "(" + _norm_sum(factor[1:-1]) + ")"
    function = re.fullmatch(r"(log|sqrt)\((.*)\)", factor)
    if function:
        return f"{function.group(1)}({_norm_sum(function.group(2))})"
    return factor.replace(" ", "")


def _norm_product(term: str) -> str:
    factors = [_norm_factor(f) for f in _split_top(term, " *")]
    return "*".join(sorted(factors))


def _norm_sum(text: str) -> str:
    return "+".join(sorted(_norm_product(t) for t in _split_top(text, "+")))


def normalize_expr(text: str) -> str:
    """A canonical spelling of a complexity expression: ordering and spacing of factors and terms
    do not matter (`O(m * n)` == `O(n * m)`), but the variables, powers and logs do. Variable
    *names* are compared literally, so an asymmetric answer such as `m log n` is NOT equal to
    `n log m`."""
    inner = text.strip()
    if inner.startswith("O(") and inner.endswith(")"):
        inner = inner[2:-1]
    inner = re.sub(r"\b(log|sqrt)\s+([A-Za-z_]\w*)", r"\1(\2)", inner)
    return _norm_sum(inner)


# --------------------------------------------------------------------------------------------
# per-case results
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    group: str
    language: str
    tags: tuple[str, ...]
    has_overrides: bool
    expected: Expected
    predicted: Prediction

    @property
    def known_limitation(self) -> bool:
        return KNOWN_LIMITATION in self.tags

    @property
    def time_ok(self) -> bool:
        return self.predicted.time_class == self.expected.time_class

    @property
    def space_ok(self) -> bool:
        return self.predicted.space_class == self.expected.space_class

    @property
    def time_rank_error(self) -> float:
        if self.predicted.time_class is None:
            return _MISSING_TIME_PENALTY
        return float(
            abs(
                time_rank(TimeClass(self.predicted.time_class))
                - time_rank(TimeClass(self.expected.time_class))
            )
        )

    @property
    def space_rank_error(self) -> float:
        if self.predicted.space_class is None:
            return _MISSING_SPACE_PENALTY
        return float(
            abs(
                space_rank(SpaceClass(self.predicted.space_class))
                - space_rank(SpaceClass(self.expected.space_class))
            )
        )

    @property
    def time_expr_ok(self) -> bool | None:
        if self.predicted.time_expr is None:
            return None
        return normalize_expr(self.predicted.time_expr) == normalize_expr(self.expected.time_expr)

    @property
    def space_expr_ok(self) -> bool | None:
        if self.predicted.space_expr is None:
            return None
        return normalize_expr(self.predicted.space_expr) == normalize_expr(self.expected.space_expr)


def score_predictor(
    predictor: Predictor,
    cases: Iterable[GoldenCase],
    languages: Sequence[str] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        for language in case.languages:
            if languages is not None and language not in languages:
                continue
            results.append(
                CaseResult(
                    case_id=case.id,
                    group=case.group,
                    language=language,
                    tags=case.tags,
                    has_overrides=bool(case.overrides),
                    expected=case.expected(language),
                    predicted=predictor(case.sources[language], language),
                )
            )
    return results


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Slice:
    n: int
    time_accuracy: float
    space_accuracy: float
    mean_time_rank_error: float
    mean_space_rank_error: float
    time_expr_accuracy: float | None = None
    space_expr_accuracy: float | None = None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_mean(flags: Sequence[bool | None]) -> float | None:
    known = [f for f in flags if f is not None]
    return _mean([1.0 if f else 0.0 for f in known]) if known else None


def _slice(results: Sequence[CaseResult]) -> Slice:
    return Slice(
        n=len(results),
        time_accuracy=_mean([1.0 if r.time_ok else 0.0 for r in results]),
        space_accuracy=_mean([1.0 if r.space_ok else 0.0 for r in results]),
        mean_time_rank_error=_mean([r.time_rank_error for r in results]),
        mean_space_rank_error=_mean([r.space_rank_error for r in results]),
        time_expr_accuracy=_optional_mean([r.time_expr_ok for r in results]),
        space_expr_accuracy=_optional_mean([r.space_expr_ok for r in results]),
    )


@dataclass(frozen=True, slots=True)
class Summary:
    overall: Slice
    known_limitations: Slice
    per_language: dict[str, Slice]
    per_time_class: dict[str, Slice]
    per_space_class: dict[str, Slice]
    per_group: dict[str, Slice]
    cross_language_consistency: float
    cross_language_correct: float
    cross_language_cases: int
    errors: int
    failing: tuple[tuple[str, str], ...] = field(default=())  # gated (case_id, language) misses


def summarize(results: Sequence[CaseResult], cases: Sequence[GoldenCase]) -> Summary:
    gated = [r for r in results if not r.known_limitation]
    hard = [r for r in results if r.known_limitation]

    def group_by(key) -> dict:  # noqa: ANN001, ANN202
        buckets: dict[str, list[CaseResult]] = defaultdict(list)
        for r in gated:
            buckets[key(r)].append(r)
        return buckets

    language_buckets = group_by(lambda r: r.language)
    per_language = {
        lang: _slice(language_buckets[lang]) for lang in ALL_LANGUAGES if lang in language_buckets
    }
    time_buckets = group_by(lambda r: r.expected.time_class)
    space_buckets = group_by(lambda r: r.expected.space_class)
    per_time_class = {c: _slice(time_buckets[c]) for c in _TIME_ORDER if c in time_buckets}
    per_space_class = {c: _slice(space_buckets[c]) for c in _SPACE_ORDER if c in space_buckets}
    group_buckets = group_by(lambda r: r.group)
    per_group = {g: _slice(rs) for g, rs in sorted(group_buckets.items())}

    by_case: dict[str, list[CaseResult]] = defaultdict(list)
    for r in gated:
        if not r.has_overrides:
            by_case[r.case_id].append(r)
    comparable = {cid: rs for cid, rs in by_case.items() if len(rs) >= 2}
    consistent = sum(
        1
        for rs in comparable.values()
        if all(r.predicted.error is None for r in rs)
        and len({(r.predicted.time_class, r.predicted.space_class) for r in rs}) == 1
    )
    correct = sum(1 for rs in comparable.values() if all(r.time_ok and r.space_ok for r in rs))
    n_comparable = len(comparable)

    return Summary(
        overall=_slice(gated),
        known_limitations=_slice(hard),
        per_language=per_language,
        per_time_class=per_time_class,
        per_space_class=per_space_class,
        per_group=per_group,
        cross_language_consistency=consistent / n_comparable if n_comparable else 0.0,
        cross_language_correct=correct / n_comparable if n_comparable else 0.0,
        cross_language_cases=n_comparable,
        errors=sum(1 for r in results if r.predicted.error is not None),
        failing=tuple((r.case_id, r.language) for r in gated if not (r.time_ok and r.space_ok)),
    )


# --------------------------------------------------------------------------------------------
# perturbation invariance
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InvarianceStats:
    n: int
    unchanged: int

    @property
    def rate(self) -> float:
        return self.unchanged / self.n if self.n else 1.0


@dataclass(frozen=True, slots=True)
class InvarianceReport:
    overall: InvarianceStats
    by_perturbation: dict[str, InvarianceStats]
    by_language: dict[str, InvarianceStats]
    flips: tuple[tuple[str, str, str], ...] = field(default=())  # (case_id, language, perturbation)

    @property
    def overall_rate(self) -> float:
        return self.overall.rate


def invariance_report(
    predictor: Predictor,
    cases: Iterable[GoldenCase],
    languages: Sequence[str] | None = None,
) -> InvarianceReport:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    lang_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    flips: list[tuple[str, str, str]] = []
    total = [0, 0]
    for case in cases:
        if KNOWN_LIMITATION in case.tags:
            continue
        for language in case.languages:
            if languages is not None and language not in languages:
                continue
            source = case.sources[language]
            base = predictor(source, language)
            if base.error is not None:
                continue
            for name, variant in apply_all(source, language):
                after = predictor(variant, language)
                same = (
                    after.error is None
                    and (after.time_class, after.space_class) == (base.time_class, base.space_class)
                )
                for bucket in (counts[name], lang_counts[language], total):
                    bucket[0] += 1
                    bucket[1] += 1 if same else 0
                if not same:
                    flips.append((case.id, language, name))
    return InvarianceReport(
        overall=InvarianceStats(*total),
        by_perturbation={k: InvarianceStats(*v) for k, v in sorted(counts.items())},
        by_language={k: InvarianceStats(*v) for k, v in lang_counts.items()},
        flips=tuple(flips),
    )


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt_invariance(stats: InvarianceStats | None) -> str:
    return "-" if stats is None else f"{_pct(stats.rate)} ({stats.unchanged}/{stats.n})"


def _headline(summaries: dict[str, Summary]) -> list[str]:
    header = [
        "predictor", "n", "time acc", "space acc", "time rank err", "space rank err",
        "time expr", "space expr", "errors",
    ]  # fmt: skip
    rows = []
    for name, s in summaries.items():
        o = s.overall
        rows.append(
            [
                name,
                str(o.n),
                _pct(o.time_accuracy),
                _pct(o.space_accuracy),
                f"{o.mean_time_rank_error:.2f}",
                f"{o.mean_space_rank_error:.2f}",
                _pct(o.time_expr_accuracy),
                _pct(o.space_expr_accuracy),
                str(s.errors),
            ]
        )
    title = "## Headline (gated cases only; known-hard cases are reported separately)"
    return [title, "", _table(header, rows), ""]


def _per_language(summaries: dict[str, Summary]) -> list[str]:
    names = list(summaries)
    rows = []
    for lang in ALL_LANGUAGES:
        slices = [summaries[n].per_language.get(lang) for n in names]
        present = [sl for sl in slices if sl is not None]
        if not present:
            continue
        cells = [
            "-" if sl is None else f"{_pct(sl.time_accuracy)} / {_pct(sl.space_accuracy)}"
            for sl in slices
        ]
        rows.append([lang, str(present[0].n), *cells])
    return [
        "## Per language (time accuracy / space accuracy)",
        "",
        _table(["language", "n", *names], rows),
        "",
    ]


def _per_class(
    summaries: dict[str, Summary], title: str, attr: str, order: list[str], metric: str
) -> list[str]:
    names = list(summaries)
    rows = []
    for cls in order:
        slices = [getattr(summaries[n], attr).get(cls) for n in names]
        present = [sl for sl in slices if sl is not None]
        if not present:
            continue
        cells = ["-" if sl is None else _pct(getattr(sl, metric)) for sl in slices]
        rows.append([cls, str(present[0].n), *cells])
    return [f"## {title}", "", _table(["expected class", "n", *names], rows), ""]


def _cross_language(summaries: dict[str, Summary]) -> list[str]:
    intro = (
        "The same algorithm written in every language must get ONE answer. *Consistent* = every "
        "language got the same (time, space) answer, right or wrong; *all correct* = every "
        "language matched its label. Cases with deliberate per-language overrides are excluded."
    )
    rows = [
        [
            name,
            str(s.cross_language_cases),
            _pct(s.cross_language_consistency),
            _pct(s.cross_language_correct),
        ]
        for name, s in summaries.items()
    ]
    table = _table(["predictor", "cases", "consistent", "all correct"], rows)
    return ["## Cross-language consistency", "", intro, "", table, ""]


def _invariance_section(
    summaries: dict[str, Summary], invariances: dict[str, InvarianceReport]
) -> list[str]:
    names = list(summaries)
    intro = (
        "Share of semantics-preserving rewrites (dead O(1) statements, renamed locals, comments, "
        "re-indentation, an unused O(1) helper, reordered functions) after which the predicted "
        "(time, space) classes did not change."
    )
    kinds = sorted({k for rep in invariances.values() for k in rep.by_perturbation})
    rows = []
    for kind in kinds:
        cells = [
            _fmt_invariance(invariances[n].by_perturbation.get(kind)) if n in invariances else "-"
            for n in names
        ]
        rows.append([kind, *cells])
    overall = [_fmt_invariance(invariances[n].overall) if n in invariances else "-" for n in names]
    rows.append(["**overall**", *overall])
    return ["## Perturbation invariance", "", intro, "", _table(["perturbation", *names], rows), ""]


def _known_hard(summaries: dict[str, Summary]) -> list[str]:
    rows = [
        [
            name,
            str(s.known_limitations.n),
            _pct(s.known_limitations.time_accuracy),
            _pct(s.known_limitations.space_accuracy),
        ]
        for name, s in summaries.items()
    ]
    title = "## Known-hard cases (tagged `known-limitation`; reported, never gated)"
    return [title, "", _table(["predictor", "n", "time acc", "space acc"], rows), ""]


def _per_group(summaries: dict[str, Summary]) -> list[str]:
    names = list(summaries)
    groups = sorted({g for n in names for g in summaries[n].per_group})
    rows = []
    for group in groups:
        cells = []
        for n in names:
            sl = summaries[n].per_group.get(group)
            cells.append("-" if sl is None else _pct(sl.time_accuracy))
        rows.append([group, *cells])
    return ["## Per group (time accuracy)", "", _table(["group", *names], rows), ""]


def render_markdown(
    summaries: dict[str, Summary],
    invariances: dict[str, InvarianceReport] | None = None,
    title: str = "PolyO golden-suite scorecard",
    preamble: str = "",
) -> str:
    out: list[str] = [f"# {title}", ""]
    if preamble:
        out += [preamble, ""]
    out += _headline(summaries)
    out += _per_language(summaries)
    out += _per_class(
        summaries,
        "Per time class (recall of the expected class)",
        "per_time_class",
        _TIME_ORDER,
        "time_accuracy",
    )
    out += _per_class(
        summaries,
        "Per space class (recall of the expected class)",
        "per_space_class",
        _SPACE_ORDER,
        "space_accuracy",
    )
    out += _cross_language(summaries)
    if invariances:
        out += _invariance_section(summaries, invariances)
    out += _known_hard(summaries)
    out += _per_group(summaries)
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score predictors on the golden suite.")
    parser.add_argument(
        "--predictors", default="rule,gnn", help=f"comma list of {available_predictors()}"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the Markdown report here")
    parser.add_argument("--no-invariance", action="store_true", help="skip the perturbation sweep")
    parser.add_argument("--title", default="PolyO golden-suite scorecard")
    parser.add_argument("--preamble", default="", help="paragraph printed under the title")
    args = parser.parse_args(argv)

    cases = load_cases()
    names = [n.strip() for n in args.predictors.split(",") if n.strip()]
    summaries: dict[str, Summary] = {}
    invariances: dict[str, InvarianceReport] = {}
    for name in names:
        predictor = get_predictor(name)
        print(f"scoring {name} ...", file=sys.stderr)
        summaries[name] = summarize(score_predictor(predictor, cases), cases)
        if not args.no_invariance:
            print(f"perturbing {name} ...", file=sys.stderr)
            invariances[name] = invariance_report(predictor, cases)
    text = render_markdown(
        summaries, invariances or None, title=args.title, preamble=args.preamble
    )
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
