"""The honest comparison: symbolic engine, ML model, rule baseline and the product, side by side, on
data none of them was tuned against, with a promotion rule that cannot be passed by luck.

    python -m eval.compare --out eval/COMPARISON.md --promotion eval/promotion.json

Evidence sets (independent labels, never used to tune the engine's answers recorded here):
  * blind_v1   144 programs written and labelled by agents that never saw the engine
               (`tests/data/blind_v1_cases.json`); the engine's FIRST-CONTACT answers are the ones
               scored, not what the fixed engine says today.
  * audit_v1   randomly drawn corpus programs (BigO(Bench), CodeComplex) labelled afresh by
               independent agents under the project conventions (`tests/data/corpus_audit_v1.json`);
               first-contact engine answers again.
Reported but not evidence: the golden suite (the engine was developed against it), the held-out
regression corpus (fixed against), and agreement with the raw corpus labels (they follow the
measurement conventions of their source, not worst-case static analysis: see `LABEL_AUDIT.md`).

Promotion rule (plan section 2.1, made strict): the ML model answers a slice instead of the engine
only if the lower end of its 95% Wilson interval is above the upper end of the engine's on that
slice of the evidence, with at least `MIN_ROWS` rows. Time and space are judged separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from eval.predictors import Predictor

ROOT = Path(__file__).resolve().parent.parent
MIN_ROWS = 20
Z = 1.96
PAD_SIZES = (10, 40)
INTRO = (
    "The symbolic engine, the ML model, the rule baseline and the product (engine first, model as "
    "fallback), scored on data none of them was tuned against."
)
READING = (
    "`symbolic` is the engine's first-contact answer, recorded before any fix made because of "
    "these programs, so it is a measurement. `gnn` and `rule` were never fitted to them and run "
    "now. `product` runs the engine as it is today, which was fixed against these same programs "
    "after first contact: its rows are a regression net, not a measurement."
)


# ------------------------------------------------------------------------------------- statistics
def wilson(correct: int, total: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval for a proportion: honest at small n and at 0% / 100%."""
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Tally:
    correct: int = 0
    total: int = 0

    def add(self, hit: bool) -> None:
        self.correct += bool(hit)
        self.total += 1

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.correct, self.total)


# ----------------------------------------------------------------------------------------- rows
@dataclass(frozen=True)
class Row:
    """One program, its independent labels, and what each predictor answered."""

    dataset: str
    case_id: str
    language: str
    label_time: str | None
    label_space: str | None
    answers: dict[str, tuple[str | None, str | None]]
    certainty: str = ""  # of the engine's FIRST-CONTACT answer
    tags: dict[str, str] = field(default_factory=dict)


PREDICTORS = ("symbolic", "gnn", "rule", "product")


def _hit(answer: str | None, label: str | None) -> bool:
    return answer is not None and answer == label


def tally(
    rows: Sequence[Row], predictor: str, dimension: str, key: Callable[[Row], str] | None = None
) -> dict[str, Tally]:
    """Accuracy of `predictor` on `dimension`, grouped by `key` (one group by default).
    A row without a label for the dimension is skipped; a missing answer counts as wrong."""
    out: dict[str, Tally] = defaultdict(Tally)
    index = 0 if dimension == "time" else 1
    for row in rows:
        label = row.label_time if dimension == "time" else row.label_space
        if label is None or predictor not in row.answers:
            continue
        out[key(row) if key else "all"].add(_hit(row.answers[predictor][index], label))
    return dict(out)


def promotion(rows: Sequence[Row]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per slice (language x engine certainty) and dimension, who answers: the ML model only if it
    is better than the engine beyond the noise of the sample, else the engine."""
    decisions: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in ("time", "space"):
        engine = tally(rows, "symbolic", dimension, lambda r: f"{r.language}/{r.certainty}")
        model = tally(rows, "gnn", dimension, lambda r: f"{r.language}/{r.certainty}")
        for slice_key in sorted(set(engine) | set(model)):
            e, m = engine.get(slice_key, Tally()), model.get(slice_key, Tally())
            e_low, e_high = e.interval
            m_low, m_high = m.interval
            enough = e.total >= MIN_ROWS
            promoted = enough and m_low > e_high
            decisions.setdefault(slice_key, {})[dimension] = {
                "rows": e.total,
                "engine": [e.correct, e.total, round(e_low, 3), round(e_high, 3)],
                "model": [m.correct, m.total, round(m_low, 3), round(m_high, 3)],
                "answers": "model" if promoted else "engine",
                "reason": (
                    "model's interval is above the engine's"
                    if promoted
                    else (
                        "model not better beyond the noise of the sample"
                        if enough
                        else f"fewer than {MIN_ROWS} rows: the engine stays the default"
                    )
                ),
            }
    return decisions


# ---------------------------------------------------------------------------------------- loading
def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def blind_rows() -> list[Row]:
    """The blind corpus: engine = first contact; the other predictors run now."""
    from eval.predictors import get_predictor

    cases = _load_json(ROOT / "tests" / "data" / "blind_v1_cases.json")
    first = _load_json(ROOT / "tests" / "data" / "blind_v1_first_contact.json")
    assert isinstance(cases, list) and isinstance(first, dict)
    predictors = {name: get_predictor(name) for name in ("gnn", "rule", "product")}
    rows = []
    for case in cases:
        record = first[case["id"]]
        answers: dict[str, tuple[str | None, str | None]] = {
            "symbolic": (record.get("time_class"), record.get("space_class"))
        }
        for name, predict in predictors.items():
            got = predict(case["source"], case["language"])
            answers[name] = (got.time_class, got.space_class)
        rows.append(
            Row(
                "blind_v1",
                case["id"],
                case["language"],
                case["time_class"],
                case["space_class"],
                answers,
                certainty=str(record["certainty"]),
            )
        )
    return rows


def _audit_cases() -> list[tuple[str, Any, dict[str, Any]]]:
    """The audit's random-sample programs found in the local test split with the recorded hash, as
    (id, corpus record, audit entry); empty without the local corpus (the committed file holds ids,
    hashes and labels, never the dataset's code)."""
    import hashlib

    from data.corpus import read_jsonl

    path = ROOT / "tests" / "data" / "corpus_audit_v1.json"
    split = ROOT / "data" / "processed" / "split_test.jsonl"
    if not (path.exists() and split.exists()):
        return []
    audit = _load_json(path)
    assert isinstance(audit, list)
    wanted = {
        entry["id"]: entry
        for entry in audit
        if entry["kind"] == "random" and entry["confidence"] != "low"
    }
    codes = {
        f"{r.source}:{r.solution_id}": r
        for r in read_jsonl(split)
        if f"{r.source}:{r.solution_id}" in wanted
    }
    found = []
    for case_id, entry in wanted.items():
        record = codes.get(case_id)
        if record is None:
            continue
        if hashlib.sha256(record.code.encode()).hexdigest() == entry["sha256"]:
            found.append((case_id, record, entry))
    return found


def audit_rows() -> list[Row]:
    """Randomly drawn corpus programs labelled afresh by independent agents (`_audit_cases`)."""
    from eval.predictors import get_predictor

    predictors = {name: get_predictor(name) for name in ("gnn", "rule", "product")}
    rows = []
    for case_id, record, entry in _audit_cases():
        answers = {"symbolic": (entry["engine_time"], entry["engine_space"])}
        for name, predict in predictors.items():
            got = predict(record.code, record.language)
            answers[name] = (got.time_class, got.space_class)
        rows.append(
            Row(
                "audit_v1",
                case_id,
                record.language,
                entry["agent_time"],
                entry["agent_space"],
                answers,
                certainty=str(entry["certainty"]),
                tags={"source": record.source},
            )
        )
    return rows


def padding_survival(
    cases: Sequence[tuple[str, str]],
    predictors: Mapping[str, Predictor],
    sizes: Sequence[int] = PAD_SIZES,
) -> dict[str, dict[int, tuple[int, int]]]:
    """For each predictor and padding size: how many (time, space) answers are unchanged when that
    many do-nothing statements are added to the program, out of the programs that could be padded
    and that the predictor answered to begin with. `cases` are `(source, language)` pairs."""
    from eval.perturb import add_dead_code

    before = {
        name: [predict(source, language) for source, language in cases]
        for name, predict in predictors.items()
    }
    out: dict[str, dict[int, tuple[int, int]]] = {name: {} for name in predictors}
    for size in sizes:
        padded = [add_dead_code(source, language, k=size) for source, language in cases]
        for name, predict in predictors.items():
            same = total = 0
            for (_, language), code, answer in zip(cases, padded, before[name], strict=True):
                if code is None or (answer.time_class, answer.space_class) == (None, None):
                    continue
                after = predict(code, language)
                total += 1
                same += (after.time_class, after.space_class) == (
                    answer.time_class,
                    answer.space_class,
                )
            out[name][size] = (same, total)
    return out


def blind_invariance(sizes: Sequence[int] = PAD_SIZES) -> dict[str, dict[int, tuple[int, int]]]:
    """`padding_survival` over the blind corpus, every predictor run live (the symbolic engine as it
    is today: invariance is a property of the code that ships, not of a first contact)."""
    from eval.predictors import get_predictor

    cases = _load_json(ROOT / "tests" / "data" / "blind_v1_cases.json")
    assert isinstance(cases, list)
    return padding_survival(
        [(case["source"], case["language"]) for case in cases],
        {name: get_predictor(name) for name in PREDICTORS},
        sizes,
    )


@dataclass(frozen=True, slots=True)
class ModelAnswer:
    """One dimension of one program as the served GNN alone answered it."""

    hit: bool
    confidence: float  # the calibrated probability of the class it picked
    covered: bool  # the true class is in its 90% conformal set
    set_size: int
    abstained: bool


def calibration_summary(answers: Sequence[ModelAnswer], bins: int = 10) -> dict[str, float]:
    """Accuracy against the confidence the model claimed: the mean of both, the expected
    calibration error (0 = means what it says), and what its conformal sets did."""
    n = len(answers)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
            "ece": 0.0,
            "coverage": 0.0,
            "mean_set_size": 0.0,
            "abstain_share": 0.0,
        }
    buckets: list[list[ModelAnswer]] = [[] for _ in range(bins)]
    for answer in answers:
        buckets[min(bins - 1, int(answer.confidence * bins))].append(answer)
    ece = sum(
        len(bucket)
        / n
        * abs(
            sum(a.hit for a in bucket) / len(bucket)
            - sum(a.confidence for a in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )
    return {
        "n": n,
        "accuracy": sum(a.hit for a in answers) / n,
        "mean_confidence": sum(a.confidence for a in answers) / n,
        "ece": ece,
        "coverage": sum(a.covered for a in answers) / n,
        "mean_set_size": sum(a.set_size for a in answers) / n,
        "abstain_share": sum(a.abstained for a in answers) / n,
    }


def model_answers(
    programs: Sequence[tuple[str, str, str | None, str | None]],
) -> dict[str, list[ModelAnswer]]:
    """Runs the served GNN alone (`mode="ml"`) over `(source, language, time label, space label)`
    programs; a dimension without a label contributes no row."""
    from api.predict import predict
    from eval import predictors

    registry = predictors._registry()
    out: dict[str, list[ModelAnswer]] = {"time": [], "space": []}
    for source, language, time_label, space_label in programs:
        response = predict(registry, source, language, mode="ml")
        for dimension, label in (("time", time_label), ("space", space_label)):
            if label is None:
                continue
            part = cast(dict[str, Any], response[dimension])
            out[dimension].append(
                ModelAnswer(
                    hit=part["class"] == label,
                    confidence=float(part["confidence"]),
                    covered=label in part["conformal_set"],
                    set_size=len(part["conformal_set"]),
                    abstained=bool(part["abstain"]),
                )
            )
    return out


def model_calibration() -> dict[str, dict[str, dict[str, float]]]:
    """`calibration_summary` of the served GNN on each evidence set that is available here."""
    cases = _load_json(ROOT / "tests" / "data" / "blind_v1_cases.json")
    assert isinstance(cases, list)
    programs = {
        "blind_v1": [
            (c["source"], c["language"], c["time_class"], c["space_class"]) for c in cases
        ],
        "audit_v1": [
            (record.code, record.language, entry["agent_time"], entry["agent_space"])
            for _, record, entry in _audit_cases()
        ],
    }
    return {
        dataset: {
            dimension: calibration_summary(answers)
            for dimension, answers in model_answers(batch).items()
        }
        for dataset, batch in programs.items()
        if batch
    }


# --------------------------------------------------------------------------------------- reporting
def _pct(tally_: Tally | None) -> str:
    if tally_ is None or tally_.total == 0:
        return "n/a"
    low, high = tally_.interval
    return f"{tally_.accuracy:.0%} ({tally_.correct}/{tally_.total}, {low:.0%}-{high:.0%})"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / max(1, total):.0%})"


def render(
    rows_by_dataset: dict[str, list[Row]],
    decisions: dict[str, dict[str, dict[str, Any]]],
    invariance: Mapping[str, Mapping[int, tuple[int, int]]] | None = None,
    calibration: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
) -> str:
    out = ["# Engine vs model vs rule: the honest comparison", "", INTRO, "", READING, ""]
    for dataset, rows in rows_by_dataset.items():
        if not rows:
            out += [f"## {dataset}", "", "_not available here (needs the local corpus)_", ""]
            continue
        out += [f"## {dataset} ({len(rows)} programs)", ""]
        for dimension in ("time", "space"):
            table = [
                [name, _pct(tally(rows, name, dimension).get("all"))]
                for name in PREDICTORS
                if any(name in r.answers for r in rows)
            ]
            out += [
                f"### {dimension}, exact class",
                "",
                _table(["predictor", "accuracy (95% Wilson)"], table),
                "",
            ]
        languages = sorted({r.language for r in rows})
        for dimension in ("time", "space"):
            table = []
            for language in languages:
                subset = [r for r in rows if r.language == language]
                table.append(
                    [language, str(len(subset))]
                    + [_pct(tally(subset, name, dimension).get("all")) for name in PREDICTORS]
                )
            out += [
                f"### {dimension} by language",
                "",
                _table(["language", "n", *PREDICTORS], table),
                "",
            ]
        certainties = sorted({r.certainty for r in rows if r.certainty})
        table = []
        for certainty in certainties:
            subset = [r for r in rows if r.certainty == certainty]
            table.append(
                [certainty, str(len(subset))]
                + [_pct(tally(subset, name, "time").get("all")) for name in PREDICTORS]
            )
        out += [
            "### time by the engine's first-contact certainty",
            "",
            _table(["certainty", "n", *PREDICTORS], table),
            "",
        ]
    if invariance:
        sizes = sorted({size for per_size in invariance.values() for size in per_size})
        out += [
            "## Answers unchanged by dead code (blind_v1)",
            "",
            "Programs whose time and space answers stay the same when do-nothing statements are "
            "added at the top of the first function: a predictor that reads structure keeps its "
            "answer, one that reads size does not.",
            "",
            _table(
                ["predictor", *[f"{size} statements" for size in sizes]],
                [
                    [name, *[_share(*per_size.get(size, (0, 0))) for size in sizes]]
                    for name, per_size in invariance.items()
                ],
            ),
            "",
        ]
    if calibration:
        out += [
            "## The ML model's own confidence on independent programs",
            "",
            'The served GNN run alone (`mode="ml"`): how often it is right, how sure it says it '
            "is (mean probability of the class it picked), the expected calibration error (ECE, 0 "
            "= means what it says), and its 90% conformal sets: the share that contain the true "
            "class, their mean size (of 7 time or 5 space classes) and how often it abstains (a "
            "set of four classes or more).",
            "",
            _table(
                [
                    "dataset",
                    "dimension",
                    "n",
                    "accuracy",
                    "mean confidence",
                    "ECE",
                    "set coverage (90% nominal)",
                    "mean set size",
                    "abstains",
                ],
                [
                    [
                        dataset,
                        dimension,
                        str(int(s["n"])),
                        f"{s['accuracy']:.0%}",
                        f"{s['mean_confidence']:.0%}",
                        f"{s['ece']:.3f}",
                        f"{s['coverage']:.0%}",
                        f"{s['mean_set_size']:.1f}",
                        f"{s['abstain_share']:.0%}",
                    ]
                    for dataset, per_dimension in calibration.items()
                    for dimension, s in per_dimension.items()
                ],
            ),
            "",
        ]
    out += ["## Promotion (who answers each slice)", ""]
    table = []
    for slice_key, per_dimension in decisions.items():
        for dimension, decision in per_dimension.items():
            table.append(
                [
                    slice_key,
                    dimension,
                    str(decision["rows"]),
                    f"{decision['engine'][0]}/{decision['engine'][1]}",
                    f"{decision['model'][0]}/{decision['model'][1]}",
                    str(decision["answers"]),
                    str(decision["reason"]),
                ]
            )
    out += [_table(["slice", "dimension", "rows", "engine", "model", "answers", "why"], table), ""]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--promotion", type=Path, default=None)
    args = parser.parse_args(argv)
    rows_by_dataset = {"blind_v1": blind_rows(), "audit_v1": audit_rows()}
    evidence = [row for rows in rows_by_dataset.values() for row in rows]
    decisions = promotion(evidence)
    report = render(rows_by_dataset, decisions, blind_invariance(), model_calibration())
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    if args.promotion is not None:
        args.promotion.write_text(
            json.dumps(decisions, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
