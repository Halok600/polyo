#!/usr/bin/env python3
"""Trains rungs 0-2, evaluates on held-out test data, and writes the model
card (plan §14 Phase 3: "Model card v1 with first confusion matrices").

Pipeline per dimension (time, space): fit rung 1 (TF-IDF+logreg) and rung 2
(IR features+LightGBM) on `data/build.py`'s train split, temperature-scale
each on the val split, then score everything -- including rung 0's fixed
rule, which needs no fitting -- on the held-out test split. Only the test
split's numbers go in the model card; val is spent entirely on calibration,
never scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.corpus import CorpusRecord, read_jsonl
from eval.report import SPACE_CLASSES, TIME_CLASSES, Metrics, compute_metrics, plot_confusion_matrix
from models import gbdt, rule, tfidf
from models.calibrate import Calibrator, fit_temperature
from models.dataset import ParsedExample, build_examples, space_labels, time_labels, with_label

FIGURES_DIR = Path("eval/figures")
MODEL_CARD_PATH = Path("MODEL_CARD.md")

_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("time", TIME_CLASSES),
    ("space", SPACE_CLASSES),
)


@dataclass(frozen=True, slots=True)
class RungResult:
    metrics: Metrics
    temperature: float | None = None
    feature_importance: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class DimensionResult:
    rungs: dict[str, RungResult]
    test_languages: list[str]


def _rule_predictions(examples: list[ParsedExample], dimension: str) -> list[str]:
    if dimension == "time":
        return [rule.predict_time(e.features).value for e in examples]
    return [rule.predict_space(e.features).value for e in examples]


def _drop_labels_unseen_in_training(
    examples: list[ParsedExample], labels: list[str], known_classes: list[str]
) -> tuple[list[ParsedExample], list[str]]:
    """A rare class can end up with zero training examples after a
    problem-level split (plan §9) -- a model can never predict a class it
    never saw, so calibration on those examples is undefined. Test-set
    scoring doesn't need this: `compute_metrics`'s confusion matrix is built
    over the *full* taxonomy regardless of what the model actually saw."""
    known = set(known_classes)
    kept_ex = [e for e, y in zip(examples, labels, strict=True) if y in known]
    kept_y = [y for y in labels if y in known]
    return kept_ex, kept_y


def _score_with_calibration(
    decision_scores: np.ndarray, classes: list[str], calibrator: Calibrator
) -> tuple[list[str], np.ndarray]:
    proba = calibrator.calibrate(decision_scores)
    pred_idx = proba.argmax(axis=1)
    predictions = [classes[i] for i in pred_idx]
    confidences = proba.max(axis=1)
    return predictions, confidences


def _evaluate_dimension(
    dimension: str,
    classes: tuple[str, ...],
    train: list[ParsedExample],
    val: list[ParsedExample],
    test: list[ParsedExample],
) -> DimensionResult:
    label_fn = time_labels if dimension == "time" else space_labels
    train_ex, train_y = with_label(train, label_fn(train))
    val_ex, val_y = with_label(val, label_fn(val))
    test_ex, test_y = with_label(test, label_fn(test))

    rungs: dict[str, RungResult] = {}

    # Rung 0: fixed rule, no fitting, no probability output -- no ECE.
    rule_pred = _rule_predictions(test_ex, dimension)
    rungs["rung0_rule"] = RungResult(metrics=compute_metrics(dimension, test_y, rule_pred, classes))

    # Rung 1: TF-IDF + logistic regression.
    tfidf_model = tfidf.fit(train_ex, train_y)
    tfidf_classes = tfidf_model.classes
    tfidf_val_ex, tfidf_val_y = _drop_labels_unseen_in_training(val_ex, val_y, tfidf_classes)
    tfidf_val_scores = np.asarray(tfidf_model.decision_function(tfidf_val_ex))
    tfidf_calibrator = fit_temperature(tfidf_val_scores, tfidf_val_y, tuple(tfidf_classes))
    tfidf_test_scores = np.asarray(tfidf_model.decision_function(test_ex))
    tfidf_pred, tfidf_conf = _score_with_calibration(
        tfidf_test_scores, tfidf_classes, tfidf_calibrator
    )
    rungs["rung1_tfidf_logreg"] = RungResult(
        metrics=compute_metrics(dimension, test_y, tfidf_pred, classes, tfidf_conf),
        temperature=tfidf_calibrator.temperature,
    )

    # Rung 2: IR features + LightGBM.
    gbdt_model = gbdt.fit(train_ex, train_y)
    gbdt_classes = gbdt_model.classes
    gbdt_val_ex, gbdt_val_y = _drop_labels_unseen_in_training(val_ex, val_y, gbdt_classes)
    gbdt_val_scores = np.asarray(gbdt_model.decision_function(gbdt_val_ex))
    gbdt_calibrator = fit_temperature(gbdt_val_scores, gbdt_val_y, tuple(gbdt_classes))
    gbdt_test_scores = np.asarray(gbdt_model.decision_function(test_ex))
    gbdt_pred, gbdt_conf = _score_with_calibration(gbdt_test_scores, gbdt_classes, gbdt_calibrator)
    rungs["rung2_gbdt"] = RungResult(
        metrics=compute_metrics(dimension, test_y, gbdt_pred, classes, gbdt_conf),
        feature_importance=gbdt_model.feature_importance(),
    )

    return DimensionResult(rungs=rungs, test_languages=[e.record.language for e in test_ex])


def _metrics_table(rows: dict[str, RungResult]) -> str:
    header = "| Rung | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |\n"
    header += "|---|---|---|---|---|---|\n"
    lines = [header]
    for name, result in rows.items():
        metrics = result.metrics
        ece = f"{metrics.ece:.3f}" if metrics.ece is not None else "n/a"
        lines.append(
            f"| {name} | {metrics.n} | {metrics.accuracy:.3f} | {metrics.macro_f1:.3f} | "
            f"{metrics.mean_ordinal_distance:.3f} | {ece} |\n"
        )
    return "".join(lines)


def _write_model_card(
    corpus_report: dict[str, object],
    dataset_stats: dict[str, object],
    results_by_dimension: dict[str, DimensionResult],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sections = [
        "# PolyO model card v1\n",
        "Phase 3 of 7 (plan §14). Rungs 0-2 (rule -> TF-IDF+logistic regression -> "
        "IR features+LightGBM), evaluated on a held-out, problem-level test split "
        "-- see plan §9: solutions to the same problem never cross a split "
        "boundary, and `tests/test_data_splits.py` enforces it in CI.\n",
        "\n## Corpus\n",
        "Python-only in this phase (plan §7's BigO(Bench) + CodeComplex ingestion, "
        "`data/ingest_bigobench.py` / `data/ingest_codecomplex.py`); C++ parses "
        "(Phase 1) but has no labelled corpus yet, so it isn't in this evaluation. "
        "Split sizes (problem-level, `data/build.py`):\n\n",
        "```json\n" + json.dumps(corpus_report, indent=2) + "\n```\n",
        "\nParsing/feature-extraction survival rate on the held-out test split "
        "(`models/dataset.py`) -- a real, arbitrary competitive-programming corpus "
        "occasionally trips a walker Phase 1's small hand-written examples never "
        "did:\n\n",
        "```json\n" + json.dumps(dataset_stats, indent=2) + "\n```\n",
    ]

    for dimension, dim_result in results_by_dimension.items():
        rungs = dim_result.rungs
        sections.append(f"\n## {dimension.title()} complexity\n\n")
        sections.append(_metrics_table(rungs))

        # Best by macro-F1, not accuracy (plan §9: never bare accuracy) --
        # rung0's fixed rule can never predict several classes at all
        # (e.g. O(log n)), so it can win on accuracy while losing badly on
        # macro-F1; ranking by accuracy would call that rung "best".
        learned_rungs = {name: r for name, r in rungs.items() if name != "rung0_rule"}
        best_rung_name = max(learned_rungs, key=lambda k: learned_rungs[k].metrics.macro_f1)
        img_path = FIGURES_DIR / f"confusion_{dimension}_{best_rung_name}.png"
        plot_confusion_matrix(
            rungs[best_rung_name].metrics,
            f"{dimension.title()} complexity -- {best_rung_name} (test set)",
            img_path,
        )
        sections.append(f"\n![{dimension} confusion matrix]({img_path.as_posix()})\n")

        rule_img_path = FIGURES_DIR / f"confusion_{dimension}_rung0_rule.png"
        plot_confusion_matrix(
            rungs["rung0_rule"].metrics,
            f"{dimension.title()} complexity -- rung0_rule baseline (test set)",
            rule_img_path,
        )
        sections.append(f"\n![{dimension} confusion matrix, rung 0]({rule_img_path.as_posix()})\n")

        importances = rungs["rung2_gbdt"].feature_importance
        if importances:
            top = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
            sections.append(f"\n### Top rung-2 features ({dimension})\n\n")
            sections.append("| feature | importance |\n|---|---|\n")
            for name, value in top:
                sections.append(f"| {name} | {value:.1f} |\n")

        sections.append(f"\n### Reading these numbers ({dimension})\n\n")
        rule_m, tfidf_m, gbdt_m = (
            rungs["rung0_rule"].metrics,
            rungs["rung1_tfidf_logreg"].metrics,
            rungs["rung2_gbdt"].metrics,
        )
        if rule_m.accuracy > max(tfidf_m.accuracy, gbdt_m.accuracy):
            sections.append(
                f"- rung0_rule has the *highest accuracy* ({rule_m.accuracy:.3f}) but the "
                f"*lowest macro-F1* ({rule_m.macro_f1:.3f} vs. "
                f"{max(tfidf_m.macro_f1, gbdt_m.macro_f1):.3f}) of the three -- it can only ever "
                "predict the classes its loop/alloc-depth lookup table covers "
                f"(`models/rule.py`), so its F1 on every class outside that table is exactly "
                "0.0, which is invisible in accuracy but not in macro-F1 (plan §9: never bare "
                "accuracy -- this is the concrete case that guards against).\n"
            )
        if gbdt_m.macro_f1 < tfidf_m.macro_f1:
            sections.append(
                f"- rung1 (TF-IDF over the full IR symbol sequence) currently beats rung2 "
                f"(macro-F1 {tfidf_m.macro_f1:.3f} vs. {gbdt_m.macro_f1:.3f}) -- rung2's feature "
                "set deliberately excludes loop-bound shape and hash/set-lookup signals (see "
                "Known limitations below), which rung1's raw symbol n-grams still capture "
                "implicitly. Closing that feature gap is the most direct next step, not "
                "switching models.\n"
            )

    sections.append(
        "\n## Known limitations (see `oracle/README.md`, `data/README.md`, "
        "`features/tabular.py` for full detail)\n\n"
        "- BigO(Bench) labels are per-input-variable; only single-variable labels "
        "map to this taxonomy, and that's a real, measured drop rate, not a guess.\n"
        "- Rung-2 features deliberately don't include loop-bound *shape* "
        "(const/input-dependent/halving) or hash/set-lookup detection -- both need "
        "static-analysis work beyond Phase 1's node-to-symbol IR mapping, or type "
        "inference this project's static design doesn't attempt.\n"
        "- Space labels come from BigO(Bench) only (CodeComplex is time-only), so "
        "the space model trains on a smaller, less problem-diverse slice than time.\n"
    )

    MODEL_CARD_PATH.write_text("".join(sections), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    splits: dict[str, list[CorpusRecord]] = {}
    corpus_report: dict[str, object] = {}
    for name in ("train", "val", "test"):
        path = args.processed_dir / f"split_{name}.jsonl"
        if not path.is_file():
            print(f"error: {path} not found -- run `python -m data.build` first", file=sys.stderr)
            return 1
        records = read_jsonl(path)
        splits[name] = records
        corpus_report[name] = {"records": len(records)}

    examples_by_split: dict[str, list[ParsedExample]] = {}
    dataset_stats: dict[str, object] = {}
    for name, records in splits.items():
        examples, stats = build_examples(records)
        examples_by_split[name] = examples
        dataset_stats[name] = stats.as_dict()

    results_by_dimension: dict[str, DimensionResult] = {}
    for dimension, classes in _DIMENSIONS:
        results_by_dimension[dimension] = _evaluate_dimension(
            dimension, classes, examples_by_split["train"], examples_by_split["val"],
            examples_by_split["test"],
        )
        print(
            json.dumps(
                {
                    name: rung.metrics.as_dict()
                    for name, rung in results_by_dimension[dimension].rungs.items()
                },
                indent=2,
            )
        )

    _write_model_card(corpus_report, dataset_stats, results_by_dimension)
    print(f"wrote {MODEL_CARD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
