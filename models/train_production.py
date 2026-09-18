#!/usr/bin/env python3
"""Trains and exports the models `api/` actually serves (plan §13's "model
artifacts"). Two models, two jobs:

- The multi-task GNN (rung 3 -- the best macro-F1 on the full corpus per
  PHASE5_REPORT.md) provides the served class prediction. Exported to numpy
  (`models/export_numpy.py`) so the served image never needs torch.
- The GBDT (rung 2) provides per-prediction attribution -- the GNN's numpy
  reimplementation has no gradient-based equivalent, and rung 2 was always
  plan §8's "feature engineering, interpretability" rung. Only its
  normalised global feature importance is exported as JSON, not the model
  itself: LightGBM's live per-prediction SHAP needs `lightgbm` importable
  at request time, which unconditionally imports `scipy` (~115MB
  installed) -- see `api/attribution.py`'s module docstring for that
  tradeoff in full.

Run once, locally -- the GNN needs a GPU for reasonable wall-clock (see
project memory on training practicalities). Not part of CI: no GPU there,
and these artifacts don't change unless the corpus or model does.

    python -m models.train_production
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.corpus import CorpusRecord, read_jsonl
from eval.report import SPACE_CLASSES, TIME_CLASSES, compute_metrics
from models import gbdt, gnn
from models.calibrate import fit_temperature
from models.dataset import ParsedExample, build_examples, space_labels, time_labels, with_label
from models.export_numpy import export_gnn

ARTIFACTS_DIR = Path("models/artifacts")
_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("time", TIME_CLASSES),
    ("space", SPACE_CLASSES),
)


def _load_splits(processed_dir: Path) -> dict[str, list[CorpusRecord]]:
    splits: dict[str, list[CorpusRecord]] = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"split_{name}.jsonl"
        if not path.is_file():
            print(f"error: {path} not found -- run `python -m data.build` first", file=sys.stderr)
            raise SystemExit(1)
        splits[name] = read_jsonl(path)
    return splits


def _train_and_export_gnn(
    train_ex: list[ParsedExample],
    train_time_y: list[str | None],
    train_space_y: list[str | None],
    val_ex: list[ParsedExample],
    val_time_y: list[str | None],
    val_space_y: list[str | None],
    test_ex: list[ParsedExample],
    test_time_y: list[str | None],
    test_space_y: list[str | None],
    args: argparse.Namespace,
) -> None:
    print("=== Training production GNN (multi-task, all edges) ===", flush=True)
    time_model, space_model = gnn.fit_multitask(
        train_ex,
        train_time_y,
        train_space_y,
        val_examples=val_ex,
        val_time_labels=val_time_y,
        val_space_labels=val_space_y,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=True,
    )

    calibration: dict[str, object] = {}
    for dimension, model, val_labels, test_labels in (
        ("time", time_model, val_time_y, test_time_y),
        ("space", space_model, val_space_y, test_space_y),
    ):
        val_scored_ex, val_scored_y = with_label(val_ex, val_labels)
        val_scores = model.decision_function(val_scored_ex)
        calibrator = fit_temperature(val_scores, val_scored_y, tuple(model.classes))

        test_scored_ex, test_scored_y = with_label(test_ex, test_labels)
        test_scores = model.decision_function(test_scored_ex)
        proba = calibrator.calibrate(test_scores)
        predictions = [model.classes[i] for i in proba.argmax(axis=1)]
        metrics = compute_metrics(
            dimension, test_scored_y, predictions, tuple(model.classes), proba.max(axis=1)
        )
        print(f"GNN {dimension} (calibrated, test set):")
        print(json.dumps(metrics.as_dict(), indent=2), flush=True)
        calibration[dimension] = {
            "temperature": calibrator.temperature,
            "classes": list(model.classes),
        }

    export_gnn(time_model.core, time_model.edge_kinds, ARTIFACTS_DIR / "gnn.npz")
    calibration_path = ARTIFACTS_DIR / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def _normalized_importance(model: gbdt.GbdtModel) -> dict[str, float]:
    importances = model.feature_importance()
    total = sum(importances.values())
    if total <= 0:
        return dict.fromkeys(importances, 0.0)
    return {name: value / total for name, value in importances.items()}


def _feature_scales(examples: list[ParsedExample]) -> dict[str, float]:
    """Per-feature standard deviation over the training set -- attribution
    (`api/attribution.py`) divides each request's raw feature value by this
    before weighting, so a large-magnitude, whole-program feature like
    `node_count`/`edge_count` doesn't systematically dominate a small-
    magnitude, span-bearing one like `max_loop_nesting_depth` just because
    its raw numbers are bigger. Floored well above zero so an
    almost-always-zero feature (e.g. `binary_search_call_count`) doesn't
    get an artificially huge normalised value the one time it's nonzero."""
    from features.tabular import TabularFeatures

    names = tuple(TabularFeatures(0, 0).as_dict().keys())
    columns: dict[str, list[float]] = {name: [] for name in names}
    for example in examples:
        row = example.features.as_dict()
        for name in names:
            columns[name].append(row[name])

    scales: dict[str, float] = {}
    for name, values in columns.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        scales[name] = max(variance**0.5, 0.5)
    return scales


def _train_and_export_gbdt(
    train_ex: list[ParsedExample],
    train_time_y: list[str | None],
    train_space_y: list[str | None],
    test_ex: list[ParsedExample],
    test_time_y: list[str | None],
    test_space_y: list[str | None],
) -> None:
    """Trains rung 2 for attribution only (plan §8's "feature engineering,
    interpretability" rung) -- not saved as a LightGBM model file. Only
    its normalised global feature importance (plus each feature's training-
    set scale, see `_feature_scales`) is exported as JSON, which is what
    `api/attribution.py` needs and nothing more (see that module's
    docstring for why lightgbm itself never reaches `api/`)."""
    print("=== Training production GBDT (attribution only) ===", flush=True)
    weights_by_dimension: dict[str, dict[str, float]] = {}
    for dimension, classes, train_labels, test_labels in (
        ("time", TIME_CLASSES, train_time_y, test_time_y),
        ("space", SPACE_CLASSES, train_space_y, test_space_y),
    ):
        tr_ex, tr_y = with_label(train_ex, train_labels)
        te_ex, te_y = with_label(test_ex, test_labels)
        model = gbdt.fit(tr_ex, tr_y)
        predictions = model.predict(te_ex)
        metrics = compute_metrics(dimension, te_y, predictions, classes)
        print(f"GBDT {dimension} (test set):")
        print(json.dumps(metrics.as_dict(), indent=2), flush=True)

        weights_by_dimension[dimension] = _normalized_importance(model)

    scales = _feature_scales(train_ex)
    importance_path = ARTIFACTS_DIR / "feature_importance.json"
    importance_path.write_text(
        json.dumps({"weights": weights_by_dimension, "scales": scales}, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    splits = _load_splits(args.processed_dir)
    examples_by_split: dict[str, list[ParsedExample]] = {}
    for name, records in splits.items():
        examples, stats = build_examples(records)
        examples_by_split[name] = examples
        print(f"{name}: {json.dumps(stats.as_dict())}", flush=True)
    train_ex = examples_by_split["train"]
    val_ex = examples_by_split["val"]
    test_ex = examples_by_split["test"]

    train_time_y, train_space_y = time_labels(train_ex), space_labels(train_ex)
    val_time_y, val_space_y = time_labels(val_ex), space_labels(val_ex)
    test_time_y, test_space_y = time_labels(test_ex), space_labels(test_ex)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _train_and_export_gnn(
        train_ex,
        train_time_y,
        train_space_y,
        val_ex,
        val_time_y,
        val_space_y,
        test_ex,
        test_time_y,
        test_space_y,
        args,
    )
    _train_and_export_gbdt(
        train_ex, train_time_y, train_space_y, test_ex, test_time_y, test_space_y
    )

    print(f"wrote artifacts to {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
