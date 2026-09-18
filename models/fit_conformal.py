#!/usr/bin/env python3
"""Fits conformal calibration against the ALREADY-TRAINED, already-exported
served model (`models/artifacts/gnn.npz` + `calibration.json`) and writes
`models/artifacts/conformal.json`.

Deliberately standalone, not folded into `models/train_production.py`'s GNN
training run: conformal calibration only needs the model's own outputs on
the val/test splits, not a fresh training pass, and running the ALREADY-
EXPORTED pure-numpy model (models/export_numpy.py -- the same one `api/`
actually serves) over val+test is CPU-only and fast, exactly because that
numpy port exists. Re-running this after any future full retrain (a new
gnn.npz) is still correct and cheap; it does not require GPU time on its
own and should never gate on it.

    python -m models.fit_conformal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data.corpus import CorpusRecord, read_jsonl
from eval.report import SPACE_CLASSES, TIME_CLASSES
from models.calibrate import Calibrator
from models.conformal import DEFAULT_ALPHAS, evaluate_conformal, fit_conformal
from models.dataset import ParsedExample, build_examples, space_labels, time_labels, with_label
from models.export_numpy import NumpyGnnModel
from models.graph_batch import to_example_graph

ARTIFACTS_DIR = Path("models/artifacts")
_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("time", TIME_CLASSES),
    ("space", SPACE_CLASSES),
)


def _load_splits(processed_dir: Path) -> dict[str, list[CorpusRecord]]:
    splits: dict[str, list[CorpusRecord]] = {}
    for name in ("val", "test"):
        path = processed_dir / f"split_{name}.jsonl"
        if not path.is_file():
            print(f"error: {path} not found -- run `python -m data.build` first", file=sys.stderr)
            raise SystemExit(1)
        splits[name] = read_jsonl(path)
    return splits


def _numpy_scores(
    gnn: NumpyGnnModel, examples: list[ParsedExample], dimension: str
) -> np.ndarray:
    """Raw (pre-softmax) per-class scores for every example, via the
    served pure-numpy forward pass -- one example at a time (that's the
    model's own interface, see models/export_numpy.py), which is still
    fast: no GPU, no gradients, just small per-example matrix multiplies."""
    rows = []
    for example in examples:
        graph = to_example_graph(example.ir)
        rows.append(gnn.forward(graph.symbol_ids, graph.edges_by_kind)[dimension])
    return np.stack(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)

    gnn_path = args.artifacts_dir / "gnn.npz"
    calibration_path = args.artifacts_dir / "calibration.json"
    if not (gnn_path.is_file() and calibration_path.is_file()):
        print(
            f"error: {gnn_path} / {calibration_path} not found -- run "
            "`python -m models.train_production` first",
            file=sys.stderr,
        )
        return 1

    gnn = NumpyGnnModel.load(gnn_path)
    calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))

    splits = _load_splits(args.processed_dir)
    val_ex, val_stats = build_examples(splits["val"])
    test_ex, test_stats = build_examples(splits["test"])
    print(f"val: {json.dumps(val_stats.as_dict())}", flush=True)
    print(f"test: {json.dumps(test_stats.as_dict())}", flush=True)

    val_time_y, val_space_y = time_labels(val_ex), space_labels(val_ex)
    test_time_y, test_space_y = time_labels(test_ex), space_labels(test_ex)

    conformal: dict[str, object] = {}
    for dimension, val_labels, test_labels in (
        ("time", val_time_y, test_time_y),
        ("space", val_space_y, test_space_y),
    ):
        classes = tuple(calibration_raw[dimension]["classes"])
        temperature = calibration_raw[dimension]["temperature"]
        calibrator = Calibrator(temperature=temperature, classes=classes)

        val_scored_ex, val_scored_y = with_label(val_ex, val_labels)
        val_scores = _numpy_scores(gnn, val_scored_ex, dimension)
        val_proba = calibrator.calibrate(val_scores)
        conformal_calibration = fit_conformal(
            val_proba, val_scored_y, classes, alphas=DEFAULT_ALPHAS
        )

        test_scored_ex, test_scored_y = with_label(test_ex, test_labels)
        test_scores = _numpy_scores(gnn, test_scored_ex, dimension)
        test_proba = calibrator.calibrate(test_scores)

        print(f"GNN {dimension} conformal (test set, risk-coverage):", flush=True)
        for alpha in DEFAULT_ALPHAS:
            result = evaluate_conformal(conformal_calibration, test_proba, test_scored_y, alpha)
            print(json.dumps(result.as_dict(), indent=2), flush=True)

        conformal[dimension] = {
            "classes": list(classes),
            "mass_threshold_by_alpha": conformal_calibration.mass_threshold_by_alpha,
        }

    out_path = args.artifacts_dir / "conformal.json"
    out_path.write_text(json.dumps(conformal, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
