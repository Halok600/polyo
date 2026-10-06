"""The production trainer's options (PolyO v2 phase 6), exercised on a tiny synthetic corpus.

Kept CPU-fast like `test_gnn.py`: a few dozen programs, one epoch. The real run is local and
GPU-backed (`models/train_production.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.corpus import CorpusRecord, write_jsonl
from models import train_production
from models.export_numpy import NumpyGnnModel

_TEMPLATES = (
    ("def {name}(xs):\n    return len(xs) + {n}\n", "O(1)", "O(1)"),
    (
        "def {name}(xs):\n    total = 0\n    for x in xs:\n        total += x + {n}\n"
        "    return total\n",
        "O(n)",
        "O(1)",
    ),
    (
        "def {name}(xs):\n    out = []\n    for x in xs:\n        out.append(x + {n})\n"
        "    return out\n",
        "O(n)",
        "O(n)",
    ),
    (
        "def {name}(xs):\n    total = 0\n    for x in xs:\n        for y in xs:\n"
        "            total += x * y + {n}\n    return total\n",
        "O(n^2)",
        "O(1)",
    ),
)
_HIDDEN = 16


def _corpus(directory: Path, per_template: dict[str, int]) -> None:
    for split, count in per_template.items():
        records = [
            CorpusRecord(
                f"{split}{i}",
                f"{split}{i}",
                "synthetic",
                "python",
                template.format(name=f"f{index}_{i}", n=i),
                time_label,
                space_label,
            )
            for index, (template, time_label, space_label) in enumerate(_TEMPLATES)
            for i in range(count)
        ]
        write_jsonl(records, directory / f"split_{split}.jsonl")


def _train(tmp_path: Path, *flags: str) -> Path:
    corpus, artifacts = tmp_path / "corpus", tmp_path / "artifacts"
    _corpus(corpus, {"train": 12, "val": 6, "test": 6})
    argv = [
        "--processed-dir", str(corpus),
        "--artifacts-dir", str(artifacts),
        "--skip-gbdt",
        "--max-epochs", "1",
        "--hidden-dim", str(_HIDDEN),
        "--num-layers", "2",
        "--batch-size", "16",
        *flags,
    ]  # fmt: skip
    assert train_production.main(argv) == 0
    return artifacts


def test_the_default_run_writes_a_mean_pooled_model_where_it_is_told(tmp_path) -> None:
    artifacts = _train(tmp_path)
    model = NumpyGnnModel.load(artifacts / "gnn.npz")
    assert model.pooling == "mean"
    assert model.weights["heads.time.weight"].shape[1] == _HIDDEN
    for name in ("calibration.json", "conformal.json"):
        assert json.loads((artifacts / name).read_text("utf-8")).keys() == {"time", "space"}
    assert not (artifacts / "feature_importance.json").exists(), "--skip-gbdt leaves it alone"


def test_a_max_pooled_model_can_be_trained_and_is_served_as_such(tmp_path) -> None:
    artifacts = _train(tmp_path, "--pooling", "meanmax")
    model = NumpyGnnModel.load(artifacts / "gnn.npz")
    assert model.pooling == "meanmax"
    # the head reads the mean and the max of the node embeddings: twice the hidden width
    assert model.weights["heads.time.weight"].shape[1] == 2 * _HIDDEN
    assert np.isfinite(model.weights["heads.time.weight"]).all()


def test_dead_code_augmentation_adds_a_padded_copy_of_every_training_program(
    tmp_path, capsys
) -> None:
    _train(tmp_path, "--augment-dead-code")
    assert "augmented training set: 48 -> 96 records" in capsys.readouterr().out
