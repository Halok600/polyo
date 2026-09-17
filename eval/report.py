"""Evaluation metrics and the model card (plan §9, §14 "Model card v1").

Metrics never report bare accuracy alone (plan §9's class-imbalance risk):
accuracy, macro-F1, mean ordinal distance (classes are ordinal, not
categorical -- plan §3), ECE, a confusion matrix, and a per-class breakdown
always run together. Confusion-matrix charts use the dataviz skill's
validated sequential-blue ramp (one hue, light->dark -- never a rainbow
colormap) rather than an arbitrary matplotlib default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

from core.taxonomy import SpaceClass, TimeClass, ordinal_distance_space, ordinal_distance_time
from models.calibrate import expected_calibration_error

# dataviz skill's validated sequential-blue ramp (references/palette.md,
# steps 100->700) and chart chrome -- reused verbatim, not eyeballed.
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

TIME_CLASSES: tuple[str, ...] = tuple(c.value for c in TimeClass)
SPACE_CLASSES: tuple[str, ...] = tuple(c.value for c in SpaceClass)


def _ordinal_distance(dimension: str, a: str, b: str) -> int:
    if dimension == "time":
        return ordinal_distance_time(TimeClass(a), TimeClass(b))
    return ordinal_distance_space(SpaceClass(a), SpaceClass(b))


@dataclass(frozen=True, slots=True)
class Metrics:
    dimension: str
    accuracy: float
    macro_f1: float
    mean_ordinal_distance: float
    ece: float | None
    n: int
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    classes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "mean_ordinal_distance": round(self.mean_ordinal_distance, 4),
            "ece": round(self.ece, 4) if self.ece is not None else None,
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
        }


def compute_metrics(
    dimension: str,
    true_labels: list[str],
    predicted_labels: list[str],
    classes: tuple[str, ...],
    confidences: np.ndarray | None = None,
) -> Metrics:
    if not true_labels:
        raise ValueError("compute_metrics requires at least one example")

    correct = np.array(
        [t == p for t, p in zip(true_labels, predicted_labels, strict=True)]
    )
    accuracy = float(np.mean(correct))
    macro_f1 = float(
        f1_score(
            true_labels, predicted_labels, labels=list(classes), average="macro", zero_division=0
        )
    )
    per_class = f1_score(
        true_labels, predicted_labels, labels=list(classes), average=None, zero_division=0
    )
    distances = [
        _ordinal_distance(dimension, t, p)
        for t, p in zip(true_labels, predicted_labels, strict=True)
    ]
    ece = (
        float(expected_calibration_error(confidences, correct))
        if confidences is not None
        else None
    )

    return Metrics(
        dimension=dimension,
        accuracy=accuracy,
        macro_f1=macro_f1,
        mean_ordinal_distance=float(np.mean(distances)),
        ece=ece,
        n=len(true_labels),
        per_class_f1=dict(zip(classes, per_class.tolist(), strict=True)),
        confusion=confusion_matrix(true_labels, predicted_labels, labels=list(classes)),
        classes=classes,
    )


def per_language_metrics(
    dimension: str,
    true_labels: list[str],
    predicted_labels: list[str],
    languages: list[str],
    classes: tuple[str, ...],
) -> dict[str, Metrics]:
    by_language: dict[str, Metrics] = {}
    for language in sorted(set(languages)):
        idx = [i for i, lang in enumerate(languages) if lang == language]
        by_language[language] = compute_metrics(
            dimension,
            [true_labels[i] for i in idx],
            [predicted_labels[i] for i in idx],
            classes,
        )
    return by_language


def sequential_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("polyo_sequential_blue", SEQUENTIAL_BLUE_STEPS)


def plot_confusion_matrix(metrics: Metrics, title: str, out_path: Path) -> None:
    """Row-normalised (recall) heatmap, sequential single-hue ramp, direct
    count+percentage labels per cell -- never requires the colorbar alone to
    read a value (dataviz skill: color follows magnitude, labels carry the
    number)."""
    conf = metrics.confusion.astype(float)
    row_sums = conf.sum(axis=1, keepdims=True)
    normalized = np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums > 0)

    n = len(metrics.classes)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(normalized, cmap=sequential_blue_cmap(), vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metrics.classes, rotation=45, ha="right", color=INK_MUTED, fontsize=9)
    ax.set_yticklabels(metrics.classes, color=INK_MUTED, fontsize=9)
    ax.set_xlabel("Predicted class", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("True class", color=INK_PRIMARY, fontsize=10)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=12)

    for i in range(n):
        for j in range(n):
            count = int(conf[i, j])
            if count == 0:
                continue
            pct = normalized[i, j]
            text_color = SURFACE if pct > 0.6 else INK_PRIMARY
            ax.text(
                j, i, f"{count}\n{pct:.0%}", ha="center", va="center",
                color=text_color, fontsize=7.5,
            )

    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=GRIDLINE, length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.set_label("Row-normalised share (recall)", color=INK_MUTED, fontsize=8)
    cbar.outline.set_edgecolor(GRIDLINE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
