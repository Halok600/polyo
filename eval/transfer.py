"""Zero-shot cross-language transfer (plan §9's headline experiment): train
on {Python, Java}, evaluate on every other language present in the test
split, unseen. Reported as a language x dimension heatmap.

**Known limitation, named not hidden** (already flagged in project memory
after Phase 4/5 prep): the corpus's only non-Python/Java code is the
84-record parallel synthetic generator (`data/synth.py`) -- `data/scrape.py`
(plan §7's "scraped public solution repos" source) was never built. After
the problem-level split, the test split's C++/JavaScript/Go/C cells have on
the order of 5 examples each, not hundreds. This experiment still runs and
reports real numbers (the plan calls for it explicitly, and Phase 4/5's own
scope notes already call building a scraper "a materially larger task"),
but those specific cells are illustrative, not statistically robust --
`plot_transfer_heatmap` prints each cell's `n` directly on the chart so
nobody mistakes a 5-example cell for a well-powered one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from eval.report import (
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    SURFACE,
    Metrics,
    per_language_metrics,
    sequential_blue_cmap,
)
from models import gnn
from models.dataset import ParsedExample, with_label

TRAIN_LANGUAGES: tuple[str, ...] = ("python", "java")


def filter_by_language(
    examples: list[ParsedExample], languages: tuple[str, ...]
) -> list[ParsedExample]:
    langs = set(languages)
    return [e for e in examples if e.record.language in langs]


def run_transfer_experiment(
    dimension: str,
    train_ex: list[ParsedExample],
    train_y: list[str | None],
    test_ex: list[ParsedExample],
    test_y: list[str | None],
    *,
    val_ex: list[ParsedExample] | None = None,
    val_y: list[str | None] | None = None,
    gnn_kwargs: dict[str, Any] | None = None,
) -> dict[str, Metrics]:
    """Trains one single-task GNN on {Python, Java} only, then scores it on
    the full test split and breaks the result down per language (reusing
    `eval/report.py`'s existing `per_language_metrics`) -- Python/Java cells
    are in-distribution, every other language is genuinely zero-shot: the
    model never saw one example of it during training."""
    kwargs = gnn_kwargs or {}

    train_subset = filter_by_language(train_ex, TRAIN_LANGUAGES)
    train_subset_y = [
        y for e, y in zip(train_ex, train_y, strict=True) if e.record.language in TRAIN_LANGUAGES
    ]
    fit_ex, fit_y = with_label(train_subset, train_subset_y)

    val_fit_ex: list[ParsedExample] | None = None
    val_fit_y: list[str] | None = None
    if val_ex is not None and val_y is not None:
        val_subset = filter_by_language(val_ex, TRAIN_LANGUAGES)
        val_subset_y = [
            y for e, y in zip(val_ex, val_y, strict=True) if e.record.language in TRAIN_LANGUAGES
        ]
        val_fit_ex, val_fit_y = with_label(val_subset, val_subset_y)

    model = gnn.fit_single_task(
        fit_ex, fit_y, dimension, val_examples=val_fit_ex, val_labels=val_fit_y, **kwargs
    )

    scored_ex, scored_y = with_label(test_ex, test_y)
    predictions = model.predict(scored_ex)
    languages = [e.record.language for e in scored_ex]
    return per_language_metrics(dimension, scored_y, predictions, languages, tuple(model.classes))


def run_raw_token_transfer_experiment(
    dimension: str,
    train_ex: list[ParsedExample],
    train_y: list[str | None],
    test_ex: list[ParsedExample],
    test_y: list[str | None],
) -> dict[str, Metrics]:
    """The actual test of `eval/ablations.py`'s own rebuttal to raw tokens
    beating IR-symbol TF-IDF in-distribution ("raw tokens have no path to
    cross-language transfer at all") -- trains the same raw-source-text
    TF-IDF + LogisticRegression baseline on {Python, Java} only and scores
    it exactly like `run_transfer_experiment`, so the two are directly
    comparable per language. A vocabulary fit on Python/Java source tokens
    has no representation for C++/Go/JavaScript/C syntax at all, so this is
    expected to collapse toward chance on those languages even though it
    wins in-distribution -- this either confirms that prediction with a
    real number or, if it doesn't, is itself a result worth reporting
    rather than silently dropped for not matching the expected story.
    """
    train_subset = filter_by_language(train_ex, TRAIN_LANGUAGES)
    train_subset_y = [
        y for e, y in zip(train_ex, train_y, strict=True) if e.record.language in TRAIN_LANGUAGES
    ]
    fit_ex, fit_y = with_label(train_subset, train_subset_y)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    train_x = vectorizer.fit_transform([e.record.code for e in fit_ex])
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(train_x, fit_y)

    scored_ex, scored_y = with_label(test_ex, test_y)
    test_x = vectorizer.transform([e.record.code for e in scored_ex])
    predictions = list(classifier.predict(test_x))
    languages = [e.record.language for e in scored_ex]
    classes = tuple(classifier.classes_)
    return per_language_metrics(dimension, scored_y, predictions, languages, classes)


def plot_transfer_heatmap(
    results_by_dimension: dict[str, dict[str, Metrics]], out_path: Path
) -> None:
    """One row per dimension, one column per language, cell = macro-F1 with
    its `n` printed alongside -- reuses `eval/report.py`'s validated
    sequential-blue ramp and chart chrome verbatim rather than a fresh
    palette."""
    dimensions = list(results_by_dimension.keys())
    languages = sorted({lang for res in results_by_dimension.values() for lang in res})
    matrix = np.zeros((len(dimensions), len(languages)))
    counts = np.zeros((len(dimensions), len(languages)), dtype=int)
    for i, dim in enumerate(dimensions):
        for j, lang in enumerate(languages):
            metrics = results_by_dimension[dim].get(lang)
            if metrics is not None:
                matrix[i, j] = metrics.macro_f1
                counts[i, j] = metrics.n

    fig, ax = plt.subplots(
        figsize=(1.3 * len(languages) + 2, 1.3 * len(dimensions) + 1.5), facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    im = ax.imshow(matrix, cmap=sequential_blue_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(languages)))
    ax.set_yticks(range(len(dimensions)))
    ax.set_xticklabels(languages, rotation=45, ha="right", color=INK_MUTED, fontsize=9)
    ax.set_yticklabels([d.title() for d in dimensions], color=INK_MUTED, fontsize=9)
    ax.set_title(
        "Zero-shot cross-language transfer (macro-F1, trained on Python+Java)",
        color=INK_PRIMARY,
        fontsize=10,
        pad=12,
    )
    for i in range(len(dimensions)):
        for j in range(len(languages)):
            text_color = SURFACE if matrix[i, j] > 0.6 else INK_PRIMARY
            ax.text(
                j, i, f"{matrix[i, j]:.2f}\n(n={counts[i, j]})",
                ha="center", va="center", color=text_color, fontsize=8,
            )
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=GRIDLINE, length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.set_label("Macro-F1", color=INK_MUTED, fontsize=8)
    cbar.outline.set_edgecolor(GRIDLINE)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
