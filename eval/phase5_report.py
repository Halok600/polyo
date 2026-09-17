#!/usr/bin/env python3
"""Trains rung 3 (GNN), runs Phase 5's ablations and the zero-shot
cross-language transfer experiment, analyses failure buckets, and writes
PHASE5_REPORT.md (plan §14 Phase 5: "Full results table + transfer heatmap.
Best post of the sprint").

Rung 0-2 numbers are not recomputed here -- MODEL_CARD.md (`eval/model_card.py`)
already has them on the same split; this report is only what's new in Phase 5.
Real GNN training needs a GPU to run in reasonable time (this dev machine's
RTX 3050, see project memory) -- there is no GPU in CI, so this script is
never invoked from `.github/workflows/ci.yml`, only run locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from data.corpus import CorpusRecord, read_jsonl
from eval.ablations import (
    AblationResult,
    edge_type_ablation,
    feature_family_ablation,
    ir_symbols_vs_raw_tokens_ablation,
    multitask_vs_singletask_ablation,
)
from eval.failure_buckets import BUCKETS, FailureBucketReport, analyze_failures
from eval.report import SPACE_CLASSES, TIME_CLASSES, Metrics, compute_metrics, plot_confusion_matrix
from eval.transfer import plot_transfer_heatmap, run_transfer_experiment
from models import gnn
from models.dataset import ParsedExample, build_examples, space_labels, time_labels, with_label

FIGURES_DIR = Path("eval/figures")
REPORT_PATH = Path("PHASE5_REPORT.md")

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


def _metrics_table(rows: list[tuple[str, Metrics]]) -> str:
    header = "| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |\n"
    header += "|---|---|---|---|---|---|\n"
    lines = [header]
    for name, m in rows:
        ece = f"{m.ece:.3f}" if m.ece is not None else "n/a"
        lines.append(
            f"| {name} | {m.n} | {m.accuracy:.3f} | {m.macro_f1:.3f} | "
            f"{m.mean_ordinal_distance:.3f} | {ece} |\n"
        )
    return "".join(lines)


def _ablation_table(results: list[AblationResult]) -> str:
    return _metrics_table([(r.name, r.metrics) for r in results])


def _result_by_name(results: list[AblationResult], name: str) -> AblationResult:
    return next(r for r in results if r.name == name)


def _reading_multitask_vs_singletask(results: list[AblationResult]) -> str:
    multi = _result_by_name(results, "multi_task")
    single = _result_by_name(results, "single_task")
    winner, delta = (
        ("multi_task", multi.metrics.macro_f1 - single.metrics.macro_f1)
        if multi.metrics.macro_f1 >= single.metrics.macro_f1
        else ("single_task", single.metrics.macro_f1 - multi.metrics.macro_f1)
    )
    return (
        f"- **{winner}** wins on macro-F1 here (+{delta:.3f}). Either answer is a "
        "result (plan §9) -- a win for single-task doesn't undermine the shared "
        "encoder's value elsewhere (e.g. the transfer experiment still needs one "
        "encoder that has seen both dimensions' structure); it says joint training's "
        "regularisation effect didn't outweigh head-competition for gradient "
        "capacity on this split.\n"
    )


def _reading_edge_types(results: list[AblationResult]) -> str:
    best = max(results, key=lambda r: r.metrics.macro_f1)
    all_edges = _result_by_name(results, "all_edges")
    if best.name == "all_edges":
        return "- **all_edges** is the best-performing configuration here.\n"
    return (
        f"- **{best.name}** (macro-F1 {best.metrics.macro_f1:.3f}) beats "
        f"**all_edges** (macro-F1 {all_edges.metrics.macro_f1:.3f}) at this "
        "ablation's shared, capped epoch budget -- read as \"the extra edge "
        "kinds don't stack additively at this training budget\" rather than "
        "\"more structure hurts\": rung 3's own headline number above used a "
        "larger epoch budget than any ablation variant did, precisely because "
        "ablations must share one fixed, fair budget across configurations.\n"
    )


def _reading_ir_vs_raw_tokens(results: list[AblationResult]) -> str:
    ir = _result_by_name(results, "ir_symbol_tfidf")
    raw = _result_by_name(results, "raw_token_tfidf")
    winner = "raw_token_tfidf" if raw.metrics.macro_f1 > ir.metrics.macro_f1 else "ir_symbol_tfidf"
    return (
        f"- **{winner}** wins on this split's macro-F1 "
        f"(ir_symbol_tfidf {ir.metrics.macro_f1:.3f} vs. raw_token_tfidf "
        f"{raw.metrics.macro_f1:.3f}). That is a real result, not swept under the "
        "rug -- but it is not the whole test of the project's thesis (plan §2): "
        "this comparison trains and evaluates within the same language mix, "
        "where raw tokens can pick up incidental lexical cues (identifier/library "
        "naming patterns correlated with a source corpus) that plain accuracy "
        "can't distinguish from real structural signal. Raw tokens have no path "
        "to cross-language transfer at all -- Python's `for x in xs` and Java's "
        "`for (int x : xs)` share almost no raw tokens -- which is what "
        "the transfer experiment below actually tests, and where the IR's "
        "language-agnostic design is the only one of the two that can work by "
        "construction.\n"
    )


def _reading_feature_families(results: list[AblationResult]) -> str:
    full = _result_by_name(results, "full")
    families = [r for r in results if r.name != "full"]
    worst = min(families, key=lambda r: r.metrics.macro_f1)
    drop = full.metrics.macro_f1 - worst.metrics.macro_f1
    family_name = worst.name.removeprefix("without_")
    return (
        f"- Removing **{family_name}** costs the most macro-F1 ({drop:+.3f} from "
        f"the full-feature {full.metrics.macro_f1:.3f}) -- worth cross-checking "
        "against the failure buckets below: if `hidden_in_library_call` is a "
        "large bucket there too, that's the same signal showing up twice, not "
        "two unrelated findings.\n"
    )


def _failure_bucket_section(dimension: str, report: FailureBucketReport) -> list[str]:
    lines = [
        f"\n### Failure buckets ({dimension}, rung3_gnn's own misclassifications)\n\n",
        f"{report.total_misclassified} misclassified examples out of the "
        f"{dimension} test set. Buckets overlap (an example can match more than "
        "one heuristic) rather than being forced into exactly one:\n\n",
        "| Bucket | Count | Share of misclassified | Description |\n|---|---|---|---|\n",
    ]
    for name, description in BUCKETS.items():
        count = report.counts[name]
        share = count / report.total_misclassified if report.total_misclassified else 0.0
        lines.append(f"| {name} | {count} | {share:.1%} | {description} |\n")
    return lines


def _write_report(
    dataset_stats: dict[str, object],
    rung3_metrics: dict[str, Metrics],
    mt_vs_st: dict[str, list[AblationResult]],
    edge_ablation: dict[str, list[AblationResult]],
    ir_vs_raw: dict[str, list[AblationResult]],
    feature_families: dict[str, list[AblationResult]],
    transfer_results: dict[str, dict[str, Metrics]],
    heatmap_path: Path,
    failure_reports: dict[str, FailureBucketReport],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sections = [
        "# PolyO Phase 5 report\n",
        "Phase 5 of 7 (plan §14): rung 3 (GNN message-passing over the IR graph, "
        "shared encoder -> two heads), its ablations, the zero-shot "
        "cross-language transfer experiment, and failure-bucket analysis. See "
        "`MODEL_CARD.md` for rungs 0-2 on the same held-out, problem-level test "
        "split.\n",
        "\nParsing/feature-extraction survival rate per split (`models/dataset.py`):\n\n",
        "```json\n" + json.dumps(dataset_stats, indent=2) + "\n```\n",
    ]

    sections.append("\n## Rung 3: multi-task GNN (all edge kinds)\n\n")
    sections.append(_metrics_table([(dim, rung3_metrics[dim]) for dim in ("time", "space")]))
    for dimension in ("time", "space"):
        img_path = FIGURES_DIR / f"confusion_{dimension}_rung3_gnn.png"
        sections.append(f"\n![{dimension} confusion matrix, rung 3]({img_path.as_posix()})\n")

    sections.append(
        "\n## Ablation: multi-task vs. single-task\n\n"
        "Shared encoder + two heads, trained jointly, vs. two independent "
        "single-head models (plan §9: \"does joint training beat two "
        "independent models? Either answer is a result.\").\n\n"
    )
    for dimension in ("time", "space"):
        sections.append(f"\n**{dimension.title()}**\n\n")
        sections.append(_ablation_table(mt_vs_st[dimension]))
        sections.append(_reading_multitask_vs_singletask(mt_vs_st[dimension]))

    sections.append(
        "\n## Ablation: graph edge types\n\n"
        "Multi-task GNN retrained per edge-kind subset -- which edges beyond "
        "the plain AST (`AST_CHILD`/`NEXT_SIBLING`) actually earn their keep.\n\n"
    )
    for dimension in ("time", "space"):
        sections.append(f"\n**{dimension.title()}**\n\n")
        sections.append(_ablation_table(edge_ablation[dimension]))
        sections.append(_reading_edge_types(edge_ablation[dimension]))

    sections.append(
        "\n## Ablation: IR symbols vs. raw source tokens\n\n"
        "Same TF-IDF + logistic regression pipeline, differing only in "
        "whether the input text is the normalised IR symbol sequence (rung 1) "
        "or raw source code -- this is the ablation that most directly tests "
        "the project's central thesis (plan §2).\n\n"
    )
    for dimension in ("time", "space"):
        sections.append(f"\n**{dimension.title()}**\n\n")
        sections.append(_ablation_table(ir_vs_raw[dimension]))
        sections.append(_reading_ir_vs_raw_tokens(ir_vs_raw[dimension]))

    sections.append(
        "\n## Ablation: feature families (rung 2)\n\n"
        "Leave-one-family-out over rung 2's tabular feature set "
        "(`eval/ablations.py`'s `FEATURE_FAMILIES`).\n\n"
    )
    for dimension in ("time", "space"):
        sections.append(f"\n**{dimension.title()}**\n\n")
        sections.append(_ablation_table(feature_families[dimension]))
        sections.append(_reading_feature_families(feature_families[dimension]))

    sections.append(
        "\n## Zero-shot cross-language transfer\n\n"
        "Trained on Python+Java only; every other language is unseen during "
        "training. **Known limitation, not hidden:** the corpus's only "
        "non-Python/Java code comes from the 84-record parallel synthetic "
        "generator (`data/synth.py`) -- `data/scrape.py` (plan §7's scraped "
        "real-code source) was never built. After the problem-level split, "
        "the C++/JavaScript/Go/C test cells have on the order of 5 examples "
        "each (see the `n` printed on each heatmap cell), so those specific "
        "numbers are illustrative, not statistically robust. Python and Java "
        "cells are in-distribution and have the corpus's full test-set size "
        "behind them.\n\n"
    )
    sections.append(f"\n![cross-language transfer heatmap]({heatmap_path.as_posix()})\n")
    for dimension, by_language in transfer_results.items():
        sections.append(f"\n**{dimension.title()}, per language**\n\n")
        sections.append(_metrics_table(sorted(by_language.items())))

    sections.append(
        "\n## Failure buckets\n\n"
        "Computed on rung 3's own misclassifications on the held-out test "
        "split (plan §9's five named buckets, each a structural heuristic "
        "over the IR except the last, which is defined on the true/predicted "
        "label pair directly -- see `eval/failure_buckets.py`).\n"
    )
    for dimension in ("time", "space"):
        sections.extend(_failure_bucket_section(dimension, failure_reports[dimension]))

    REPORT_PATH.write_text("".join(sections), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--ablation-max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--ablation-patience", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    splits = _load_splits(args.processed_dir)
    examples_by_split: dict[str, list[ParsedExample]] = {}
    dataset_stats: dict[str, object] = {}
    for name, records in splits.items():
        examples, stats = build_examples(records)
        examples_by_split[name] = examples
        dataset_stats[name] = stats.as_dict()
    train_ex = examples_by_split["train"]
    val_ex = examples_by_split["val"]
    test_ex = examples_by_split["test"]

    train_time_y, train_space_y = time_labels(train_ex), space_labels(train_ex)
    val_time_y, val_space_y = time_labels(val_ex), space_labels(val_ex)
    test_time_y, test_space_y = time_labels(test_ex), space_labels(test_ex)

    filtered_train = {
        "time": with_label(train_ex, train_time_y),
        "space": with_label(train_ex, train_space_y),
    }
    filtered_test = {
        "time": with_label(test_ex, test_time_y),
        "space": with_label(test_ex, test_space_y),
    }

    main_kwargs: dict[str, Any] = dict(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience, verbose=True,
    )
    ablation_kwargs: dict[str, Any] = dict(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, batch_size=args.batch_size,
        max_epochs=args.ablation_max_epochs, patience=args.ablation_patience, verbose=True,
    )

    print("=== Rung 3: multi-task GNN, all edge kinds ===", flush=True)
    time_model, space_model = gnn.fit_multitask(
        train_ex, train_time_y, train_space_y,
        val_examples=val_ex, val_time_labels=val_time_y, val_space_labels=val_space_y,
        **main_kwargs,
    )
    rung3_metrics: dict[str, Metrics] = {}
    rung3_predictions: dict[str, list[str]] = {}
    for dimension, model in (("time", time_model), ("space", space_model)):
        ex, y = filtered_test[dimension]
        pred = model.predict(ex)
        rung3_predictions[dimension] = pred
        rung3_metrics[dimension] = compute_metrics(dimension, y, pred, tuple(model.classes))
        plot_confusion_matrix(
            rung3_metrics[dimension],
            f"{dimension.title()} complexity -- rung3_gnn (test set)",
            FIGURES_DIR / f"confusion_{dimension}_rung3_gnn.png",
        )
        print(json.dumps(rung3_metrics[dimension].as_dict(), indent=2), flush=True)

    print("=== Ablation: multi-task vs single-task ===", flush=True)
    mt_vs_st = multitask_vs_singletask_ablation(
        train_ex, train_time_y, train_space_y, test_ex, test_time_y, test_space_y,
        val_ex=val_ex, val_time_y=val_time_y, val_space_y=val_space_y, gnn_kwargs=ablation_kwargs,
    )

    print("=== Ablation: graph edge types ===", flush=True)
    edge_ablation = edge_type_ablation(
        train_ex, train_time_y, train_space_y, test_ex, test_time_y, test_space_y,
        val_ex=val_ex, val_time_y=val_time_y, val_space_y=val_space_y, gnn_kwargs=ablation_kwargs,
    )

    print("=== Ablation: IR symbols vs raw tokens ===", flush=True)
    ir_vs_raw = {}
    for dimension, classes in _DIMENSIONS:
        tr_ex, tr_y = filtered_train[dimension]
        te_ex, te_y = filtered_test[dimension]
        ir_vs_raw[dimension] = ir_symbols_vs_raw_tokens_ablation(
            dimension, classes, tr_ex, tr_y, te_ex, te_y
        )

    print("=== Ablation: feature families ===", flush=True)
    feature_families = {}
    for dimension, classes in _DIMENSIONS:
        tr_ex, tr_y = filtered_train[dimension]
        te_ex, te_y = filtered_test[dimension]
        feature_families[dimension] = feature_family_ablation(
            dimension, classes, tr_ex, tr_y, te_ex, te_y
        )

    print("=== Cross-language transfer ===", flush=True)
    transfer_results: dict[str, dict[str, Metrics]] = {}
    for dimension in ("time", "space"):
        y_train = train_time_y if dimension == "time" else train_space_y
        y_val = val_time_y if dimension == "time" else val_space_y
        y_test = test_time_y if dimension == "time" else test_space_y
        transfer_results[dimension] = run_transfer_experiment(
            dimension, train_ex, y_train, test_ex, y_test,
            val_ex=val_ex, val_y=y_val, gnn_kwargs=ablation_kwargs,
        )
    heatmap_path = FIGURES_DIR / "transfer_heatmap.png"
    plot_transfer_heatmap(transfer_results, heatmap_path)

    print("=== Failure buckets ===", flush=True)
    failure_reports = {}
    for dimension in ("time", "space"):
        ex, y = filtered_test[dimension]
        failure_reports[dimension] = analyze_failures(
            dimension, ex, y, rung3_predictions[dimension]
        )

    _write_report(
        dataset_stats, rung3_metrics, mt_vs_st, edge_ablation, ir_vs_raw,
        feature_families, transfer_results, heatmap_path, failure_reports,
    )
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
