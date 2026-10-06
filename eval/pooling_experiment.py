"""Why does dead code move the GNN's answer, and does a cheap fix move it back?

Plan section 1.2 measured it: padding a correct two-sum with 40 straight-line statements flips the
served GNN from O(n^2) to O(n log n). The plan's hypothesis was "sum pooling or node count leaking".
The readout in `models/gnn.py` is a MEAN over node embeddings, so it cannot grow with the node
count;
what it does do is DILUTE: a loop among forty straight-line nodes is a fortieth of the average. This
experiment tests that reading and two fixes on a small, cleaned, class-capped slice of the corpus,
four variants trained with the same seed and settings:

    mean            the served architecture
    meanmax         mean and per-feature max of the node embeddings (a strong node survives)
    mean+aug        mean, trained on dead-code-padded copies of the examples as well
    meanmax+aug     both

and scored on what matters: accuracy on the blind corpus (independent labels, written by agents that
never saw any model) and the share of blind answers that survive 40 dead statements.

    python -m eval.pooling_experiment --rows 12000 --epochs 8 --out eval/POOLING_EXPERIMENT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {
    "mean": ("mean", False),
    "meanmax": ("meanmax", False),
    "mean+aug": ("mean", True),
    "meanmax+aug": ("meanmax", True),
}
PAD_STATEMENTS = (10, 40)


def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def training_slice(rows: int, per_problem: int) -> list:
    """The cleaned, capped slice: a cap per problem, IR near-duplicates collapsed, every non-Python
    record kept, Python filling up to `rows` (lowest hash first, so it is deterministic)."""
    from data.build import cap_per_problem, dedupe_by_ir
    from data.corpus import read_jsonl

    records = read_jsonl(ROOT / "data" / "processed" / "split_train.jsonl")
    records = dedupe_by_ir(cap_per_problem(records, per_problem))
    other = [r for r in records if r.language != "python"]
    python = sorted(
        (r for r in records if r.language == "python"),
        key=lambda r: _bucket(f"{r.source}:{r.solution_id}"),
    )
    return other + python[: max(0, rows - len(other))]


def validation_slice(count: int) -> list:
    from data.corpus import read_jsonl

    records = read_jsonl(ROOT / "data" / "processed" / "split_val.jsonl")
    records.sort(key=lambda r: _bucket(f"{r.source}:{r.solution_id}"))
    return records[:count]


def padded(records: list, statements: int) -> list:
    from models.augment import pad_with_dead_code

    return pad_with_dead_code(records, statements)


def blind_records() -> list:
    from data.corpus import CorpusRecord

    cases = json.loads((ROOT / "tests" / "data" / "blind_v1_cases.json").read_text("utf-8"))
    return [
        CorpusRecord(
            case["id"],
            case["id"],
            "blind",
            case["language"],
            case["source"],
            case["time_class"],
            case["space_class"],
        )
        for case in cases
    ]


def run(variant: str, train: list, val: list, epochs: int, blind: list) -> dict[str, Any]:
    from models import gnn
    from models.dataset import build_examples, space_labels, time_labels

    pooling, augment = VARIANTS[variant]
    started = time.time()
    records = list(train)
    if augment:
        records += padded(train[: len(train) // 2], PAD_STATEMENTS[0])
        records += padded(train[len(train) // 2 :], PAD_STATEMENTS[1])
    train_examples, _ = build_examples(records)
    val_examples, _ = build_examples(val)
    time_model, space_model = gnn.fit_multitask(
        train_examples,
        time_labels(train_examples),
        space_labels(train_examples),
        val_examples=val_examples,
        val_time_labels=time_labels(val_examples),
        val_space_labels=space_labels(val_examples),
        hidden_dim=64,
        num_layers=3,
        batch_size=256,
        max_epochs=epochs,
        patience=3,
        pooling=pooling,
    )
    blind_examples, _ = build_examples(blind)
    answers = list(
        zip(time_model.predict(blind_examples), space_model.predict(blind_examples), strict=True)
    )
    labels = [(e.record.time_class, e.record.space_class) for e in blind_examples]
    time_hits = sum(a[0] == label[0] for a, label in zip(answers, labels, strict=True))
    space_hits = sum(a[1] == label[1] for a, label in zip(answers, labels, strict=True))
    survive: dict[int, tuple[int, int]] = {}
    for statements in PAD_STATEMENTS:
        padded_records = padded(blind, statements)
        padded_examples, _ = build_examples(padded_records)
        by_id = {e.record.solution_id.split("+pad")[0]: e for e in padded_examples}
        same = total = 0
        for example, answer in zip(blind_examples, answers, strict=True):
            twin = by_id.get(example.record.solution_id)
            if twin is None:
                continue
            total += 1
            same += (
                time_model.predict([twin])[0] == answer[0]
                and space_model.predict([twin])[0] == answer[1]
            )
        survive[statements] = (same, total)
    return {
        "variant": variant,
        "train_records": len(train_examples),
        "blind_n": len(blind_examples),
        "blind_time_correct": time_hits,
        "blind_space_correct": space_hits,
        "unchanged_after_padding": {str(k): list(v) for k, v in survive.items()},
        "seconds": round(time.time() - started),
    }


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / max(1, total):.0%})"


def render(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# GNN readout: dilution, not size",
        "",
        f"Cleaned training slice (cap {args.per_problem} per problem, IR near-duplicates "
        f"collapsed, every non-Python record kept, Python filling up to {args.rows}), "
        f"{args.epochs} epochs, hidden 64, 3 layers, seed 42; scored on the blind corpus "
        "(independent labels).",
        "",
        "| variant | train rows | blind time | blind space | unchanged by 10 dead statements "
        "| unchanged by 40 | seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        n = r["blind_n"]
        unchanged = r["unchanged_after_padding"]
        survival = [_share(*unchanged[str(k)]) for k in PAD_STATEMENTS]
        lines.append(
            f"| {r['variant']} | {r['train_records']} | {_share(r['blind_time_correct'], n)} "
            f"| {_share(r['blind_space_correct'], n)} | {survival[0]} | {survival[1]} "
            f"| {r['seconds']} |"
        )
    lines += [
        "",
        "For comparison the symbolic engine answers 100% of these padded programs identically (the",
        "perturbation suite) and is right on 85% (time) / 88% (space) of the blind corpus at first",
        "contact.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--per-problem", type=int, default=25)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    train = training_slice(args.rows, args.per_problem)
    val = validation_slice(args.val)
    blind = blind_records()
    print(f"train {len(train)} val {len(val)} blind {len(blind)}", flush=True)
    results = []
    for variant in args.variants.split(","):
        print(f"== {variant}", flush=True)
        result = run(variant, train, val, args.epochs, blind)
        print(json.dumps(result), flush=True)
        results.append(result)
    report = render(results, args)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8"
        )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
