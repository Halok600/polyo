# PolyO model card: rungs 0-2
Rungs 0-2 (rule -> TF-IDF+logistic regression -> IR features+LightGBM), evaluated on a held-out, problem-level test split -- see plan §9: solutions to the same problem never cross a split boundary, and `tests/test_data_splits.py` enforces it in CI. Rung 3 (the GNN, the model actually served in production) and everything built on top of it -- ablations, zero-shot cross-language transfer, failure buckets -- are in [`PHASE5_REPORT.md`](PHASE5_REPORT.md), not here; this card is the classical-ML baseline story rungs 0-3 are compared against.

## Corpus
Multi-language (plan §7's BigO(Bench) + CodeComplex ingestion, plus `data/synth.py`'s parallel synthetic generator across all six Tier 1/2 languages) -- see `PHASE5_REPORT.md` for the exact per-language breakdown of how thin the non-Python/Java slice still is. Split sizes (problem-level, `data/build.py`):

```json
{
  "train": {
    "records": 144489
  },
  "val": {
    "records": 22548
  },
  "test": {
    "records": 26899
  }
}
```

Parsing/feature-extraction survival rate on the held-out test split (`models/dataset.py`) -- a real, arbitrary competitive-programming corpus occasionally trips a walker Phase 1's small hand-written examples never did:

```json
{
  "train": {
    "total": 144489,
    "parsed": 144489,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  },
  "val": {
    "total": 22548,
    "parsed": 22548,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  },
  "test": {
    "total": 26899,
    "parsed": 26899,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  }
}
```

## Time complexity

| Rung | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| rung0_rule | 25442 | 0.463 | 0.257 | 1.089 | n/a |
| rung1_tfidf_logreg | 25442 | 0.462 | 0.367 | 1.058 | 0.029 |
| rung2_gbdt | 25442 | 0.431 | 0.331 | 1.083 | 0.063 |

![time confusion matrix](eval/figures/confusion_time_rung1_tfidf_logreg.png)

![time confusion matrix, rung 0](eval/figures/confusion_time_rung0_rule.png)

### rung1_tfidf_logreg, per language (time)

| language | n | Accuracy | Macro-F1 | Mean ordinal distance |
|---|---|---|---|---|
| c | 1 | 1.000 | 0.143 | 0.000 |
| cpp | 5 | 0.600 | 0.286 | 0.600 |
| go | 5 | 0.600 | 0.286 | 0.600 |
| java | 867 | 0.475 | 0.440 | 1.158 |
| javascript | 5 | 0.600 | 0.286 | 0.600 |
| python | 24559 | 0.462 | 0.329 | 1.054 |

### Top rung-2 features (time)

| feature | importance |
|---|---|
| edge_count | 11311.0 |
| node_count | 10595.0 |
| call_count | 8799.0 |
| branch_count | 5364.0 |
| math_op_count | 4897.0 |
| alloc_count | 4677.0 |
| max_loop_nesting_depth | 2875.0 |
| loop_count | 2700.0 |
| loop_count_depth_0 | 2312.0 |
| break_count | 1886.0 |

### Reading these numbers (time)

- rung0_rule has the *highest accuracy* (0.463) but the *lowest macro-F1* (0.257 vs. 0.367) of the three -- it can only ever predict the classes its loop/alloc-depth lookup table covers (`models/rule.py`), so its F1 on every class outside that table is exactly 0.0, which is invisible in accuracy but not in macro-F1 (plan §9: never bare accuracy -- this is the concrete case that guards against).
- rung1 (TF-IDF over the full IR symbol sequence) currently beats rung2 (macro-F1 0.367 vs. 0.331) -- rung2's feature set deliberately excludes loop-bound shape and hash/set-lookup signals (see Known limitations below), which rung1's raw symbol n-grams still capture implicitly. Closing that feature gap is the most direct next step, not switching models.

## Space complexity

| Rung | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| rung0_rule | 24564 | 0.504 | 0.191 | 1.027 | n/a |
| rung1_tfidf_logreg | 24564 | 0.518 | 0.295 | 0.903 | 0.064 |
| rung2_gbdt | 24564 | 0.496 | 0.285 | 0.952 | 0.051 |

![space confusion matrix](eval/figures/confusion_space_rung1_tfidf_logreg.png)

![space confusion matrix, rung 0](eval/figures/confusion_space_rung0_rule.png)

### rung1_tfidf_logreg, per language (space)

| language | n | Accuracy | Macro-F1 | Mean ordinal distance |
|---|---|---|---|---|
| c | 1 | 0.000 | 0.000 | 4.000 |
| cpp | 5 | 0.600 | 0.150 | 1.000 |
| go | 5 | 0.800 | 0.178 | 0.800 |
| java | 5 | 0.600 | 0.150 | 1.000 |
| javascript | 5 | 0.600 | 0.150 | 1.000 |
| python | 24543 | 0.518 | 0.295 | 0.903 |

### Top rung-2 features (space)

| feature | importance |
|---|---|
| edge_count | 9632.0 |
| node_count | 8732.0 |
| call_count | 6368.0 |
| math_op_count | 3800.0 |
| branch_count | 3751.0 |
| alloc_count | 2825.0 |
| max_loop_nesting_depth | 1850.0 |
| loop_count | 1651.0 |
| break_count | 1458.0 |
| loop_count_depth_0 | 1174.0 |

### Reading these numbers (space)

- rung1 (TF-IDF over the full IR symbol sequence) currently beats rung2 (macro-F1 0.295 vs. 0.285) -- rung2's feature set deliberately excludes loop-bound shape and hash/set-lookup signals (see Known limitations below), which rung1's raw symbol n-grams still capture implicitly. Closing that feature gap is the most direct next step, not switching models.

## Known limitations (see `oracle/README.md`, `data/README.md`, `features/tabular.py` for full detail)

- BigO(Bench) labels are per-input-variable; only single-variable labels map to this taxonomy, and that's a real, measured drop rate, not a guess.
- Rung-2 features deliberately don't include loop-bound *shape* (const/input-dependent/halving) or hash/set-lookup detection -- both need static-analysis work beyond Phase 1's node-to-symbol IR mapping, or type inference this project's static design doesn't attempt.
- Space labels come from BigO(Bench) only (CodeComplex is time-only), so the space model trains on a smaller, less problem-diverse slice than time.
