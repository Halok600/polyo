# eval -- metrics and the model card (plan §9)

`report.py` (accuracy, macro-F1, mean ordinal distance, ECE, confusion
matrices, per-language breakdowns -- never bare accuracy alone) and
`model_card.py` (trains rungs 0-2 on the real corpus, evaluates on the
held-out problem-level test split, writes `MODEL_CARD.md` at the repo root
plus `eval/figures/*.png`) both ship as of Phase 3.

`transfer.py` (zero-shot cross-language transfer heatmap) and
`benchmark.py` (CodeComplex/BigO(Bench) external comparison) land in Phase 5,
once more than one language has a labelled corpus.
