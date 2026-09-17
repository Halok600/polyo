# models -- the rung ladder (plan §8)

`rule.py` (rung 0, Phase 1), `dataset.py` (shared corpus -> IR -> features
pipeline, Phase 3), `tfidf.py` (rung 1: TF-IDF over IR symbol n-grams +
logistic regression, Phase 3), `gbdt.py` (rung 2: IR features + LightGBM,
Phase 3), `calibrate.py` (temperature scaling + ECE, Phase 3) all ship.

`gnn.py` (rung 3, message-passing over the IR graph) and `export_numpy.py`
(torch-free serving weights) land in Phase 5. See `MODEL_CARD.md` at the
repo root for what rungs 0-2 actually score, and `eval/model_card.py` for
how it's produced.
