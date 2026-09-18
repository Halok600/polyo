# PolyO

**Static, multi-language time & space complexity prediction — no LLM calls,
no code execution at inference time.**

Paste code in Python, C++, Java, JavaScript, C, or Go and get its predicted
worst-case **time and space** complexity: a class, a calibrated confidence,
the code spans that drove the prediction, and a growth chart — computed by a
model that only ever *reads* the code.

**Live demo:** _pending deployment — see [Status](#status) below._

## Why this exists

Two public benchmarks already label code with time complexity
([CodeComplex](https://arxiv.org/abs/2401.08719)) or with time *and* space
complexity ([BigO(Bench)](https://arxiv.org/abs/2503.15242), Meta). Both are
Python/Java-only, and BigO(Bench)'s own inference framework has to *execute*
the code to label it. Nobody has shipped a **static**, **multi-language**
predictor of both — small enough to run for free, on a task where
[BigO(Bench) found frontier LLMs themselves struggle](https://arxiv.org/abs/2503.15242).

The approach: normalise every language into one small intermediate
representation (IR) via [tree-sitter](https://tree-sitter.github.io/), so one
feature extractor and one model serve every language — and use dynamic
profiling **only offline**, to auto-label training data, never in the
deployed path.

## Results

Full numbers, ablations, and the zero-shot cross-language transfer
experiment are in [`PHASE5_REPORT.md`](PHASE5_REPORT.md); rungs 0-2's own
baseline comparison is in [`MODEL_CARD.md`](MODEL_CARD.md). Headline:

- A **GNN message-passing over the IR graph** (rung 3) beats a rule baseline,
  TF-IDF over IR symbols, and IR-features+LightGBM on the same held-out,
  problem-level test split — **0.377 / 0.336 macro-F1** (time / space) on the
  full multi-language corpus.
- **Zero-shot cross-language transfer**: trained on Python+Java only,
  evaluated on C++, JavaScript, Go, and C it never saw during training.
  Directionally consistent with a language-agnostic IR, but honestly
  caveated: `data/scrape.py` now exists (40 real, MIT-licensed solutions
  from `github.com/TheAlgorithms`, hand-labelled by algorithm identity —
  see its module docstring), but the reported numbers below predate it and
  still reflect the smaller synthetic-only non-Python/Java test cells
  (n=5-7) — **illustrative, not yet a statistically robust result**.
  Retraining on the expanded corpus is a deliberate next step, not done
  here. Python and Java, the languages with real corpus-scale test data,
  transfer as expected. [`PHASE5_REPORT.md`](PHASE5_REPORT.md) reports the
  per-language numbers plainly rather than only the aggregate.
- Calibrated: temperature-scaled confidence, ECE reported, not just accuracy
  (never bare accuracy — a fixed-rule baseline can *win* on accuracy while
  losing badly on macro-F1, and the model card shows exactly that failure
  mode as a concrete, measured example, not a hypothetical).
- Failure-bucket analysis names *where* the model is weakest (complexity
  hidden inside a library call is the single largest failure mode) instead
  of stopping at an aggregate score.
- **LLM zero-shot baseline**: [BigO(Bench) found frontier LLMs themselves
  struggle](https://arxiv.org/abs/2503.15242) at this task; PolyO now has
  its own number rather than only citing that. Roughly comparable on time
  (0.365 vs. 0.377 macro-F1), clearly better on space (0.265 vs. 0.336) —
  and it gets there with a model that runs in milliseconds, has no
  per-request LLM call or API cost, and doesn't depend on a third party's
  model being available. [`PHASE5_REPORT.md`](PHASE5_REPORT.md) has the
  full methodology and reads the numbers honestly, including where they
  don't tell the cleanest possible story.

## Architecture

```
OFFLINE (labels training data; never deployed)      ONLINE (free tier; executes nothing)
  BigO(Bench) + CodeComplex                           Next.js (Vercel) ──▶ FastAPI (Render)
  + parallel synthetic generator                        paste box            tree-sitter → IR
        │                                                growth chart        → features → GNN
        ▼                                                                    (numpy, no torch)
  oracle: codegen → run at n → BIC curve fit
        │
        ▼
  train: rule → TF-IDF → LightGBM → GNN
```

The served model is a graph neural network trained in PyTorch, but the
*served image has no torch at all* — its forward pass is reimplemented in
pure numpy (`models/export_numpy.py`), verified numerically identical to
the trained model within `1e-4`. Per-prediction attribution similarly
avoids a ~115MB `scipy` dependency (LightGBM's live SHAP requires it) in
favour of a precomputed, scale-normalised feature-importance heuristic —
both real, measured tradeoffs made explicit in the code, not silently
eaten by the free-tier image-size budget.

**Tier 3 languages** (TypeScript, Rust, C#, Kotlin) parse and normalise
through the exact same IR mapping mechanism — proving "adding a language is
a config file, not a rewrite" — but aren't wired into the live API yet
(deliberately deferred, see `api/predict.py`).

## License

Code is MIT (`LICENSE`). Some training data carries its own, more
restrictive license — see `NOTICE.md` before reusing model weights
commercially.

## Status

All seven build phases are code-complete and CI-verified (see commit
history and `MODEL_CARD.md`/`PHASE5_REPORT.md` for what each phase actually
shipped, with real numbers, not just a checklist). What's left before this
has a live URL is account-level, not code-level: see
[`DEPLOY.md`](DEPLOY.md) for the exact steps (Render for the API, Vercel
for the frontend, both free-tier, both already Dockerized/configured).
