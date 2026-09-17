# PolyO

**Static, multi-language time & space complexity prediction — no LLM calls,
no code execution at inference time.**

Paste code in Python, C++, Java, JavaScript, C, or Go and get its predicted
worst-case **time and space** complexity: a class, a calibrated confidence,
the code spans that drove the prediction, and a growth chart — computed by a
model that only ever *reads* the code.

> 🚧 **Status: Phase 3 of 7.** Rungs 0–2 (rule → TF-IDF+logistic regression →
> IR features+LightGBM) are trained and evaluated on a real, problem-level
> split of the labelled Python corpus (BigO(Bench) + CodeComplex) — see
> [`MODEL_CARD.md`](MODEL_CARD.md) for metrics, confusion matrices, and what
> they mean. The GNN rung, other languages' oracles, and the API are not
> implemented yet — see the build plan below.

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
feature extractor and one model serve all six languages — and use dynamic
profiling **only offline**, to auto-label training data, never in the
deployed path.

## Architecture

```
OFFLINE (labels training data; never deployed)      ONLINE (free tier; executes nothing)
  BigO(Bench) + CodeComplex + scraped code            Next.js (Vercel) ──▶ FastAPI (Render)
  + parallel synthetic generator                        paste box            tree-sitter → IR
        │                                                growth chart        → features → model
        ▼
  oracle: codegen → run at n → BIC curve fit
        │
        ▼
  train: rule → TF-IDF → LightGBM → GNN (numpy-served, no torch/ONNX)
```

## License

Code is MIT (`LICENSE`). Some training data carries its own, more
restrictive license — see `NOTICE.md` before reusing model weights
commercially.

## Build plan

Tracked in 7 phases; see the project spec for full detail (data strategy,
evaluation protocol, deployment, risks).
