# PolyO

**Static, multi-language time & space complexity analysis. No LLM calls, no code
execution at inference time.**

Paste code in Python, C++, Java, JavaScript, C, or Go and get its worst-case
**time and space** complexity: a class, the exact cost expression (`O(n * m)`,
not just `O(n^2)`), how sure the analyser is, and the derivation that led there
(which loop, which call, which recursion). It only ever *reads* the code.

**Live demo: [polyo.vercel.app](https://polyo.vercel.app)**. The API runs on a free Render instance, so the first request after a quiet spell can take up to ~40s while it wakes (a scheduled health check keeps it warm most of the time).

## What changed in v2, and why

The first version was a graph neural network over a normalised IR. When it was
tested on trivial programs it turned out to read the *size* of the program graph,
not its loop nesting: O(n^2) code came back as O(n^3), 80 lines of dead code
flipped an answer, and a ten-line rule on loop depth beat it. The diagnosis, with
evidence, is in [`eval/SCORECARD.md`](eval/SCORECARD.md).

So v2 is **engine first, model second**:

* **A symbolic cost engine** ([`analysis/`](ENGINE.md)) reads the code, runs an
  abstract interpreter over it and derives a cost expression. It solves loops
  (counting, geometric, amortised pointers, worklists), recurrences (master
  theorem, memoisation, mutual recursion) and the cost of several hundred
  library calls. It reports `certain`, `assumed` (with the assumptions and their
  lines) or `unknown`.
* **The GNN is a fallback**, asked only when the engine says `unknown`. It is
  retained as a research arm that has to *beat* the engine on a slice to be
  promoted, by a rule written before it was trained. It has not, in any slice.

## Results, and what each number is worth

The engine was developed against its own test programs, so those numbers measure
coverage, not accuracy. The honest number is the one on programs it had never seen.

| evidence | what it is | time | space |
|---|---|---|---|
| **blind corpus, first contact** | 144 programs (24 per language) written and labelled by independent agents that never saw the repo | **85%** (122/144) | **88%** (127/144) |
| &nbsp;&nbsp;engine says `certain` | 112 of those | 93% | 95% |
| &nbsp;&nbsp;engine says `assumed` | 28 of those | 64% | 71% |
| &nbsp;&nbsp;retrained GNN, same programs | | 38% | 44% |
| &nbsp;&nbsp;ten-line loop-depth rule | | 35% | 33% |
| real corpus audit | engine vs an independent re-labelling of a random sample of 49 real competitive-programming programs | 69% | 76% |
| golden suite (744 programs, 6 languages) | **developed against**, so a regression ratchet, not a measurement | 100% | 100% |
| invariance | dead code, renames, comments, reformatting, function order | engine 100% unchanged; GNN 84% (the v1 GNN: 22%) | |
| cross-language consistency | the same algorithm in every language gets one answer | 100% | |

Things worth knowing before quoting any of this:

* The 85% / 88% figure is **first contact**: recorded before any fix made because of those programs
  and never re-recorded. After the fixes the same corpus reads 135/144, but that is a regression net,
  not a measurement (`tests/data/blind_v1_first_contact.json` is the evidence).
* The served confidence is measured, not asserted: the accuracy per certainty level at first contact
  (`analysis/certainty_calibration.json`), Laplace-smoothed.
* The corpus the GNN trains on is noisy. An independent reading of a random sample agrees with the
  corpus label on only 67% (time) and 69% (space) of programs, because BigO(Bench) labels a program
  against its own test harness, not against the quantities it reads. A model trained on it can reach
  that ceiling, not the project's convention. See [`data/AUDIT.md`](data/AUDIT.md) and
  [`eval/LABEL_AUDIT.md`](eval/LABEL_AUDIT.md).
* The corpus is 97.6% Python and 631 problems. The non-Python slices are thin; the cross-language claims
  rest on the engine and the golden/blind programs, not on the corpus.
* The engine keeps orders, not constants, and the known limits (a value shrinking inside its own loop,
  array-backed stacks, halving a linked list...) are listed in [`ENGINE.md`](ENGINE.md#known-limits).
  Nine blind programs it still gets wrong are strict `xfail`s with reasons.

The full comparison (Wilson intervals, per-language, per-slice promotion) is in
[`eval/COMPARISON.md`](eval/COMPARISON.md). The readout experiment that fixed the GNN's
dead-code sensitivity (a mean readout dilutes a loop among straight-line nodes; mean+max does not) is in
[`eval/POOLING_EXPERIMENT.md`](eval/POOLING_EXPERIMENT.md).

### The v1 research results, kept for context

[`PHASE5_REPORT.md`](PHASE5_REPORT.md) and [`MODEL_CARD.md`](MODEL_CARD.md) hold the original model
ladder (rule, TF-IDF, LightGBM, GNN), the ablations, the zero-shot cross-language transfer experiment,
the conformal prediction sets and the LLM zero-shot baseline. They were measured against the noisy
corpus labels and on a served model that has since been replaced; read them as the history of the
model arm, not as the product's accuracy.

## Architecture

```
ONLINE (free tier; executes nothing)

 source ─ tree-sitter ─▶ engine AST ─▶ abstract interpreter ─▶ cost expression
                                          loops · recurrences    class + certainty
                                          library costs          assumptions + derivation
                                                  │
                                       "unknown" ─┴─▶ IR graph ─▶ GNN (numpy, no torch)
```

* `analysis/`: the symbolic engine. See [`ENGINE.md`](ENGINE.md) for what it answers, the
  conventions behind a label, how it reasons and what each verification number means.
* `api/predict.py`: the hybrid. Engine first, GNN on `unknown` or an engine crash (with a warning),
  `mode="ml"` for the GNN alone. All new response fields are additive: the original seven keep their
  shape.
* `parsing/`, `core/`, `features/`: the tree-sitter normalisation into one IR (still the GNN's input).
* `models/`: the research arm (rule, TF-IDF, LightGBM, GNN). The served GNN's forward pass is
  reimplemented in numpy (`models/export_numpy.py`), verified identical to the trained model within
  `1e-4`, so the serving image has no torch.
* `oracle/`: offline only. Executes generated programs to label training data; never deployed.
* `eval/`: the golden suite, perturbations, scorecard, blind comparison and audits.

## Reproduce

```
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
python -m pytest -q                  # ~3,900 tests
python -m eval.golden_baseline       # must not lose a case
python -m eval.compare               # engine vs model vs rule on the independent corpora
```

## License

Code is MIT (`LICENSE`). Some training data carries its own, more restrictive license; see
`NOTICE.md` before reusing model weights commercially.

## Status

Done and tested: the engine, the hybrid API, the retrained model, the evaluation and the web UI. CI
(Python tests, the golden-suite ratchet, web lint/type-check/unit/build/e2e, the serving image) is green.

* **The web UI shows the v2 answer**: the exact expression as the headline, a CERTAIN / ASSUMED / MODEL badge
  with the measured accuracy of that level, the derivation inline in the code (a gutter marking each line a
  step or an assumption points at, plus a linked step list), and the model's probability bars only when the
  model answered. It has not been through a usability review, and the layout on a phone has only been
  checked at one width (390px).
* **Deployed**: the web app on Vercel ([polyo.vercel.app](https://polyo.vercel.app)) and the API on Render's
  free tier. After deployment it was checked end to end with a headless browser against the live site: all six
  language examples return the expected class, with no console errors or failed requests. [`DEPLOY.md`](DEPLOY.md)
  has the steps to reproduce it.
* One CI test is timing-based (the Java oracle recovers a library sort's complexity from wall-clock
  measurements on a shared runner) and has failed intermittently before; it is not part of the product path.
