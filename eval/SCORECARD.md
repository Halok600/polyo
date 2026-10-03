# PolyO golden-suite scorecard: Phase 0 baseline

Recorded on 2026-10-02, BEFORE any v2 engine code exists. This is the failure the rebuild fixes: the shipped GNN (the model behind gnn and product) scores below a ten-line loop-depth lookup (rule), disagrees with itself across languages, and changes its answer when harmless dead code is added. Regenerate with: python -m eval.scorecard --predictors rule,gnn --out eval/SCORECARD.md

## Headline (gated cases only; known-hard cases are reported separately)

| predictor | n | time acc | space acc | time rank err | space rank err | time expr | space expr | errors |
|---|---|---|---|---|---|---|---|---|
| rule | 744 | 57.9% | 51.2% | 0.80 | 1.03 | n/a | n/a | 0 |
| gnn | 744 | 34.5% | 49.7% | 1.12 | 1.20 | n/a | n/a | 0 |

## Per language (time accuracy / space accuracy)

| language | n | rule | gnn |
|---|---|---|---|
| python | 138 | 56.5% / 49.3% | 50.0% / 45.7% |
| cpp | 138 | 57.2% / 48.6% | 25.4% / 44.2% |
| java | 138 | 58.0% / 50.0% | 31.9% / 49.3% |
| javascript | 138 | 56.5% / 46.4% | 27.5% / 50.0% |
| go | 138 | 60.9% / 51.4% | 44.9% / 56.5% |
| c | 54 | 59.3% / 77.8% | 16.7% / 57.4% |

## Per time class (recall of the expected class)

| expected class | n | rule | gnn |
|---|---|---|---|
| O(1) | 41 | 73.2% | 56.1% |
| O(log n) | 78 | 0.0% | 9.0% |
| O(n) | 289 | 76.1% | 35.6% |
| O(n log n) | 69 | 0.0% | 14.5% |
| O(n^2) | 199 | 74.9% | 42.2% |
| O(n^3) | 42 | 76.2% | 31.0% |
| O(2^n) | 26 | 0.0% | 65.4% |

## Per space class (recall of the expected class)

| expected class | n | rule | gnn |
|---|---|---|---|
| O(1) | 378 | 96.3% | 70.9% |
| O(log n) | 47 | 0.0% | 0.0% |
| O(n) | 275 | 5.1% | 31.3% |
| O(n^2) | 44 | 6.8% | 36.4% |

## Cross-language consistency

The same algorithm written in every language must get ONE answer. *Consistent* = every language got the same (time, space) answer, right or wrong; *all correct* = every language matched its label. Cases with deliberate per-language overrides are excluded.

| predictor | cases | consistent | all correct |
|---|---|---|---|
| rule | 132 | 81.8% | 34.1% |
| gnn | 132 | 6.8% | 3.0% |

## Perturbation invariance

Share of semantics-preserving rewrites (dead O(1) statements, renamed locals, comments, re-indentation, an unused O(1) helper, reordered functions) after which the predicted (time, space) classes did not change.

| perturbation | rule | gnn |
|---|---|---|
| comments | 100.0% (744/744) | 100.0% (744/744) |
| dead_code | 100.0% (741/741) | 21.2% (157/741) |
| dead_code_heavy | 100.0% (741/741) | 18.4% (136/741) |
| reformat | 100.0% (742/742) | 100.0% (742/742) |
| rename | 100.0% (721/721) | 100.0% (721/721) |
| reorder_functions | 100.0% (47/47) | 68.1% (32/47) |
| unused_helper | 100.0% (744/744) | 65.3% (486/744) |
| **overall** | 100.0% (4480/4480) | 67.4% (3018/4480) |

## Known-hard cases (tagged `known-limitation`; reported, never gated)

| predictor | n | time acc | space acc |
|---|---|---|---|
| rule | 15 | 33.3% | 0.0% |
| gnn | 15 | 53.3% | 0.0% |

## Per group (time accuracy)

| group | rule | gnn |
|---|---|---|
| basics | 92.7% | 39.3% |
| constant | 100.0% | 55.6% |
| hashing | 57.0% | 48.8% |
| multivar | 86.7% | 41.3% |
| nlogn | 7.8% | 15.6% |
| pointers | 85.5% | 46.4% |
| programs | 50.0% | 62.5% |
| recursion | 11.2% | 23.4% |
| space | 89.4% | 31.9% |
| sublinear | 25.0% | 9.1% |
