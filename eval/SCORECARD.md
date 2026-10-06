# PolyO golden-suite scorecard: after Phase 6 (ML research arm)

Recorded on 2026-10-06. `gnn` is the Phase 6 retrain (mean+max readout, dead-code augmentation, IR-deduplicated corpus capped per problem); `rule` is unchanged. `symbolic` is the engine alone (`analysis/`, no model, no fallback); `product` is what the API serves: the engine first, the model only when the engine says unknown. Read the engine's 100% with care: it was developed against this suite, so it shows the suite is covered, not how the engine does on new code. The honest generalisation numbers are the first-contact ones on the independent blind corpus (eval/COMPARISON.md). Known-hard cases are tagged and never gated. Regenerate with: python -m eval.scorecard --predictors rule,gnn,symbolic,product --out eval/SCORECARD.md

## Headline (gated cases only; known-hard cases are reported separately)

| predictor | n | time acc | space acc | time rank err | space rank err | time expr | space expr | errors |
|---|---|---|---|---|---|---|---|---|
| rule | 744 | 57.9% | 51.2% | 0.80 | 1.03 | n/a | n/a | 0 |
| gnn | 744 | 44.6% | 59.3% | 1.04 | 0.84 | n/a | n/a | 0 |
| symbolic | 744 | 100.0% | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 0 |
| product | 744 | 100.0% | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 0 |

## Per language (time accuracy / space accuracy)

| language | n | rule | gnn | symbolic | product |
|---|---|---|---|---|---|
| python | 138 | 56.5% / 49.3% | 41.3% / 60.1% | 100.0% / 100.0% | 100.0% / 100.0% |
| cpp | 138 | 57.2% / 48.6% | 47.8% / 57.2% | 100.0% / 100.0% | 100.0% / 100.0% |
| java | 138 | 58.0% / 50.0% | 49.3% / 57.2% | 100.0% / 100.0% | 100.0% / 100.0% |
| javascript | 138 | 56.5% / 46.4% | 44.2% / 59.4% | 100.0% / 100.0% | 100.0% / 100.0% |
| go | 138 | 60.9% / 51.4% | 40.6% / 56.5% | 100.0% / 100.0% | 100.0% / 100.0% |
| c | 54 | 59.3% / 77.8% | 44.4% / 74.1% | 100.0% / 100.0% | 100.0% / 100.0% |

## Per time class (recall of the expected class)

| expected class | n | rule | gnn | symbolic | product |
|---|---|---|---|---|---|
| O(1) | 41 | 73.2% | 78.0% | 100.0% | 100.0% |
| O(log n) | 78 | 0.0% | 0.0% | 100.0% | 100.0% |
| O(n) | 289 | 76.1% | 48.4% | 100.0% | 100.0% |
| O(n log n) | 69 | 0.0% | 44.9% | 100.0% | 100.0% |
| O(n^2) | 199 | 74.9% | 49.2% | 100.0% | 100.0% |
| O(n^3) | 42 | 76.2% | 59.5% | 100.0% | 100.0% |
| O(2^n) | 26 | 0.0% | 23.1% | 100.0% | 100.0% |

## Per space class (recall of the expected class)

| expected class | n | rule | gnn | symbolic | product |
|---|---|---|---|---|---|
| O(1) | 378 | 96.3% | 87.6% | 100.0% | 100.0% |
| O(log n) | 47 | 0.0% | 6.4% | 100.0% | 100.0% |
| O(n) | 275 | 5.1% | 38.9% | 100.0% | 100.0% |
| O(n^2) | 44 | 6.8% | 0.0% | 100.0% | 100.0% |

## Cross-language consistency

The same algorithm written in every language must get ONE answer. *Consistent* = every language got the same (time, space) answer, right or wrong; *all correct* = every language matched its label. Cases with deliberate per-language overrides are excluded.

| predictor | cases | consistent | all correct |
|---|---|---|---|
| rule | 132 | 81.8% | 34.1% |
| gnn | 132 | 24.2% | 9.8% |
| symbolic | 132 | 100.0% | 100.0% |
| product | 132 | 100.0% | 100.0% |

## Perturbation invariance

Share of semantics-preserving rewrites (dead O(1) statements, renamed locals, comments, re-indentation, an unused O(1) helper, reordered functions) after which the predicted (time, space) classes did not change.

| perturbation | rule | gnn | symbolic | product |
|---|---|---|---|---|
| comments | 100.0% (744/744) | 100.0% (744/744) | 100.0% (744/744) | 100.0% (744/744) |
| dead_code | 100.0% (741/741) | 85.2% (631/741) | 100.0% (741/741) | 100.0% (741/741) |
| dead_code_heavy | 100.0% (741/741) | 83.3% (617/741) | 100.0% (741/741) | 100.0% (741/741) |
| reformat | 100.0% (742/742) | 100.0% (742/742) | 100.0% (742/742) | 100.0% (742/742) |
| rename | 100.0% (721/721) | 100.0% (721/721) | 100.0% (721/721) | 100.0% (721/721) |
| reorder_functions | 100.0% (47/47) | 68.1% (32/47) | 100.0% (47/47) | 100.0% (47/47) |
| unused_helper | 100.0% (744/744) | 82.8% (616/744) | 100.0% (744/744) | 100.0% (744/744) |
| **overall** | 100.0% (4480/4480) | 91.6% (4103/4480) | 100.0% (4480/4480) | 100.0% (4480/4480) |

## Known-hard cases (tagged `known-limitation`; reported, never gated)

| predictor | n | time acc | space acc |
|---|---|---|---|
| rule | 15 | 33.3% | 0.0% |
| gnn | 15 | 33.3% | 0.0% |
| symbolic | 15 | 0.0% | 46.7% |
| product | 15 | 0.0% | 46.7% |

## Per group (time accuracy)

| group | rule | gnn | symbolic | product |
|---|---|---|---|---|
| basics | 92.7% | 76.0% | 100.0% | 100.0% |
| constant | 100.0% | 94.4% | 100.0% | 100.0% |
| hashing | 57.0% | 45.3% | 100.0% | 100.0% |
| multivar | 86.7% | 52.0% | 100.0% | 100.0% |
| nlogn | 7.8% | 42.2% | 100.0% | 100.0% |
| pointers | 85.5% | 43.5% | 100.0% | 100.0% |
| programs | 50.0% | 47.5% | 100.0% | 100.0% |
| recursion | 11.2% | 22.4% | 100.0% | 100.0% |
| space | 89.4% | 46.8% | 100.0% | 100.0% |
| sublinear | 25.0% | 1.1% | 100.0% | 100.0% |
