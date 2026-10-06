# What is in the training corpus (measured 2026-10-03)

`python -m data.audit` recomputes everything below from `data/processed/split_*.jsonl` (the corpus
itself is not committed: see `NOTICE.md`).

| | train | val | test |
|---|---|---|---|
| records | 144,549 | 22,554 | 26,921 |
| problems | 631 | 119 | 141 |
| Python | 141,113 (97.6%) | 21,880 | 26,020 |
| Java | 3,367 | 670 | 871 |
| C++ / C / JavaScript / Go | 20 / 15 / 14 / 20 | 1 each | 9 / 5 / 7 / 9 |
| BigO(Bench) / CodeComplex / synthetic / TheAlgorithms | 137,990 / 6,450 / 79 / 30 | 21,030 / 1,518 / 6 / 0 | 25,262 / 1,611 / 38 / 10 |
| no time label / no space label | 13,188 (9.1%) / 9,284 (6.4%) | 1,634 / 2,225 | 1,457 / 2,335 |
| records with neither label | 0 | 0 | 0 |

**Concentration.** The median problem has 75 solutions, the 95th percentile 783, and one problem has
13,357 (9% of the training set); 383 of the 631 training problems have more than 50. A model trained
on this sees a few dozen problems many thousand times.

**Near-duplicates.** In a deterministic sample of 20,000 training records, 8.7% have another solution
of the same problem with the same normalised-IR shape and the same labels (a lower bound for the whole
corpus: a sample sees fewer pairs). `python -m data.build --dedupe-ir --max-per-problem N` collapses
them and caps the per-problem count; the training slice used by `eval/pooling_experiment.py` is built
that way.

**Python time labels (train).** O(n) 71,269 · O(1) 30,993 · O(n log n) 13,176 · O(n²) 9,772 ·
O(log n) 1,833 · O(2^n) 463 · O(n³) 419 · none 13,188. The O(1) share (22%) is the label convention
at work: see below.

**The labels are not the project's convention.** BigO(Bench) labels a program by measuring it against
the dimension of its own test harness, so a loop over a small fixed-size input reads as O(1). An
independent reading of a random sample of 49 programs under the project's convention (worst case in
the quantities the program reads) agrees with the corpus label on **67% of time labels and 69% of space
labels**; the symbolic engine agrees on 69% / 76% at first contact. The full audit is in
`eval/LABEL_AUDIT.md` (`tests/data/corpus_audit_v1.json` holds the evidence, hashes instead of code).
A model trained on these labels can reach their ceiling, not the project's.

**Not done, deliberately.** The training rows with no time label are rows whose raw label did not map
to a single-variable class (`data/README.md`: BigO(Bench) labels are mostly per input variable, e.g.
`O(n+m**2+k)`, and only the clean single-variable forms are kept). They are excluded rather than
re-mapped: collapsing them to one variable under the project's conventions would add rows that carry
the same measurement-convention noise as the rest, on the strength of a mapping nobody has audited.
