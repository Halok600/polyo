# data -- ingestion, corpus schema, splits (plan §7, §12)

`corpus.py` is the shared `CorpusRecord` schema every ingestion script
writes. Phase 2 ships two ingestion scripts, both Python-only:

- `ingest_bigobench.py` -- BigO(Bench) (time + space labels). See its module
  docstring for the exact download command and a real limitation found while
  writing it: most of BigO(Bench)'s labels are per-input-variable (e.g.
  `O(n+m**2+k)`) and don't map to our single-variable taxonomy; only the
  clean single-variable forms are kept, loudly dropping the rest.
- `ingest_codecomplex.py` -- CodeComplex-Data (time only). Ingests either
  language split via `--lang {python,java}` (default `python`); both splits
  share the same schema and label vocabulary, verified directly against the
  real files. Java support added in Phase 5 prep to give the cross-language
  transfer experiment a real, non-synthetic second training language.

Neither script's raw or processed output is committed
(`data/raw/`, `data/processed/` are gitignored) -- both datasets carry
their own licenses (see `NOTICE.md`), and the raw files are hundreds of MB
to multi-GB. Run the ingestion scripts locally once you've downloaded the
raw files (commands are in each script's docstring); each prints a JSON
summary of kept/dropped counts.

`build.py` (Phase 3) dedupes and splits the ingested corpus by
**(source, problem_id)**, not just `problem_id` -- BigO(Bench) and
CodeComplex use overlapping id spellings, so splitting on the bare id could
leak a problem across train/val/test (plan §9's "no problem_id in more than
one split", strengthened). `tests/test_data_splits.py` enforces this in CI.
Writes `data/processed/split_{train,val,test}.jsonl` (also gitignored).

`synth.py` (Phase 4, plan §7/§14) is the parallel synthetic generator: a
small, fixed library of algorithms, each with the *same* shape written in
all six languages and an exact label by construction -- no oracle
execution, since the label is already known. Chosen specifically to cover
the classes plan §7 says real corpora barely contain (O(n^3), O(2^n) time;
O(n log n) space), and every language variant of one algorithm shares a
`problem_id`, which is what makes the cross-language transfer experiment
(Phase 5) possible. Run with `python -m data.synth` (writes
`data/processed/synth.jsonl`, picked up by `build.py` like any other
processed corpus file); `tests/test_synth.py` parses every algorithm in
every language through the real per-language parser as a correctness
check, since that needs no compiler even for C/C++/Java/Go.

Expanded from 9 to 14 algorithms in Phase 5 prep (`recursive_sum`,
`binary_search_recursive`, `halving_loop`, `merge_sort`,
`memoized_fibonacci`), all built only from IR symbols already consistently
mapped across every `lang/*.toml` (`ALGORITHMS`'s docstring comment in
`synth.py` explains why a hash-based example was considered and dropped --
the hash-container mapping is inconsistent across languages today).
`memoized_fibonacci` is a deliberate minimal-pair contrast with
`naive_fibonacci` (same recurrence, O(2^n) vs O(n) purely from memoisation)
and `binary_search_recursive` is one with the existing iterative
`binary_search` (same time class, different space class from the
recursion stack) -- both aimed at the transfer/failure-bucket analysis in
plan §9, not just raw record count.
