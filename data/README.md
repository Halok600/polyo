# data -- ingestion, corpus schema, splits (plan §7, §12)

`corpus.py` is the shared `CorpusRecord` schema every ingestion script
writes. Phase 2 ships two ingestion scripts, both Python-only:

- `ingest_bigobench.py` -- BigO(Bench) (time + space labels). See its module
  docstring for the exact download command and a real limitation found while
  writing it: most of BigO(Bench)'s labels are per-input-variable (e.g.
  `O(n+m**2+k)`) and don't map to our single-variable taxonomy; only the
  clean single-variable forms are kept, loudly dropping the rest.
- `ingest_codecomplex.py` -- CodeComplex-Data's Python split (time only).

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

`synth.py` (parallel synthetic generator, all six languages) lands in
Phase 4 -- see plan §14.
