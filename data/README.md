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

`synth.py` (parallel synthetic generator, all six languages) and `build.py`
(dedupe, problem-level splits, class balancing) land in Phase 4 and Phase 3
respectively -- see plan §14.
