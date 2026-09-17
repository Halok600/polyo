# oracle -- offline dynamic profiling (plan §6)

**OFFLINE ONLY.** This is the one package in the repo allowed to execute
solution code; `api/` must never import it (enforced by
`tests/test_isolation.py`).

Phase 2 ships the Python driver end to end: `spec.py` (test-spec schema),
`codegen/python.j2` (Jinja2 driver template), `runner.py` (subprocess
orchestration, per-run timeout), `fit.py` (BIC curve-shape classification).
`codegen/cpp.j2` `java.j2` `javascript.j2` `go.j2` and their oracle drivers
land in Phase 4, run via `.github/workflows/oracle-label.yml` since this dev
machine has no gcc/JDK/Go (see plan §0).

**Two non-obvious bugs found and fixed while building the self-check
(plan §15):**

1. `tracemalloc` measures heap allocations only -- it cannot see Python's
   C-level call stack, so it alone would misclassify a recursion-heavy
   solution (e.g. naive Fibonacci) as O(1) space even though plan §3
   requires the recursion stack to count. Fixed by measuring a second,
   independent signal -- max Python call-stack depth via `sys.settrace`,
   curve-fit the same way as heap bytes -- and taking whichever of the two
   implies the larger space class (`fit.classify_space`).
2. Measuring time, heap bytes and call depth in one instrumented call
   corrupts the *time* signal: `sys.settrace`'s default per-line tracing and
   `tracemalloc`'s own bookkeeping both carry real, non-negligible
   per-operation cost that scales with what the solution does, not with a
   fixed constant -- e.g. it made a plain O(n) linear scan measure as
   O(n log n). Fixed by (a) disabling per-line tracing on each traced frame
   (`frame.f_trace_lines = False` -- call/return tracking doesn't need it)
   and (b) measuring time, heap and depth as three separate calls per n
   (`codegen/python.j2`'s `_measure_time_ns` / `_measure_peak_bytes` /
   `_measure_call_depth`), so no instrumentation is active during the one
   call being timed. Fast (sub-microsecond) solutions are additionally
   autoranged into a batch of repeated calls (same idea as
   `timeit.Timer.autorange`), since a single call can be smaller than this
   machine's timer/scheduling noise floor.

See `tests/test_oracle_self_check.py` for the six-function verification
this was built against, and its comment on `binary_search` for the one
case that's allowed to come back honestly unclassified (plan §6's R^2
floor) rather than guessed.
