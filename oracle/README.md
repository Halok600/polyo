# oracle -- offline dynamic profiling (plan §6)

**OFFLINE ONLY.** This is the one package in the repo allowed to execute
solution code; `api/` must never import it (enforced by
`tests/test_isolation.py`).

Phase 2 shipped the Python driver end to end: `spec.py` (test-spec schema),
`codegen/python.j2` (Jinja2 driver template), `runner.py` (subprocess
orchestration, per-run timeout), `fit.py` (BIC curve-shape classification).
Phase 4 adds `codegen/{cpp,c,java,javascript,go}.j2`, `interposer.c` (C/C++
memory measurement) and generalises `runner.py` to compile/run each --
verified for real via `.github/workflows/oracle-label.yml` since this dev
machine has no gcc/JDK/Go (see plan §0); JavaScript alone also runs and is
tested locally, since Node is present here.

**Per-language execution, and what each measures (plan §6):**

| Language | Compile? | Time | Peak/allocated bytes |
|---|---|---|---|
| Python | no | `time.perf_counter_ns`, autoranged batches | `tracemalloc` peak |
| JavaScript | no (`node --expose-gc`) | `process.hrtime.bigint`, JIT-warmed | `process.memoryUsage().heapUsed`, forced GC before/after |
| Java | `javac` | `System.nanoTime`, JIT-warmed | `ThreadMXBean.getThreadAllocatedBytes` (cumulative, not live) |
| C / C++ | `gcc`/`g++ -O2` | `clock_gettime(CLOCK_MONOTONIC)` | `interposer.c`'s malloc/free interposer, `LD_PRELOAD`-ed, peak live bytes |
| Go | no (`go run`) | `time.Now()` | `runtime.MemStats.TotalAlloc` delta (cumulative, not live) |

**Solution-source contract per language** (oracle/runner.py owns the
wrapping, not the caller): Python/C/C++ solutions are used as-is; Java's is
a bare **static** method body, wrapped in `class Solution { ... }`; Go's is
a bare function, given a `package main` clause; JavaScript's is a bare
function, given a trailing `module.exports.<entrypoint> = <entrypoint>`
line. C/C++ additionally need the length of any `n`-sized `int[]` param
passed as an extra argument immediately after the array (see `codegen/c.j2`'s
header comment) -- unlike every other language here, a C/C++ array doesn't
carry its own length.

**A known, documented scope limit (plan §14 Phase 4):** only the Python
driver measures call-stack depth (`sys.settrace`), the second signal
`fit.classify_space` needs to correctly classify a solution that recurses
without allocating (see the two bugs below). C/C++'s native stack, Java's
JVM stack, V8's stack and Go's goroutine stack are none of them heap-backed,
so the other four languages have the *same* blind spot Python did before
that fix -- but a reliable, CI-verifiable equivalent (e.g. GCC's
`-finstrument-functions` for C/C++) needs live iteration this dev machine's
missing toolchains can't provide, so it was scoped out rather than shipped
half-verified. `Sample.max_call_depth` is `None` for every language but
Python; `fit.fit_space_depth` returns `None` for those, and
`fit.classify_space` correctly falls back to the byte signal alone --
honest degradation, not a silently wrong O(1). A future session with a
working CI feedback loop on this specific piece is the natural way to close
the gap.

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
