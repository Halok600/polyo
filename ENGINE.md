# The symbolic engine

PolyO answers "what is the worst-case time and auxiliary space of this code?" with a **static
cost analyser** (`analysis/`), not a model. It reads the code, derives a cost expression, and says how
sure it is. It never runs the code. The ML model (`models/`) is a fallback for the one case the
engine reports as unknown.

```
source ─ tree-sitter (ast for Python) ─▶ engine AST ─▶ abstract interpreter ─▶ cost expression
                                                         │                        │
                                       loop bounds · recurrences · library costs  ▼
                                                              projection ─▶ class + certainty
                                                                          + assumptions + derivation
```

## What it answers

```python
from analysis.engine import analyze

result = analyze(source, "java")
result.time.text       # "O(n * m)"          the exact expression, variables named n, m, k ... by parameter
result.time.cls        # "O(n^2)"            the legacy 7-class answer, rounded UP (lossy: result.time.lossy)
result.time.ext_cls    # "O(n^2)"            the extended class (adds O(sqrt n), O(n^2 log n), O(n!))
result.certainty       # "certain" | "assumed" | "unknown"
result.assumptions     # [Assumption(line, reason)]  every bound it had to assume
result.steps           # [Step(line, kind, text)]    the derivation: loops, costly calls, allocations, recursion
```

`certain`: every loop bound and every recursion was proven. `assumed`: a bound was assumed (and
reported with its line); the answer errs high, it is not guaranteed to. `unknown`: no static argument
bounds the code (a recursion with a data-dependent split, say): the API then asks the model instead.

### Conventions (what a label means)

* **Worst case**, as a function of the input sizes. Parameters get letters in order (`n`, `m`, `k` ...);
  a program that reads standard input uses `n` for what it reads; a line unpacked into `k` names has `k`
  tokens (`a, b = map(int, input().split())` is O(1), not O(length of the input)).
* Hash set/map operations are O(1) (average). `x in list` is O(n). Sorting is O(n log n); its auxiliary
  space is O(n) in Python and JavaScript and O(log n) in C++, Java (primitives) and Go.
* Repeated `s += c` on an immutable string (Python, Java, JavaScript, Go) copies it each time: O(length)
  per concatenation (C++ `std::string` amortises). A slice, a copy or a `substr` of length L costs O(L).
* **Space** is the memory the function allocates, including the value it returns, plus the recursion
  stack; the input is not counted. A hash container is as large as the number of distinct keys.
  Shallow copies (a list copy, a slice) are one slot per element; `deepcopy` and C++ by-value
  containers copy the elements too.
* A 2-D parameter is read by its name: `graph`, `adj` ... have rows of different lengths (size
  `V + E`), `edges`, `intervals`, `points` ... have rows of 2, anything else is rectangular `rows x cols`.
* Multi-variable answers keep their variables (`O(n * m)`); the legacy class collapses them into one `n`
  and rounds up: `n * m -> O(n^2)`, `n + m -> O(n)`, `sqrt n -> O(n)`, `n^2 log n -> O(n^3)`,
  anything above `n^3 -> O(n^3)`, `n!`, `3^n`, `n * 2^n -> O(2^n)`, space above `n^2 -> O(n^2)`.

## How it reasons

* **Values.** Integers carry a magnitude (a polynomial in the input sizes), containers a length and
  an element (a row of a 2-D structure, a dict's value), linked nodes the number of nodes they reach.
  Growable local containers get a symbolic capacity resolved at the end of the loop or function, so a
  list filled in a loop is as long as the loop says.
* **Cost algebra** (`poly.py`): polynomials with logs, exponentials and factorials, kept at big-O
  level (`order()` drops dominated terms), with closed forms for sums over arithmetic and geometric
  loops.
* **Loops** (`bounds.py`) are planned by the first rule that applies: counting (`range`, `i < n`),
  geometric (`i *= 2`, `i * i <= n`), Euclid (`a % b`), a potential function over the comparisons in
  the test, a worklist drained by pops (each element pushed once because of a visited marker, or
  because the structure is a tree), and **amortisation**: a pointer that never goes back (a sliding
  window's `left`, a count-down set before an enclosing loop, one pointer handed to another with
  `j = i ... i = j`) is paid for once across the enclosing loop, not once per iteration. Walks over
  adjacency lists sum the rows to `V + E`.
* **Recursion** (`recurrence.py`): every recursive call is recorded with its argument sizes, its
  multiplicity and the conditions it runs under. A potential argument gives the depth (arithmetic or
  geometric); one call gives depth times work, several give `a^depth` or a factorial, a loop of calls
  with a shrinking factor is solved with the master theorem per term of the work, recursion over a
  node structure visits each node once, memoised or visited-guarded recursion costs distinct states
  times work per state (shared through the container), mutual recursion is solved as one recurrence,
  every path is solved and the worst taken. Two calls on the two sides of a split point somewhere in
  `[lo, hi]` (quicksort) are an **assumed** `n^2`. What a recursion adds to a result container it was
  handed (backtracking: `result.append(path[:])`) is its number of calls times one call's addition.
* **Containers that feed themselves** (`res = [p + c for p in res for c in letters]`, `for r in
  list(res): res.append(r + [x])`) are exponential in the rounds.
* **Library costs** (`stdlib.py`): the cost and the allocation of several hundred calls in six
  languages, including `itertools` (`combinations(xs, 3)` is `n^3`).

## Verification (and what each number means)

| evidence | what it is | result |
|---|---|---|
| golden suite | 744 hand-written programs (6 languages); the engine was **developed against it** | 100%: it ratchets (`eval/golden_baseline.json`), it does not measure |
| held-out regression | 172 programs labelled before the engine saw them, then **fixed against** | a regression net |
| **blind corpus** | 144 programs written and labelled by independent agents that never saw the engine | **first contact: 85% time, 88% space, 81% both** (before any fix). certain: 93% / 95%; assumed: 64% / 71% |
| corpus audit | a random sample of the real training corpus (competitive-programming code), re-labelled independently | engine first contact **69% time, 76% space**; the corpus labels themselves 67% / 69% |
| invariance | dead code, renames, comments, reformatting, an unused helper, function order | 100% unchanged (the v1 GNN: 22% of blind answers survive 40 dead statements; the Phase 6 retrain: 84%) |
| cross-language | the same algorithm in every language must get one answer | 100% |

The served confidence is **measured**: the Laplace-smoothed accuracy per certainty level on the blind
corpus at first contact (`analysis/certainty_calibration.json`, recomputable with
`python -m eval.calibrate_certainty fit`). The first-contact records
(`tests/data/blind_v1_first_contact.json`, `eval/corpus_first_contact.json`) are evidence and are
never re-recorded: once the engine has been fixed against a program, re-running it would only measure
memorisation. After the fixes the blind corpus reads 135/144, a regression net rather than a
measurement. Every miss found was an engine gap fixed test-first (`tests/test_engine_blind_fixes.py`);
the nine left are strict `xfail`s with reasons (`tests/test_engine_blind.py`).

## Known limits

Not proven, so `assumed` or `unknown` (or simply wrong): a value that shrinks inside its own loop
(`while d * d <= n` with `n //= d`), loops over bit lengths, level widths that double each round,
a stack kept in an array with a top index, halving a linked list with slow and fast pointers, rows read
back from a table the same function fills, data-dependent early exits, worst-case hash collisions
(O(1) is assumed), dynamic dispatch, reflection and unknown external libraries. The interpreter keeps
orders, not constants, so "two appends per entry" reads as one append. Programs that reach a library
call the engine does not know are costed O(1) and reported as an assumption.

## Extending it

Add a failing test first: a program, the expression you expect, in the language that fails. Library
calls go in `analysis/stdlib.py` (`@lib("name", ...)`, returning a cost, an allocation and the value
the call produces); a new loop shape in `analysis/bounds.py`; a recursion shape in
`analysis/recurrence.py`. Run `python -m eval.golden_baseline` (it must not lose a case) and the engine
tests; if you are fixing against a new batch of independent programs, record the engine's first-contact
answers on it **before** changing the engine.
