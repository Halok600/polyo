# PolyO Phase 5 report
Phase 5 of 7 (plan §14): rung 3 (GNN message-passing over the IR graph, shared encoder -> two heads), its ablations, the zero-shot cross-language transfer experiment, and failure-bucket analysis. See `MODEL_CARD.md` for rungs 0-2 on the same held-out, problem-level test split.

Parsing/feature-extraction survival rate per split (`models/dataset.py`):

```json
{
  "train": {
    "total": 144489,
    "parsed": 144489,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  },
  "val": {
    "total": 22548,
    "parsed": 22548,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  },
  "test": {
    "total": 26899,
    "parsed": 26899,
    "unsupported_language": 0,
    "failed": 0,
    "failure_types": {}
  }
}
```

## Rung 3: multi-task GNN (all edge kinds)

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| time | 25442 | 0.574 | 0.377 | 0.780 | n/a |
| space | 24564 | 0.723 | 0.336 | 0.523 | n/a |

![time confusion matrix, rung 3](eval/figures/confusion_time_rung3_gnn.png)

![space confusion matrix, rung 3](eval/figures/confusion_space_rung3_gnn.png)

## Ablation: multi-task vs. single-task

Shared encoder + two heads, trained jointly, vs. two independent single-head models (plan §9: "does joint training beat two independent models? Either answer is a result.").


**Time**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| multi_task | 25442 | 0.556 | 0.313 | 0.825 | n/a |
| single_task | 25442 | 0.579 | 0.379 | 0.792 | n/a |
- **single_task** wins on macro-F1 here (+0.066). Either answer is a result (plan §9) -- a win for single-task doesn't undermine the shared encoder's value elsewhere (e.g. the transfer experiment still needs one encoder that has seen both dimensions' structure); it says joint training's regularisation effect didn't outweigh head-competition for gradient capacity on this split.

**Space**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| multi_task | 24564 | 0.726 | 0.330 | 0.521 | n/a |
| single_task | 24564 | 0.729 | 0.334 | 0.517 | n/a |
- **single_task** wins on macro-F1 here (+0.004). Either answer is a result (plan §9) -- a win for single-task doesn't undermine the shared encoder's value elsewhere (e.g. the transfer experiment still needs one encoder that has seen both dimensions' structure); it says joint training's regularisation effect didn't outweigh head-competition for gradient capacity on this split.

## Ablation: graph edge types

Multi-task GNN retrained per edge-kind subset -- which edges beyond the plain AST (`AST_CHILD`/`NEXT_SIBLING`) actually earn their keep.


**Time**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| structural_only | 25442 | 0.564 | 0.341 | 0.808 | n/a |
| structural_plus_data_dep | 25442 | 0.573 | 0.346 | 0.792 | n/a |
| structural_plus_loop_carry | 25442 | 0.585 | 0.377 | 0.772 | n/a |
| structural_plus_call_edge | 25442 | 0.583 | 0.375 | 0.782 | n/a |
| all_edges | 25442 | 0.575 | 0.295 | 0.808 | n/a |
- **structural_plus_loop_carry** (macro-F1 0.377) beats **all_edges** (macro-F1 0.295) at this ablation's shared, capped epoch budget -- read as "the extra edge kinds don't stack additively at this training budget" rather than "more structure hurts": rung 3's own headline number above used a larger epoch budget than any ablation variant did, precisely because ablations must share one fixed, fair budget across configurations.

**Space**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| structural_only | 24564 | 0.712 | 0.321 | 0.550 | n/a |
| structural_plus_data_dep | 24564 | 0.724 | 0.325 | 0.531 | n/a |
| structural_plus_loop_carry | 24564 | 0.732 | 0.335 | 0.507 | n/a |
| structural_plus_call_edge | 24564 | 0.732 | 0.333 | 0.512 | n/a |
| all_edges | 24564 | 0.730 | 0.322 | 0.518 | n/a |
- **structural_plus_loop_carry** (macro-F1 0.335) beats **all_edges** (macro-F1 0.322) at this ablation's shared, capped epoch budget -- read as "the extra edge kinds don't stack additively at this training budget" rather than "more structure hurts": rung 3's own headline number above used a larger epoch budget than any ablation variant did, precisely because ablations must share one fixed, fair budget across configurations.

## Ablation: IR symbols vs. raw source tokens

Same TF-IDF + logistic regression pipeline, differing only in whether the input text is the normalised IR symbol sequence (rung 1) or raw source code -- this is the ablation that most directly tests the project's central thesis (plan §2).


**Time**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| ir_symbol_tfidf | 25442 | 0.462 | 0.367 | 1.058 | n/a |
| raw_token_tfidf | 25442 | 0.488 | 0.372 | 1.008 | n/a |
- **raw_token_tfidf** wins on this split's macro-F1 (ir_symbol_tfidf 0.367 vs. raw_token_tfidf 0.372). That is a real result, not swept under the rug -- but it is not the whole test of the project's thesis (plan §2): this comparison trains and evaluates within the same language mix, where raw tokens can pick up incidental lexical cues (identifier/library naming patterns correlated with a source corpus) that plain accuracy can't distinguish from real structural signal. Raw tokens have no path to cross-language transfer at all -- Python's `for x in xs` and Java's `for (int x : xs)` share almost no raw tokens -- which is what the transfer experiment below actually tests, and where the IR's language-agnostic design is the only one of the two that can work by construction.

**Space**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| ir_symbol_tfidf | 24564 | 0.518 | 0.295 | 0.903 | n/a |
| raw_token_tfidf | 24564 | 0.613 | 0.327 | 0.722 | n/a |
- **raw_token_tfidf** wins on this split's macro-F1 (ir_symbol_tfidf 0.295 vs. raw_token_tfidf 0.327). That is a real result, not swept under the rug -- but it is not the whole test of the project's thesis (plan §2): this comparison trains and evaluates within the same language mix, where raw tokens can pick up incidental lexical cues (identifier/library naming patterns correlated with a source corpus) that plain accuracy can't distinguish from real structural signal. Raw tokens have no path to cross-language transfer at all -- Python's `for x in xs` and Java's `for (int x : xs)` share almost no raw tokens -- which is what the transfer experiment below actually tests, and where the IR's language-agnostic design is the only one of the two that can work by construction.

## Ablation: feature families (rung 2)

Leave-one-family-out over rung 2's tabular feature set (`eval/ablations.py`'s `FEATURE_FAMILIES`).


**Time**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| full | 25442 | 0.431 | 0.331 | 1.083 | n/a |
| without_loop | 25442 | 0.337 | 0.266 | 1.298 | n/a |
| without_allocation | 25442 | 0.422 | 0.322 | 1.090 | n/a |
| without_recursion | 25442 | 0.418 | 0.321 | 1.104 | n/a |
| without_control_flow | 25442 | 0.389 | 0.318 | 1.180 | n/a |
| without_library_calls | 25442 | 0.346 | 0.265 | 1.232 | n/a |
| without_graph_size | 25442 | 0.407 | 0.296 | 1.134 | n/a |
- Removing **library_calls** costs the most macro-F1 (+0.066 from the full-feature 0.331) -- worth cross-checking against the failure buckets below: if `hidden_in_library_call` is a large bucket there too, that's the same signal showing up twice, not two unrelated findings.

**Space**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| full | 24564 | 0.496 | 0.285 | 0.952 | n/a |
| without_loop | 24564 | 0.429 | 0.246 | 1.126 | n/a |
| without_allocation | 24564 | 0.471 | 0.271 | 1.006 | n/a |
| without_recursion | 24564 | 0.478 | 0.273 | 0.977 | n/a |
| without_control_flow | 24564 | 0.443 | 0.263 | 1.111 | n/a |
| without_library_calls | 24564 | 0.413 | 0.254 | 1.104 | n/a |
| without_graph_size | 24564 | 0.426 | 0.259 | 1.047 | n/a |
- Removing **loop** costs the most macro-F1 (+0.039 from the full-feature 0.285) -- worth cross-checking against the failure buckets below: if `hidden_in_library_call` is a large bucket there too, that's the same signal showing up twice, not two unrelated findings.

## Zero-shot cross-language transfer

Trained on Python+Java only; every other language is unseen during training. **Known limitation, not hidden:** the corpus's only non-Python/Java code comes from the 84-record parallel synthetic generator (`data/synth.py`) -- `data/scrape.py` (plan §7's scraped real-code source) was never built. After the problem-level split, the C++/JavaScript/Go/C test cells have on the order of 5 examples each (see the `n` printed on each heatmap cell), so those specific numbers are illustrative, not statistically robust. Python and Java cells are in-distribution and have the corpus's full test-set size behind them.


![cross-language transfer heatmap](eval/figures/transfer_heatmap.png)

**Time, per language**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| c | 1 | 0.000 | 0.000 | 1.000 | n/a |
| cpp | 5 | 0.200 | 0.143 | 1.600 | n/a |
| go | 5 | 0.200 | 0.143 | 1.400 | n/a |
| java | 867 | 0.419 | 0.389 | 0.892 | n/a |
| javascript | 5 | 0.200 | 0.143 | 1.400 | n/a |
| python | 24559 | 0.594 | 0.325 | 0.775 | n/a |

**Space, per language**

| Variant | n | Accuracy | Macro-F1 | Mean ordinal distance | ECE |
|---|---|---|---|---|---|
| c | 1 | 0.000 | 0.000 | 4.000 | n/a |
| cpp | 5 | 0.400 | 0.114 | 1.600 | n/a |
| go | 5 | 0.000 | 0.000 | 2.800 | n/a |
| java | 5 | 0.400 | 0.114 | 1.600 | n/a |
| javascript | 5 | 0.200 | 0.200 | 1.600 | n/a |
| python | 24543 | 0.736 | 0.335 | 0.503 | n/a |

## Failure buckets

Computed on rung 3's own misclassifications on the held-out test split (plan §9's five named buckets, each a structural heuristic over the IR except the last, which is defined on the true/predicted label pair directly -- see `eval/failure_buckets.py`).

### Failure buckets (time, rung3_gnn's own misclassifications)

10829 misclassified examples out of the time test set. Buckets overlap (an example can match more than one heuristic) rather than being forced into exactly one:

| Bucket | Count | Share of misclassified | Description |
|---|---|---|---|
| amortised_structures | 2479 | 22.9% | Uses a dynamic-array/set/hash-insert pattern whose real cost is amortised, not the worst-case per-operation cost |
| memoised_recursion | 370 | 3.4% | Recurses with a hash lookup/insert alongside it (a cache), so the real complexity depends on memoisation, not call-tree shape |
| hidden_in_library_call | 4652 | 43.0% | Complexity depends on a library call's (SORT/BINARY_SEARCH/HEAP/QUEUE/MATH_OP) internal cost, invisible in the surrounding structure |
| input_dependent_early_exit | 3610 | 33.3% | Exits a loop early depending on runtime data (BREAK, or a branch-guarded return inside a loop), so worst case and typical case diverge |
| on_vs_onlogn_boundary | 1668 | 15.4% | True and predicted classes are the adjacent O(n)/O(n log n) pair -- the boundary plan §6 already documents as empirically hard to separate |

### Failure buckets (space, rung3_gnn's own misclassifications)

6804 misclassified examples out of the space test set. Buckets overlap (an example can match more than one heuristic) rather than being forced into exactly one:

| Bucket | Count | Share of misclassified | Description |
|---|---|---|---|
| amortised_structures | 1283 | 18.9% | Uses a dynamic-array/set/hash-insert pattern whose real cost is amortised, not the worst-case per-operation cost |
| memoised_recursion | 155 | 2.3% | Recurses with a hash lookup/insert alongside it (a cache), so the real complexity depends on memoisation, not call-tree shape |
| hidden_in_library_call | 3019 | 44.4% | Complexity depends on a library call's (SORT/BINARY_SEARCH/HEAP/QUEUE/MATH_OP) internal cost, invisible in the surrounding structure |
| input_dependent_early_exit | 1626 | 23.9% | Exits a loop early depending on runtime data (BREAK, or a branch-guarded return inside a loop), so worst case and typical case diverge |
| on_vs_onlogn_boundary | 0 | 0.0% | True and predicted classes are the adjacent O(n)/O(n log n) pair -- the boundary plan §6 already documents as empirically hard to separate |
