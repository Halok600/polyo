"""Failure-bucket analysis (plan §9): amortised structures · memoised
recursion · complexity hidden inside a library call · input-dependent early
exit · the O(n)/O(n log n) boundary.

Each bucket is a structural heuristic over the IR -- no bucket needs ground
truth beyond "this example was misclassified" -- except the last, which is
defined directly on the true/predicted label pair, since there is no IR
signal for "this is exactly the hard case", only the outcome that reveals
it (plan §6 already documents O(n) vs O(n log n) as empirically hard to
separate; this bucket counts how often that specific confusion actually
happens).

Buckets overlap, deliberately: a memoised recursive function that also
calls a library sort belongs in two buckets at once, which is realistic and
reported as such rather than forced into exactly one.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.dataset import ParsedExample

_LIBRARY_SYMBOLS = frozenset(
    {"SORT", "BINARY_SEARCH", "HEAP_PUSH", "HEAP_POP", "QUEUE_OP", "MATH_OP"}
)
_MEMO_LOOKUP_SYMBOLS = frozenset({"HASH_LOOKUP", "HASH_INSERT"})
_AMORTISED_SYMBOLS = frozenset({"LIST_APPEND", "SET_OP", "HASH_INSERT"})
_LOOP_SYMBOLS = frozenset({"LOOP_FOR", "LOOP_WHILE"})

BUCKETS: dict[str, str] = {
    "amortised_structures": (
        "Uses a dynamic-array/set/hash-insert pattern whose real cost is "
        "amortised, not the worst-case per-operation cost"
    ),
    "memoised_recursion": (
        "Recurses with a hash lookup/insert alongside it (a cache), so the "
        "real complexity depends on memoisation, not call-tree shape"
    ),
    "hidden_in_library_call": (
        "Complexity depends on a library call's (SORT/BINARY_SEARCH/HEAP/"
        "QUEUE/MATH_OP) internal cost, invisible in the surrounding structure"
    ),
    "input_dependent_early_exit": (
        "Exits a loop early depending on runtime data (BREAK, or a "
        "branch-guarded return inside a loop), so worst case and typical "
        "case diverge"
    ),
    "on_vs_onlogn_boundary": (
        "True and predicted classes are the adjacent O(n)/O(n log n) pair "
        "-- the boundary plan §6 already documents as empirically hard to "
        "separate"
    ),
}


def _symbols(example: ParsedExample) -> frozenset[str]:
    return frozenset(n.symbol for n in example.ir.nodes)


def _is_memoised_recursion(example: ParsedExample) -> bool:
    """RECURSE alongside either a hash lookup/insert (a dict-based cache) or
    an indexed-assign pattern (an array/DP-table cache) counts. The IR has
    no type inference (`features/tabular.py`'s own documented limitation --
    `cache[n]` and a plain array index are syntactically identical
    `ARRAY_INDEX` nodes), so HASH_LOOKUP/HASH_INSERT alone would miss most
    real Python memoization, which indexes a dict or list the same way
    either kind of cache does. This is a real widening, not just a
    workaround: array/DP-table memoization is memoization too."""
    symbols = _symbols(example)
    if "RECURSE" not in symbols:
        return False
    has_indexed_assign = "ARRAY_INDEX" in symbols and "ASSIGN" in symbols
    return bool(symbols & _MEMO_LOOKUP_SYMBOLS) or has_indexed_assign


def _hides_complexity_in_library_call(example: ParsedExample) -> bool:
    return bool(_symbols(example) & _LIBRARY_SYMBOLS)


def _has_input_dependent_early_exit(example: ParsedExample) -> bool:
    symbols = _symbols(example)
    if "BREAK" in symbols:
        return True
    has_loop = bool(symbols & _LOOP_SYMBOLS)
    return has_loop and "BRANCH" in symbols and "RETURN" in symbols


def _uses_amortised_structure(example: ParsedExample) -> bool:
    return bool(_symbols(example) & _AMORTISED_SYMBOLS)


def _is_on_vs_onlogn_boundary(dimension: str, true_label: str, predicted_label: str) -> bool:
    if dimension != "time":
        return False
    return {true_label, predicted_label} == {"O(n)", "O(n log n)"}


@dataclass(frozen=True, slots=True)
class FailureBucketReport:
    dimension: str
    total_misclassified: int
    counts: dict[str, int]
    example_ids: dict[str, list[str]]  # bucket -> a few "source:solution_id" examples


def analyze_failures(
    dimension: str,
    examples: list[ParsedExample],
    true_labels: list[str],
    predicted_labels: list[str],
    *,
    max_examples_per_bucket: int = 5,
) -> FailureBucketReport:
    counts: dict[str, int] = dict.fromkeys(BUCKETS, 0)
    example_ids: dict[str, list[str]] = {name: [] for name in BUCKETS}
    total = 0
    triples = zip(examples, true_labels, predicted_labels, strict=True)
    for example, true_label, predicted_label in triples:
        if true_label == predicted_label:
            continue
        total += 1
        record_id = f"{example.record.source}:{example.record.solution_id}"
        checks = {
            "amortised_structures": _uses_amortised_structure(example),
            "memoised_recursion": _is_memoised_recursion(example),
            "hidden_in_library_call": _hides_complexity_in_library_call(example),
            "input_dependent_early_exit": _has_input_dependent_early_exit(example),
            "on_vs_onlogn_boundary": _is_on_vs_onlogn_boundary(
                dimension, true_label, predicted_label
            ),
        }
        for name, matched in checks.items():
            if matched:
                counts[name] += 1
                if len(example_ids[name]) < max_examples_per_bucket:
                    example_ids[name].append(record_id)

    return FailureBucketReport(
        dimension=dimension, total_misclassified=total, counts=counts, example_ids=example_ids
    )
