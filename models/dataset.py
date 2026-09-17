"""Shared corpus -> parsed-example pipeline for rung 1/2 training (plan §8).

Both models start from the same step -- parse each corpus record's code
into an IR graph -- and diverge only in what they read off it (rung 1: the
IR symbol sequence; rung 2: `features.tabular`'s feature vector). This
module owns that shared, failure-tolerant parsing step so it isn't
duplicated between `models/tfidf.py` and `models/gbdt.py`.

Real, arbitrary competitive-programming solutions (plan §7) occasionally
trip up a parser or the feature walker in ways Phase 1's small hand-written
examples never did. Catching broadly here (not narrowly) is deliberate: this
is a training-data ETL boundary processing thousands of untrusted,
heterogeneous real-world inputs, not application logic with known-safe
inputs, so one malformed example must not fail an entire training run.
`DatasetStats` reports exactly how many were dropped and the exception types
that caused it, so that's never silent.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field

from core.ir import IRGraph
from data.corpus import CorpusRecord
from features.tabular import TabularFeatures, extract_features
from parsing.normalize import normalize_source
from parsing.parse import UnsupportedLanguageError

# Real solutions occasionally chain hundreds of binary operators or nest
# deeply enough to exceed Python's default recursion limit while walking the
# CST (parsing/normalize.py) or the (now-iterative, but IR construction
# itself still recurses in a couple of spots) parser internals. Raised only
# for the duration of dataset construction, not process-wide.
_RECURSION_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class ParsedExample:
    record: CorpusRecord
    ir: IRGraph
    features: TabularFeatures


@dataclass(slots=True)
class DatasetStats:
    total: int = 0
    parsed: int = 0
    unsupported_language: int = 0
    failure_types: Counter[str] = field(default_factory=Counter)

    @property
    def failed(self) -> int:
        return sum(self.failure_types.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "parsed": self.parsed,
            "unsupported_language": self.unsupported_language,
            "failed": self.failed,
            "failure_types": dict(self.failure_types),
        }


def build_examples(records: list[CorpusRecord]) -> tuple[list[ParsedExample], DatasetStats]:
    stats = DatasetStats(total=len(records))
    examples: list[ParsedExample] = []

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, _RECURSION_LIMIT))
    try:
        for record in records:
            try:
                ir = normalize_source(record.code, record.language)
                features = extract_features(ir)
            except UnsupportedLanguageError:
                stats.unsupported_language += 1
                continue
            except Exception as exc:  # broad on purpose -- see module docstring
                stats.failure_types[type(exc).__name__] += 1
                continue
            examples.append(ParsedExample(record=record, ir=ir, features=features))
            stats.parsed += 1
    finally:
        sys.setrecursionlimit(previous_limit)

    return examples, stats


def time_labels(examples: list[ParsedExample]) -> list[str | None]:
    return [e.record.time_class for e in examples]


def space_labels(examples: list[ParsedExample]) -> list[str | None]:
    return [e.record.space_class for e in examples]


def with_label(
    examples: list[ParsedExample], labels: list[str | None]
) -> tuple[list[ParsedExample], list[str]]:
    """Filters out examples whose label for this dimension is None (that
    source doesn't label it, or the raw label didn't map to our taxonomy --
    see data/ingest_*.py) -- a model can't train or be scored on those."""
    kept_examples = []
    kept_labels = []
    for example, label in zip(examples, labels, strict=True):
        if label is not None:
            kept_examples.append(example)
            kept_labels.append(label)
    return kept_examples, kept_labels
