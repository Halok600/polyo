"""Canonical complexity taxonomy: classes, ordinal ranks, and mapping from
source-dataset label vocabularies into this project's classes.

Classes are ORDINAL, not categorical -- predicting a neighbouring class is a
better error than predicting a distant one (see plan SS3, SS9). Every model,
loss and metric in this project must respect the `rank` ordering, not just
class identity.

Source-dataset label tables (BigO(Bench), CodeComplex, ...) are populated in
Phase 2/4 ingestion, once the real released files are on disk and their exact
label spellings can be verified rather than guessed. Only the "canonical"
identity mapping -- a class's own string form mapping to itself -- is
pre-registered here.
"""
from __future__ import annotations

from enum import Enum


class TimeClass(str, Enum):
    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n^2)"
    O_N3 = "O(n^3)"
    O_2N = "O(2^n)"


class SpaceClass(str, Enum):
    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n^2)"


_TIME_RANK: dict[TimeClass, int] = {c: i for i, c in enumerate(TimeClass)}
_SPACE_RANK: dict[SpaceClass, int] = {c: i for i, c in enumerate(SpaceClass)}


def time_rank(c: TimeClass) -> int:
    return _TIME_RANK[c]


def space_rank(c: SpaceClass) -> int:
    return _SPACE_RANK[c]


def ordinal_distance_time(a: TimeClass, b: TimeClass) -> int:
    return abs(_TIME_RANK[a] - _TIME_RANK[b])


def ordinal_distance_space(a: SpaceClass, b: SpaceClass) -> int:
    return abs(_SPACE_RANK[a] - _SPACE_RANK[b])


class UnmappableLabelError(ValueError):
    """A source dataset's label has no registered mapping to our taxonomy.

    Raised, never swallowed: a silently-dropped label is a silent data leak,
    and this project would rather fail ingestion loudly than train on a
    mis-mapped class.
    """


_SOURCE_TIME_MAPS: dict[str, dict[str, TimeClass]] = {
    "canonical": {c.value: c for c in TimeClass},
}

_SOURCE_SPACE_MAPS: dict[str, dict[str, SpaceClass]] = {
    "canonical": {c.value: c for c in SpaceClass},
}


def register_time_source(source: str, table: dict[str, TimeClass]) -> None:
    """Register (or extend) a source dataset's time-label vocabulary.

    Called during ingestion (Phase 2/4) once a dataset's real label strings
    are known. Kept as an explicit registration step, rather than a guessed
    table hardcoded here, so every mapping is verified against real data
    before it can affect training.
    """
    _SOURCE_TIME_MAPS.setdefault(source, {}).update(table)


def register_space_source(source: str, table: dict[str, SpaceClass]) -> None:
    _SOURCE_SPACE_MAPS.setdefault(source, {}).update(table)


def map_time_label(source: str, raw_label: str) -> TimeClass:
    try:
        table = _SOURCE_TIME_MAPS[source]
    except KeyError as e:
        raise UnmappableLabelError(f"unknown time-label source: {source!r}") from e
    try:
        return table[raw_label]
    except KeyError as e:
        raise UnmappableLabelError(
            f"{source} time label {raw_label!r} has no mapping to our taxonomy"
        ) from e


def map_space_label(source: str, raw_label: str) -> SpaceClass:
    try:
        table = _SOURCE_SPACE_MAPS[source]
    except KeyError as e:
        raise UnmappableLabelError(f"unknown space-label source: {source!r}") from e
    try:
        return table[raw_label]
    except KeyError as e:
        raise UnmappableLabelError(
            f"{source} space label {raw_label!r} has no mapping to our taxonomy"
        ) from e
