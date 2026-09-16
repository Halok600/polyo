import pytest

from core.taxonomy import (
    SpaceClass,
    TimeClass,
    UnmappableLabelError,
    map_space_label,
    map_time_label,
    ordinal_distance_space,
    ordinal_distance_time,
    space_rank,
    time_rank,
)


def test_time_ranks_are_monotonic_and_unique():
    ranks = [time_rank(c) for c in TimeClass]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_space_ranks_are_monotonic_and_unique():
    ranks = [space_rank(c) for c in SpaceClass]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_ordinal_distance_is_zero_for_identical_class():
    assert ordinal_distance_time(TimeClass.O_1, TimeClass.O_1) == 0
    assert ordinal_distance_space(SpaceClass.O_1, SpaceClass.O_1) == 0


def test_ordinal_distance_grows_with_class_separation():
    near = ordinal_distance_time(TimeClass.O_1, TimeClass.O_N)
    far = ordinal_distance_time(TimeClass.O_1, TimeClass.O_2N)
    assert near < far


def test_space_taxonomy_is_smaller_than_time_taxonomy():
    # Space intentionally has no cubic/exponential class -- see plan SS3.
    assert len(SpaceClass) < len(TimeClass)
    assert not hasattr(SpaceClass, "O_N3")
    assert not hasattr(SpaceClass, "O_2N")


def test_canonical_labels_map_to_themselves():
    assert map_time_label("canonical", "O(n log n)") == TimeClass.O_N_LOG_N
    assert map_space_label("canonical", "O(n)") == SpaceClass.O_N


def test_unknown_source_raises_loudly():
    with pytest.raises(UnmappableLabelError):
        map_time_label("made_up_dataset", "O(n)")


def test_unknown_label_from_known_source_raises_loudly():
    with pytest.raises(UnmappableLabelError):
        map_time_label("canonical", "O(n!)")
