"""Training-time augmentation (PolyO v2 phase 6): dead-code-padded copies of training records."""

from __future__ import annotations

from analysis.engine import analyze
from data.corpus import CorpusRecord
from models.augment import augment_with_dead_code, pad_with_dead_code

_SUM = "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n"
_PAIRS = (
    "def f(xs):\n    n = 0\n    for x in xs:\n        for y in xs:\n            n += 1\n"
    "    return n\n"
)


def _record(solution_id: str, code: str, language: str = "python") -> CorpusRecord:
    return CorpusRecord("p", solution_id, "s", language, code, "O(n)", "O(1)")


def test_a_padded_copy_is_longer_and_keeps_everything_else() -> None:
    [copy] = pad_with_dead_code([_record("a", _SUM)], 10)
    original = _record("a", _SUM)
    assert len(copy.code.splitlines()) >= len(original.code.splitlines()) + 10
    assert copy.solution_id == "a+pad10"
    assert (copy.problem_id, copy.source, copy.language) == ("p", "s", "python")
    assert (copy.time_class, copy.space_class) == ("O(n)", "O(1)")


def test_padding_does_not_change_what_the_engine_says() -> None:
    """The augmentation is only sound if the padded code has the same complexity as the original:
    the engine is the independent witness."""
    for code in (_SUM, _PAIRS):
        [copy] = pad_with_dead_code([_record("a", code)], 40)
        before, after = analyze(code, "python"), analyze(copy.code, "python")
        assert (after.time.cls, after.space.cls) == (before.time.cls, before.space.cls)


def test_code_that_cannot_be_padded_is_left_out() -> None:
    assert pad_with_dead_code([_record("a", "x", language="cobol")], 10) == []


def test_augmenting_keeps_every_original_and_adds_one_copy_each_cycling_the_sizes() -> None:
    records = [_record(str(i), _SUM) for i in range(4)]
    out = augment_with_dead_code(records, (10, 40))
    assert out[:4] == records
    assert [r.solution_id for r in out[4:]] == ["0+pad10", "1+pad40", "2+pad10", "3+pad40"]


def test_augmenting_nothing_is_nothing() -> None:
    assert augment_with_dead_code([], (10,)) == []
