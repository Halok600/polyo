"""Corpus hygiene (PolyO v2 phase 6): near-duplicates by IR shape and a cap per problem."""

from __future__ import annotations

from data.build import cap_per_problem, dedupe_by_ir, ir_signature
from data.corpus import CorpusRecord

_SUM = "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n"
_SUM_RENAMED = (
    "def total(values):\n    acc = 0\n    for v in values:\n        acc += v\n    return acc\n"
)
_SUM_COMMENTED = (
    "# adds them up\ndef f(xs):\n    t = 0  # running total\n    for x in xs:\n        t += x\n"
    "    return t\n"
)
_PAIRS = (
    "def f(xs):\n    n = 0\n    for x in xs:\n        for y in xs:\n            n += 1\n"
    "    return n\n"
)


def _record(
    solution_id: str,
    code: str,
    problem_id: str = "p",
    time_class: str | None = "O(n)",
    language: str = "python",
) -> CorpusRecord:
    return CorpusRecord(problem_id, solution_id, "s", language, code, time_class, None)


def test_names_and_comments_do_not_change_the_signature() -> None:
    signatures = {
        ir_signature(_record(str(i), code))
        for i, code in enumerate([_SUM, _SUM_RENAMED, _SUM_COMMENTED])
    }
    assert len(signatures) == 1 and None not in signatures


def test_a_different_shape_has_a_different_signature() -> None:
    assert ir_signature(_record("a", _SUM)) != ir_signature(_record("b", _PAIRS))


def test_code_that_does_not_parse_has_no_signature() -> None:
    assert ir_signature(_record("a", "x", language="cobol")) is None


def test_near_duplicates_of_one_problem_collapse_to_one() -> None:
    records = [
        _record("1", _SUM),
        _record("2", _SUM_RENAMED),
        _record("3", _SUM_COMMENTED),
        _record("4", _PAIRS, time_class="O(n^2)"),
    ]
    kept = dedupe_by_ir(records)
    assert len(kept) == 2
    assert {r.time_class for r in kept} == {"O(n)", "O(n^2)"}


def test_the_same_shape_in_two_problems_is_kept_in_both() -> None:
    records = [_record("1", _SUM, problem_id="a"), _record("2", _SUM_RENAMED, problem_id="b")]
    assert len(dedupe_by_ir(records)) == 2


def test_the_same_shape_with_different_labels_is_kept_in_both() -> None:
    """Disagreeing labels on identical code are exactly what a label-noise audit has to see."""
    records = [_record("1", _SUM, time_class="O(n)"), _record("2", _SUM_RENAMED, time_class="O(1)")]
    assert len(dedupe_by_ir(records)) == 2


def test_dedupe_is_deterministic_and_keeps_the_lowest_hash() -> None:
    records = [_record(str(i), _SUM) for i in range(10)]
    first = [r.solution_id for r in dedupe_by_ir(records)]
    again = [r.solution_id for r in dedupe_by_ir(list(reversed(records)))]
    assert first == again and len(first) == 1


def test_unparseable_records_are_never_dropped() -> None:
    records = [_record(str(i), "x", language="cobol") for i in range(3)]
    assert len(dedupe_by_ir(records)) == 3


def test_a_cap_per_problem_keeps_the_same_ones_whatever_the_order() -> None:
    records = [_record(str(i), _SUM, problem_id="big") for i in range(50)]
    records += [_record("solo", _PAIRS, problem_id="small")]
    kept = cap_per_problem(records, 5)
    assert len([r for r in kept if r.problem_id == "big"]) == 5
    assert len([r for r in kept if r.problem_id == "small"]) == 1
    again = cap_per_problem(list(reversed(records)), 5)
    assert {r.solution_id for r in kept} == {r.solution_id for r in again}


def _build(tmp_path, *flags: str):
    """Runs `data.build` over a synthetic corpus (3 problems x 10 solutions of one shape that
    differ only in a name, and one of another shape) into a separate output directory; returns the
    kept records of all splits."""
    from data.build import main
    from data.corpus import read_jsonl, write_jsonl

    source = tmp_path / "corpus"
    records = [
        _record(str(i), _SUM.replace("def f(", f"def f{i}("), problem_id=f"p{i % 3}")
        for i in range(30)
    ]
    records += [_record("q", _PAIRS, problem_id="p0", time_class="O(n^2)")]
    write_jsonl(records, source / "synthetic.jsonl")
    clean = tmp_path / "clean"
    assert main(["--processed-dir", str(source), "--out-dir", str(clean), *flags]) == 0
    assert not list(source.glob("split_*.jsonl")), "the source directory is left alone"
    return [
        record
        for name in ("train", "val", "test")
        for record in read_jsonl(clean / f"split_{name}.jsonl")
    ]


def test_build_writes_the_splits_to_the_output_directory(tmp_path) -> None:
    assert len(_build(tmp_path)) == 31


def test_build_caps_every_problem_when_asked(tmp_path) -> None:
    kept = _build(tmp_path, "--max-per-problem", "2")
    assert len(kept) == 6
    assert {r.problem_id for r in kept} == {"p0", "p1", "p2"}


def test_build_collapses_near_duplicates_when_asked(tmp_path) -> None:
    kept = _build(tmp_path, "--dedupe-ir")
    assert len(kept) == 4  # one per problem, plus the different shape of p0
