"""Training-time augmentation: dead-code-padded copies of the training records.

A graph readout that averages over nodes lets a loop be diluted by straight-line code around it
(`eval/POOLING_EXPERIMENT.md`). Padding a record with statements that do nothing teaches the model
that such statements do not change the answer; the label of the copy is the label of the original
because the engine's answer is unchanged by the padding (`tests/test_models_augment.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from data.corpus import CorpusRecord


def pad_with_dead_code(records: Sequence[CorpusRecord], statements: int) -> list[CorpusRecord]:
    """A copy of each record with `statements` do-nothing statements at the top of its first
    function; records the padder cannot handle (an unsupported language, a syntax error) are left
    out. The copy's `solution_id` carries a `+pad<k>` suffix."""
    from eval.perturb import add_dead_code

    out: list[CorpusRecord] = []
    for record in records:
        code = add_dead_code(record.code, record.language, k=statements)
        if code is not None:
            out.append(
                replace(record, code=code, solution_id=f"{record.solution_id}+pad{statements}")
            )
    return out


def augment_with_dead_code(
    records: Sequence[CorpusRecord], counts: Sequence[int] = (10, 40)
) -> list[CorpusRecord]:
    """The records, then one padded copy of each: the i-th record is padded by
    `counts[i % len(counts)]` statements, so every size is seen on every kind of program."""
    out = list(records)
    if not counts:
        return out
    for index, record in enumerate(records):
        out.extend(pad_with_dead_code([record], counts[index % len(counts)]))
    return out
