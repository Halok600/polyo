"""Oracle runner: executes a rendered driver for a (spec, solution) pair
across the n grid, in an isolated subprocess (plan §6).

OFFLINE ONLY. This is the one part of the codebase permitted to run
solution code -- see tests/test_isolation.py, which enforces that api/ can
never reach here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from oracle.codegen import render_driver
from oracle.spec import TestSpec

# Hard ceiling for one (spec, solution) sweep across the whole n grid -- a
# malformed or infinite-looping solution must not hang the oracle (plan §16).
_SUBPROCESS_TIMEOUT_S = 30


class OracleRunError(RuntimeError):
    """The driver subprocess failed to produce any usable samples."""


@dataclass(frozen=True, slots=True)
class Sample:
    n: int
    time_ns: int
    peak_bytes: int
    max_call_depth: int


def run(spec: TestSpec, solution_source: str, language: str = "python") -> list[Sample]:
    with tempfile.TemporaryDirectory(prefix="polyo-oracle-") as tmp:
        tmp_dir = Path(tmp)
        solution_path = tmp_dir / "_solution.py"
        solution_path.write_text(solution_source, encoding="utf-8")
        driver_path = tmp_dir / "_driver.py"
        driver_path.write_text(render_driver(language, spec, solution_path), encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(driver_path)],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise OracleRunError(
                f"driver for entrypoint {spec.entrypoint!r} exceeded "
                f"{_SUBPROCESS_TIMEOUT_S}s -- likely an infinite loop"
            ) from e

    samples: list[Sample] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "error" in record:
            continue
        samples.append(
            Sample(
                n=record["n"],
                time_ns=record["time_ns"],
                peak_bytes=record["peak_bytes"],
                max_call_depth=record["max_call_depth"],
            )
        )

    if not samples:
        raise OracleRunError(
            f"driver for entrypoint {spec.entrypoint!r} produced no usable "
            f"samples (exit {proc.returncode}, stderr: {proc.stderr.strip()!r})"
        )
    return samples
