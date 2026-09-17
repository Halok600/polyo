"""Oracle runner: executes a rendered driver for a (spec, solution) pair
across the n grid, in an isolated subprocess (plan §6).

OFFLINE ONLY. This is the one part of the codebase permitted to run
solution code -- see tests/test_isolation.py, which enforces that api/ can
never reach here.

Phase 2 shipped Python only, loading a separate solution file dynamically by
path. Phase 4 adds Java, JavaScript, C, C++ and Go, each needing a bit of
runner-owned glue to turn a bare, raw solution snippet (the same "just the
function" shape as the Python driver already assumes) into something that
language's compiler/runtime can call -- Java's method wrapped in
`class Solution`, Go's function given a `package main` clause, JavaScript's
function given a `module.exports` line, and C/C++'s translation-unit-order
requirement satisfied by concatenating the solution directly above the
driver rather than compiling separate files. See each codegen template's own
header comment for the per-language contract this implies for solution
authors (data/synth.py, ingestion scripts).
"""
from __future__ import annotations

import json
import os
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
# Separate, more generous ceiling for compiling a driver (C/C++/Java/Go) --
# unrelated to the solution's own runtime behaviour, so it gets its own
# budget rather than competing with the run-time ceiling above.
_COMPILE_TIMEOUT_S = 60

_INTERPOSER_SOURCE = Path(__file__).parent / "interposer.c"


class OracleRunError(RuntimeError):
    """The driver subprocess failed to produce any usable samples."""


@dataclass(frozen=True, slots=True)
class Sample:
    n: int
    time_ns: int
    peak_bytes: int
    # None for every non-Python language (plan §14 Phase 4): the recursion-
    # stack signal this project's Python driver measures via `sys.settrace`
    # has no equally reliable, CI-verifiable equivalent across C/C++, Java,
    # JavaScript and Go's very different runtimes (native thread stack, JVM
    # stack, V8 stack, goroutine stack) without per-language instrumentation
    # this dev machine cannot test locally -- see oracle/README.md. Reporting
    # a fake constant here (e.g. 1) would silently misclassify a recursive
    # solution as O(1) space, reproducing the exact bug this signal was
    # built to catch -- so it is left unmeasured and honestly propagated as
    # None (`fit.fit_space_depth` returns None for it) rather than guessed.
    max_call_depth: int | None


def _wrap_java_solution(source: str) -> str:
    return f"class Solution {{\n{source}\n}}\n"


def _wrap_go_solution(source: str) -> str:
    return f"package main\n\n{source}\n"


def _wrap_javascript_solution(entrypoint: str, source: str) -> str:
    return (
        f"{source}\n"
        f"if (typeof {entrypoint} !== 'undefined') {{ "
        f"module.exports.{entrypoint} = {entrypoint}; }}\n"
    )


def _run_subprocess(
    cmd: list[str], spec: TestSpec, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S, env=env
        )
    except subprocess.TimeoutExpired as e:
        raise OracleRunError(
            f"driver for entrypoint {spec.entrypoint!r} exceeded "
            f"{_SUBPROCESS_TIMEOUT_S}s -- likely an infinite loop"
        ) from e


def _compile(cmd: list[str], spec: TestSpec, *, what: str) -> None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        raise OracleRunError(
            f"compiling {what} for entrypoint {spec.entrypoint!r} exceeded "
            f"{_COMPILE_TIMEOUT_S}s"
        ) from e
    if proc.returncode != 0:
        raise OracleRunError(
            f"compiling {what} for entrypoint {spec.entrypoint!r} failed "
            f"(exit {proc.returncode}): {proc.stderr.strip()!r}"
        )


def _ensure_interposer_built() -> Path:
    """Builds oracle/interposer.c into a shared library once, cached across
    calls (and across process runs, via a stable tempdir path) rather than
    recompiling it for every C/C++ sample -- see interposer.c's own
    docstring for what it does and why it must be LD_PRELOAD-ed, not linked
    directly."""
    cache_dir = Path(tempfile.gettempdir()) / "polyo-oracle-interposer"
    cache_dir.mkdir(exist_ok=True)
    so_path = cache_dir / "interposer.so"
    if so_path.exists() and so_path.stat().st_mtime >= _INTERPOSER_SOURCE.stat().st_mtime:
        return so_path
    proc = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_path), str(_INTERPOSER_SOURCE), "-ldl"],
        capture_output=True,
        text=True,
        timeout=_COMPILE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise OracleRunError(f"failed to build oracle/interposer.c: {proc.stderr.strip()!r}")
    return so_path


def _run_python(
    tmp_dir: Path, spec: TestSpec, solution_source: str
) -> subprocess.CompletedProcess[str]:
    solution_path = tmp_dir / "_solution.py"
    solution_path.write_text(solution_source, encoding="utf-8")
    driver_path = tmp_dir / "_driver.py"
    driver_path.write_text(render_driver("python", spec, solution_path), encoding="utf-8")
    return _run_subprocess([sys.executable, str(driver_path)], spec)


def _run_javascript(
    tmp_dir: Path, spec: TestSpec, solution_source: str
) -> subprocess.CompletedProcess[str]:
    solution_path = tmp_dir / "_solution.js"
    solution_path.write_text(
        _wrap_javascript_solution(spec.entrypoint, solution_source), encoding="utf-8"
    )
    driver_path = tmp_dir / "_driver.js"
    driver_path.write_text(render_driver("javascript", spec, solution_path), encoding="utf-8")
    return _run_subprocess(["node", "--expose-gc", str(driver_path)], spec)


def _run_c_like(
    language: str, tmp_dir: Path, spec: TestSpec, solution_source: str
) -> subprocess.CompletedProcess[str]:
    ext = "c" if language == "c" else "cpp"
    compiler = "gcc" if language == "c" else "g++"
    combined_path = tmp_dir / f"_driver.{ext}"
    # solution_source is concatenated directly above the driver body rather
    # than compiled as a separate translation unit -- see c.j2/cpp.j2's
    # header comment for why (declare-before-use, no prototype synthesis
    # needed). `solution_path` is unused by these two templates.
    driver_body = render_driver(language, spec, combined_path)
    combined_path.write_text(solution_source + "\n\n" + driver_body, encoding="utf-8")

    interposer_path = _ensure_interposer_built()
    binary_path = tmp_dir / "driver_bin"
    _compile(
        [compiler, "-O2", str(combined_path), "-ldl", "-o", str(binary_path)],
        spec,
        what="driver",
    )

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(interposer_path)
    return _run_subprocess([str(binary_path)], spec, env=env)


def _run_java(
    tmp_dir: Path, spec: TestSpec, solution_source: str
) -> subprocess.CompletedProcess[str]:
    solution_path = tmp_dir / "Solution.java"
    solution_path.write_text(_wrap_java_solution(solution_source), encoding="utf-8")
    driver_path = tmp_dir / "Driver.java"
    driver_path.write_text(render_driver("java", spec, solution_path), encoding="utf-8")

    _compile(
        ["javac", "-d", str(tmp_dir), str(solution_path), str(driver_path)],
        spec,
        what="driver",
    )
    return _run_subprocess(["java", "-cp", str(tmp_dir), "Driver"], spec)


def _run_go(
    tmp_dir: Path, spec: TestSpec, solution_source: str
) -> subprocess.CompletedProcess[str]:
    solution_path = tmp_dir / "_solution.go"
    solution_path.write_text(_wrap_go_solution(solution_source), encoding="utf-8")
    driver_path = tmp_dir / "_driver.go"
    driver_path.write_text(render_driver("go", spec, solution_path), encoding="utf-8")
    # `go run` compiles and executes multiple explicitly-listed files as one
    # ad-hoc main package -- no go.mod needed for stdlib-only solutions.
    return _run_subprocess(["go", "run", str(solution_path), str(driver_path)], spec)


_EXEC_BY_LANGUAGE = {
    "python": _run_python,
    "javascript": _run_javascript,
    "java": _run_java,
    "go": _run_go,
}


def run(spec: TestSpec, solution_source: str, language: str = "python") -> list[Sample]:
    with tempfile.TemporaryDirectory(prefix="polyo-oracle-") as tmp:
        tmp_dir = Path(tmp)
        if language in ("c", "cpp"):
            proc = _run_c_like(language, tmp_dir, spec, solution_source)
        else:
            try:
                exec_fn = _EXEC_BY_LANGUAGE[language]
            except KeyError as e:
                raise OracleRunError(
                    f"no oracle execution strategy for language: {language!r}"
                ) from e
            proc = exec_fn(tmp_dir, spec, solution_source)

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
