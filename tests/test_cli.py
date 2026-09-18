"""CLI end-to-end test (plan §14 Phase 1: "CLI: file → IR + rule prediction").

Exercises the real subprocess entrypoint deliberately -- this is the one
Phase-1 test meant to catch a wiring mistake between parsing, normalisation,
features and the rule model, not just each piece in isolation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_predicts_linear_time_for_single_loop_python_file(tmp_path: Path):
    src = tmp_path / "solution.py"
    src.write_text("def f(xs):\n    for x in xs:\n        print(x)\n")
    result = _run_cli(str(src))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["language"] == "python"
    assert payload["time"] == "O(n)"
    assert payload["space"] == "O(1)"
    assert payload["ir"]["nodes"] > 0


def test_cli_predicts_quadratic_time_for_nested_loops_cpp_file(tmp_path: Path):
    src = tmp_path / "solution.cpp"
    src.write_text(
        "void f(int n) {\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        for (int j = 0; j < n; j++) {}\n"
        "    }\n"
        "}\n"
    )
    result = _run_cli(str(src))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["language"] == "cpp"
    assert payload["time"] == "O(n^2)"


def test_cli_reports_error_for_unsupported_extension(tmp_path: Path):
    # ".rs" was this test's earlier example of an unsupported extension;
    # Phase 7 added Rust (Tier 3), so this now uses one that's still
    # unmapped.
    src = tmp_path / "solution.rb"
    src.write_text("puts 'hi'\n")
    result = _run_cli(str(src))
    assert result.returncode != 0
    assert "rb" in result.stderr or "language" in result.stderr.lower()


def test_cli_reports_error_for_missing_file():
    result = _run_cli(str(REPO_ROOT / "does_not_exist.py"))
    assert result.returncode != 0
