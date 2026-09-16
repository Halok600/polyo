"""Enforces the project's central safety invariant: the deployed API never
executes untrusted code. See plan SS10.

`oracle/` runs code, offline, only during dataset labelling. `api/` serves
live traffic. This test makes the boundary between them a CI failure, not a
matter of discipline, and it must keep passing as both packages grow.
"""
from __future__ import annotations

import ast
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1] / "api"
FORBIDDEN_CALLS = {"exec", "eval", "compile"}
FORBIDDEN_IMPORT_ROOTS = {"subprocess", "oracle"}


def _api_py_files() -> list[Path]:
    return sorted(API_DIR.rglob("*.py"))


def test_api_has_source_files_to_check():
    # Guards against this test silently passing over an empty/misnamed dir.
    assert _api_py_files(), "expected at least one .py file under api/"


def test_api_never_imports_oracle_or_subprocess():
    for path in _api_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_IMPORT_ROOTS, (
                        f"{path} imports {alias.name!r} -- api/ must never "
                        "reach oracle/ or subprocess"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in FORBIDDEN_IMPORT_ROOTS, (
                    f"{path} imports from {node.module!r} -- api/ must "
                    "never reach oracle/ or subprocess"
                )


def test_api_contains_no_dynamic_code_execution():
    for path in _api_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, (
                    f"{path} calls {node.func.id}() -- api/ must never "
                    "execute code dynamically"
                )
