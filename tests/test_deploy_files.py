"""The serving image must contain, and be able to import, everything the API needs.

Phase 5 made `api/predict.py` import the symbolic engine (`analysis/`). The Dockerfile only COPYs
the packages it lists, and `requirements-api.txt` only installs what the slim image budget
allows, so a package the API reaches but the image lacks means the deployed service crashes at
import time while every local test passes (the dev environment has everything). Two checks keep
that from happening:

* the packages the API reaches from its real imports must all be COPYed by the Dockerfile;
* the app must import and answer a request with ONLY the COPYed files and ONLY the packages the
  slim requirements install (and their dependencies): everything else is made unimportable.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_LOCAL = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}


def _imports(path: Path) -> set[str]:
    """Top-level repo packages and repo modules a file imports, as dotted module names."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name.split(".")[0] in _LOCAL}


def _module_path(dotted: str) -> Path | None:
    base = ROOT.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _reachable_packages(entry_package: str) -> set[str]:
    """Top-level packages reachable by import from every module of `entry_package`."""
    seen: set[Path] = set()
    todo = sorted((ROOT / entry_package).glob("*.py"))
    packages: set[str] = {entry_package}
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for dotted in _imports(path):
            packages.add(dotted.split(".")[0])
            target = _module_path(dotted)
            if target is not None:
                todo.append(target)
    return packages


def _copied_by_the_dockerfile() -> set[str]:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    return set(re.findall(r"^COPY\s+(\w+)\s+\./\1\s*$", text, flags=re.MULTILINE))


def test_the_api_reaches_the_symbolic_engine() -> None:
    assert "analysis" in _reachable_packages("api")


def test_the_serving_image_copies_every_package_the_api_can_import() -> None:
    needed = _reachable_packages("api")
    missing = needed - _copied_by_the_dockerfile()
    assert not missing, f"the Dockerfile does not COPY: {sorted(missing)}"


def test_the_serving_image_does_not_ship_what_the_api_never_imports() -> None:
    """Training and evaluation code stays out of the slim image (plan section 13)."""
    shipped = _copied_by_the_dockerfile()
    assert not shipped & {"eval", "oracle", "data", "tests", "web"}


def test_the_calibration_file_ships_with_the_analysis_package() -> None:
    assert (ROOT / "analysis" / "certainty_calibration.json").is_file()


# ------------------------------------------------------------------ the image as it will really run
def _slim_requirements() -> list[str]:
    names = []
    for line in (ROOT / "requirements-api.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def _installed_import_names(distribution: str) -> set[str]:
    files = metadata.distribution(distribution).files or []
    names = {Path(str(f)).parts[0] for f in files if Path(str(f)).suffix in (".py", ".pyd", ".so")}
    return {n.removesuffix(".py").split(".")[0] for n in names if not n.endswith("dist-info")}


def _closure(distributions: list[str]) -> set[str]:
    """The slim requirements and everything they pull in (what `pip install -r` installs)."""
    from packaging.requirements import Requirement

    seen: set[str] = set()
    todo = list(distributions)
    while todo:
        name = todo.pop().lower().replace("_", "-")
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for raw in requires:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                todo.append(requirement.name)
    return seen


def _allowed_third_party() -> set[str]:
    allowed: set[str] = set()
    for distribution in _closure(_slim_requirements()):
        try:
            allowed |= _installed_import_names(distribution)
        except metadata.PackageNotFoundError:
            continue
    return allowed


_SLIM_PROBE = """
import importlib.abc
import sys

ALLOWED = {allowed!r}
STDLIB = set(sys.stdlib_module_names)


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in ALLOWED or top in STDLIB or top.startswith("_"):
            return None
        raise ModuleNotFoundError("No module named " + repr(fullname) + " (not in the image)")


sys.path = [p for p in sys.path if not p.endswith("Leetcoach")]
sys.path.insert(0, {root!r})
sys.meta_path.insert(0, Blocker())

import api.main  # the app itself
from api.models_registry import load_registry
from api.predict import predict

registry = load_registry()
code = "def f(a):\\n    for x in a:\\n        for y in a:\\n            pass\\n"
result = predict(registry, code, "python")
assert result["time"]["class"] == "O(n^2)", result["time"]
print("IMAGE-OK", result["engine"])
"""


def test_the_api_runs_with_only_the_files_and_packages_the_image_has(tmp_path: Path) -> None:
    """Copy exactly what the Dockerfile COPYs, make every package outside the slim requirements (and
    their dependencies) unimportable, then import the app and answer a request. Before this existed
    the image failed at import: separate modules reached torch, lightgbm, grammars the slim
    requirements do not install, and a package the Dockerfile never copied."""
    copied = sorted(_copied_by_the_dockerfile())
    for package in copied:
        shutil.copytree(
            ROOT / package, tmp_path / package, ignore=shutil.ignore_patterns("__pycache__")
        )
    allowed = _allowed_third_party() | set(copied)
    assert {"numpy", "fastapi", "pydantic", "tree_sitter"} <= allowed, sorted(allowed)
    assert not allowed & {"torch", "lightgbm", "sklearn", "scipy", "pandas"}
    script = _SLIM_PROBE.format(allowed=allowed, root=str(tmp_path))
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert "IMAGE-OK" in done.stdout, done.stderr[-3000:]
