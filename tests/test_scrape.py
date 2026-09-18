"""Tests for the real, scraped multi-language source (plan §7, `data/scrape.py`).

Network-free by design: `scrape()`'s own network call (`_fetch`) is
monkeypatched rather than hit for real, the same way this project keeps
`oracle/`-dependent work out of the normal test suite (see
tests/test_isolation.py) -- a flaky/slow live-network test in CI would be
exactly the kind of thing plan §9 warns against accepting as "probably
fine." The scraper's actual output was verified for real, once, locally
(all 40 files fetched, all 40 parsed with zero failures) -- see
PHASE5_REPORT.md's corpus section, not repeated here on every CI run.
"""
from __future__ import annotations

from pathlib import Path

import data.scrape as scrape_module
from core.taxonomy import SpaceClass, TimeClass
from data.corpus import read_jsonl
from data.scrape import ALGORITHMS, scrape
from parsing.normalize import normalize_source

# The languages this scraper is wired for (data/scrape.py's own
# _REPO_BY_LANGUAGE) -- kept as a plain literal here rather than importing
# that private mapping, matching tests/test_synth.py's own convention.
_EXPECTED_LANGUAGES = {"python", "java", "cpp", "c", "go"}


def test_every_algorithm_covers_every_served_language():
    for algo in ALGORITHMS:
        assert set(algo.path_by_language) == _EXPECTED_LANGUAGES, algo.name


def test_algorithm_names_are_unique():
    names = [a.name for a in ALGORITHMS]
    assert len(names) == len(set(names))


def test_algorithm_labels_are_valid_taxonomy_values():
    time_values = {c.value for c in TimeClass}
    space_values = {c.value for c in SpaceClass}
    for algo in ALGORITHMS:
        assert algo.time_class.value in time_values
        assert algo.space_class.value in space_values


def test_scrape_writes_one_record_per_algorithm_per_language(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape_module, "_fetch", lambda url, **kwargs: f"// stub for {url}\n")
    out = tmp_path / "out.jsonl"

    report = scrape(out)

    assert report.records_written == len(ALGORITHMS) * len(_EXPECTED_LANGUAGES)
    assert report.missing == []
    records = read_jsonl(out)
    assert len(records) == report.records_written
    assert {r.source for r in records} == {"thealgorithms"}


def test_scrape_reports_missing_files_without_failing(tmp_path, monkeypatch):
    # Each language repo names files differently (bubble_sort.py vs
    # BubbleSort.java vs bubblesort.go), so target exactly one real URL
    # rather than a naming-convention-fragile substring match.
    first = ALGORITHMS[0]
    missing_url = (
        f"https://raw.githubusercontent.com/TheAlgorithms/Python/master/{first.path_by_language['python']}"
    )

    def fake_fetch(url: str, **kwargs: object) -> str | None:
        return None if url == missing_url else f"// stub for {url}\n"

    monkeypatch.setattr(scrape_module, "_fetch", fake_fetch)
    out = tmp_path / "out.jsonl"

    report = scrape(out)

    assert report.missing == [f"{first.name}/python ({missing_url})"]
    assert report.records_written == len(ALGORITHMS) * len(_EXPECTED_LANGUAGES) - 1
    records = read_jsonl(out)
    assert not any(r.problem_id == first.name and r.language == "python" for r in records)


def test_scrape_shares_problem_id_across_languages_per_algorithm(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape_module, "_fetch", lambda url, **kwargs: f"// stub for {url}\n")
    out = tmp_path / "out.jsonl"

    scrape(out)

    records = read_jsonl(out)
    by_algo: dict[str, set[str]] = {}
    for r in records:
        by_algo.setdefault(r.problem_id, set()).add(r.language)
    assert len(by_algo) == len(ALGORITHMS)
    for problem_id, languages in by_algo.items():
        assert languages == _EXPECTED_LANGUAGES, problem_id


def test_real_output_file_parses_with_zero_failures():
    # The actual, already-fetched corpus (not a stub) -- checked here so a
    # future re-scrape that breaks parsing (e.g. a source file changing
    # shape upstream) is caught by CI, not just trusted from the one-time
    # manual run.
    path = Path("data/processed/thealgorithms.jsonl")
    if not path.is_file():
        return  # not fetched in this environment (e.g. a fresh clone) -- nothing to check
    records = read_jsonl(path)
    for r in records:
        ir = normalize_source(r.code, r.language)
        assert ir.symbol_histogram()["FUNC_DEF"] >= 1, f"{r.problem_id}/{r.language}: no FUNC_DEF"
