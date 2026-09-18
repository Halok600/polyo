#!/usr/bin/env python3
"""Real, non-synthetic, permissively-licensed multi-language source corpus
(plan §7's "scraped public solution repos" -- never built until now, see
PHASE5_REPORT.md's transfer-experiment section for the gap this closes).

**Source and license, checked directly, not assumed.** github.com/TheAlgorithms
org: Python, Java, C-Plus-Plus, C, and Go repos are each individually MIT
licensed (confirmed by reading each repo's own LICENSE file, not inferred
from the org). TheAlgorithms/JavaScript is GPL-3.0 -- deliberately excluded:
GPL's copyleft has real, contested implications for ML training-data
provenance that MIT doesn't, and this project would rather have 5 clean
languages than 6 with one legally murky one. An earlier plan to use Rosetta
Code instead was dropped *before* writing any code, after actually reading
its copyright page: Rosetta Code content is GFDL 1.2, which the page itself
states is "not compatible with most software licenses, including OSI-approved
licenses" -- MIT included. See NOTICE.md.

**Labels are hand-curated, not scraped or inferred from docstrings.** Each
entry's time_class/space_class is the algorithm's own well-established
textbook worst-case complexity -- the same "exact label by construction"
approach data/synth.py already uses for its own corpus, not a program that
parses (often inconsistent, sometimes internally contradictory) inline
complexity comments across thousands of files. Deliberately narrower in
scope than an exhaustive scrape as a result: only algorithms where (a) the
task name essentially forces one canonical implementation shape (sorting/
searching over a single array of length n) and (b) every language's actual
file was individually read to confirm it matches the claimed approach --
not assumed from the filename. A candidate "fibonacci" entry was dropped
after checking: Python's version stores the whole sequence (O(n) space),
Go/C++'s versions use rolling variables (O(1) space), and Java's file
bundles three different techniques with different complexities together --
naming similarity across repos does not imply the same implementation, and
guessing here would mean shipping a wrong label.

Fetches real file content over HTTPS from raw.githubusercontent.com at each
repo's `master` branch -- no local clone needed. Network access required;
this is a manually-run ingestion script, same as data/ingest_codecomplex.py,
never part of CI or the served path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from core.taxonomy import SpaceClass, TimeClass
from data.corpus import CorpusRecord, write_jsonl

SOURCE = "thealgorithms"
_DEFAULT_OUT_PATH = Path("data/processed/thealgorithms.jsonl")

# github.com/TheAlgorithms/<repo> -- every one of these individually
# confirmed MIT-licensed (see module docstring). JavaScript deliberately
# excluded (GPL-3.0).
_REPO_BY_LANGUAGE: dict[str, str] = {
    "python": "Python",
    "java": "Java",
    "cpp": "C-Plus-Plus",
    "c": "C",
    "go": "Go",
}
_RAW_URL = "https://raw.githubusercontent.com/TheAlgorithms/{repo}/master/{path}"


@dataclass(frozen=True, slots=True)
class ScrapedAlgorithm:
    name: str
    time_class: TimeClass
    space_class: SpaceClass
    path_by_language: dict[str, str]


# One canonical worst-case complexity per algorithm, and one specific,
# individually-read file per language -- see module docstring for why this
# is curated rather than scraped/inferred, and why coverage is deliberately
# narrower than exhaustive.
ALGORITHMS: tuple[ScrapedAlgorithm, ...] = (
    ScrapedAlgorithm(
        name="bubble_sort",
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "sorts/bubble_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/BubbleSort.java",
            "cpp": "sorting/bubble_sort.cpp",
            "c": "sorting/bubble_sort.c",
            "go": "sort/bubblesort.go",
        },
    ),
    ScrapedAlgorithm(
        name="insertion_sort",
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "sorts/insertion_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/InsertionSort.java",
            "cpp": "sorting/insertion_sort.cpp",
            "c": "sorting/insertion_sort.c",
            "go": "sort/insertionsort.go",
        },
    ),
    ScrapedAlgorithm(
        name="selection_sort",
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "sorts/selection_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/SelectionSort.java",
            "cpp": "sorting/selection_sort_iterative.cpp",
            "c": "sorting/selection_sort.c",
            "go": "sort/selectionsort.go",
        },
    ),
    ScrapedAlgorithm(
        name="merge_sort",
        time_class=TimeClass.O_N_LOG_N,
        space_class=SpaceClass.O_N,
        path_by_language={
            "python": "sorts/merge_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/MergeSort.java",
            "cpp": "sorting/merge_sort.cpp",
            "c": "sorting/merge_sort.c",
            "go": "sort/mergesort.go",
        },
    ),
    ScrapedAlgorithm(
        name="quick_sort",
        # Worst case (this project's taxonomy is worst-case throughout, see
        # README) -- O(n^2) on an adversarial pivot choice, not the
        # textbook-average O(n log n). Confirmed each file's own docstring
        # states this explicitly (e.g. Java's: "Worst Case: O(n^2)").
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_LOG_N,
        path_by_language={
            "python": "sorts/quick_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/QuickSort.java",
            "cpp": "sorting/quick_sort.cpp",
            "c": "sorting/quick_sort.c",
            "go": "sort/quicksort.go",
        },
    ),
    ScrapedAlgorithm(
        name="heap_sort",
        time_class=TimeClass.O_N_LOG_N,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "sorts/heap_sort.py",
            "java": "src/main/java/com/thealgorithms/sorts/HeapSort.java",
            "cpp": "sorting/heap_sort.cpp",
            "c": "sorting/heap_sort.c",
            "go": "sort/heapsort.go",
        },
    ),
    ScrapedAlgorithm(
        name="binary_search",
        time_class=TimeClass.O_LOG_N,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "searches/binary_search.py",
            "java": "src/main/java/com/thealgorithms/searches/BinarySearch.java",
            "cpp": "search/binary_search.cpp",
            "c": "searching/binary_search.c",
            "go": "search/binary.go",
        },
    ),
    ScrapedAlgorithm(
        name="linear_search",
        time_class=TimeClass.O_N,
        space_class=SpaceClass.O_1,
        path_by_language={
            "python": "searches/linear_search.py",
            "java": "src/main/java/com/thealgorithms/searches/LinearSearch.java",
            "cpp": "search/linear_search.cpp",
            "c": "searching/linear_search.c",
            "go": "search/linear.go",
        },
    ),
)


def _fetch(url: str, timeout: float = 15.0, retries: int = 3) -> str | None:
    """Returns the file's text, None on a real 404, or raises after
    exhausting retries on a transient network error -- a silently-empty
    record would look like a parse failure downstream instead of the
    network problem it actually was."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- fixed https:// raw.githubusercontent.com URLs only
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return None


@dataclass(frozen=True, slots=True)
class ScrapeReport:
    algorithms: int
    languages_attempted: int
    records_written: int
    missing: list[str]


def scrape(out_path: Path = _DEFAULT_OUT_PATH) -> ScrapeReport:
    records: list[CorpusRecord] = []
    missing: list[str] = []
    for algo in ALGORITHMS:
        for language, path in algo.path_by_language.items():
            repo = _REPO_BY_LANGUAGE[language]
            url = _RAW_URL.format(repo=repo, path=path)
            code = _fetch(url)
            if code is None:
                missing.append(f"{algo.name}/{language} ({url})")
                continue
            records.append(
                CorpusRecord(
                    problem_id=algo.name,
                    solution_id=f"{algo.name}_{language}",
                    source=SOURCE,
                    language=language,
                    code=code,
                    time_class=algo.time_class.value,
                    space_class=algo.space_class.value,
                )
            )

    write_jsonl(records, out_path)
    return ScrapeReport(
        algorithms=len(ALGORITHMS),
        languages_attempted=sum(len(a.path_by_language) for a in ALGORITHMS),
        records_written=len(records),
        missing=missing,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    report = scrape(args.out)
    print(json.dumps(asdict(report), indent=2))
    if report.missing:
        print(
            f"warning: {len(report.missing)} file(s) not found -- see 'missing' above",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
