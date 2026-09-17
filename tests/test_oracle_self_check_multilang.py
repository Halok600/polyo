"""Multi-language oracle self-check (plan §14 Phase 4, §15): the same two
algorithms, run through each language's driver, must recover the same known
complexity as the Python self-check already does (tests/test_oracle_self_check.py).

Deliberately narrower than the Python suite: no naive-recursion case here.
Recursion-stack space (plan §3's requirement that the call stack count) has
no measured signal in these five languages (see oracle/runner.py's
Sample.max_call_depth docstring) -- a recursive case would only demonstrate
a known, already-documented gap, not verify anything.

This dev machine has no C/C++/Java/Go toolchain (plan §0) -- those cases
skip here and run for real in CI via .github/workflows/oracle-label.yml,
which has gcc/g++, a JDK and Go via actions/setup-*. JavaScript runs for
real right here (Node is present locally).
"""
from __future__ import annotations

import shutil

import pytest

from oracle.fit import classify_space, classify_time, fit_space_bytes, fit_space_depth, fit_time
from oracle.runner import run
from oracle.spec import NGrid, ParamDist, ParamSpec, ParamType, TestSpec

_OUT_OF_RANGE_TARGET = 999_999  # never present in the [-1000, 1000] fill

_LINEAR_SCAN_SOURCE = {
    "javascript": """
function linearScan(arr, target) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === target) { return i; }
    }
    return -1;
}
""",
    "java": """
static int linearScan(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) { return i; }
    }
    return -1;
}
""",
    "go": """
func linearScan(arr []int, target int) int {
    for i, x := range arr {
        if x == target {
            return i
        }
    }
    return -1
}
""",
    # C needs the array length as an explicit extra parameter -- see
    # oracle/codegen/c.j2's header comment.
    "c": """
int linear_scan(int arr[], int arr_len, int target) {
    for (int i = 0; i < arr_len; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
    "cpp": """
int linear_scan(int arr[], int arr_len, int target) {
    for (int i = 0; i < arr_len; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
}

_LIBRARY_SORT_SOURCE = {
    # Each of these sorts a COPY and leaves the input untouched, matching
    # Python's `sorted(arr)` (non-mutating) in the existing single-language
    # self-check -- oracle/runner.py's timing loop cycles a small pool of
    # pre-built args across many calls (see oracle/codegen/python.j2's
    # _ARG_POOL_SIZE), so an in-place sort would make later cycles through
    # the same slot progressively cheaper (already-sorted input), biasing
    # the measured curve steeper than the algorithm's real complexity.
    "javascript": """
function libSort(arr) {
    return [...arr].sort((a, b) => a - b);
}
""",
    "java": """
static int[] libSort(int[] arr) {
    int[] copy = arr.clone();
    java.util.Arrays.sort(copy);
    return copy;
}
""",
    "go": """
import "sort"

func libSort(arr []int) []int {
    cp := make([]int, len(arr))
    copy(cp, arr)
    sort.Ints(cp)
    return cp
}
""",
    "c": """
static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}
void lib_sort(int arr[], int arr_len) {
    int *copy = malloc((size_t)arr_len * sizeof(int));
    memcpy(copy, arr, (size_t)arr_len * sizeof(int));
    qsort(copy, (size_t)arr_len, sizeof(int), cmp_int);
    free(copy);
}
""",
    "cpp": """
#include <algorithm>
#include <cstring>
void lib_sort(int arr[], int arr_len) {
    int *copy = (int *)malloc((size_t)arr_len * sizeof(int));
    memcpy(copy, arr, (size_t)arr_len * sizeof(int));
    std::sort(copy, copy + arr_len);
    free(copy);
}
""",
}

_TOOLCHAIN_FOR_LANGUAGE = {
    "javascript": "node",
    "java": "javac",
    "go": "go",
    "c": "gcc",
    "cpp": "g++",
}


def _skip_reason(language: str) -> str | None:
    tool = _TOOLCHAIN_FOR_LANGUAGE[language]
    if shutil.which(tool) is None:
        return f"no {tool} on PATH -- runs in CI (.github/workflows/oracle-label.yml) instead"
    return None


_LANGUAGES = ("javascript", "java", "go", "c", "cpp")
_MEASUREMENT_ATTEMPTS = 3  # see test_oracle_self_check.py's docstring on why


@pytest.mark.parametrize("language", _LANGUAGES)
def test_oracle_recovers_linear_scan_across_languages(language):
    reason = _skip_reason(language)
    if reason:
        pytest.skip(reason)

    entrypoint = "linear_scan" if language in ("c", "cpp") else "linearScan"
    spec = TestSpec(
        entrypoint=entrypoint,
        params=(
            ParamSpec(type=ParamType.INT_ARRAY, size="n"),
            ParamSpec(type=ParamType.INT, dist=ParamDist.CONST, const=_OUT_OF_RANGE_TARGET),
        ),
        n_grid=NGrid(start_exp=8, stop_exp=18),
        budget_ms=5000,
    )

    last_error: AssertionError | None = None
    for _attempt in range(_MEASUREMENT_ATTEMPTS):
        samples = run(spec, _LINEAR_SCAN_SOURCE[language], language=language)
        try:
            time_result = classify_time(fit_time(samples))
            assert time_result is not None, f"{language}: time fit fell below the R^2 floor"
            assert time_result[0].value == "O(n)"
        except AssertionError as e:
            last_error = e
            continue
        return
    raise last_error


@pytest.mark.parametrize("language", _LANGUAGES)
def test_oracle_recovers_library_sort_time_across_languages(language):
    reason = _skip_reason(language)
    if reason:
        pytest.skip(reason)

    entrypoint = "lib_sort" if language in ("c", "cpp") else "libSort"
    spec = TestSpec(
        entrypoint=entrypoint,
        params=(ParamSpec(type=ParamType.INT_ARRAY, size="n"),),
        n_grid=NGrid(start_exp=10, stop_exp=17),
        budget_ms=5000,
    )

    last_error: AssertionError | None = None
    for _attempt in range(_MEASUREMENT_ATTEMPTS):
        samples = run(spec, _LIBRARY_SORT_SOURCE[language], language=language)
        try:
            time_result = classify_time(fit_time(samples))
            assert time_result is not None, f"{language}: time fit fell below the R^2 floor"
            assert time_result[0].value == "O(n log n)"
        except AssertionError as e:
            last_error = e
            continue
        return
    raise last_error


@pytest.mark.parametrize("language", _LANGUAGES)
def test_oracle_reports_no_call_depth_signal_for_non_python_languages(language):
    reason = _skip_reason(language)
    if reason:
        pytest.skip(reason)

    entrypoint = "linear_scan" if language in ("c", "cpp") else "linearScan"
    spec = TestSpec(
        entrypoint=entrypoint,
        params=(
            ParamSpec(type=ParamType.INT_ARRAY, size="n"),
            ParamSpec(type=ParamType.INT, dist=ParamDist.CONST, const=_OUT_OF_RANGE_TARGET),
        ),
        n_grid=NGrid(start_exp=8, stop_exp=12),
        budget_ms=5000,
    )
    samples = run(spec, _LINEAR_SCAN_SOURCE[language], language=language)
    assert all(s.max_call_depth is None for s in samples)
    assert fit_space_depth(samples) is None
    # classify_space must still work from the byte signal alone.
    classify_space(fit_space_bytes(samples), fit_space_depth(samples))
