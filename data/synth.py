#!/usr/bin/env python3
"""Parallel synthetic generator (plan §7, §14 Phase 4): one algorithm, the
same shape in all six languages, with an EXACT label by construction --
no oracle execution involved, unlike every other ingestion script. This is
what the plan calls the highest-leverage data source: it fills the rare
classes real corpora barely contain (O(n^3), O(2^n) time; O(n log n) space)
and gives a perfectly aligned cross-language example set for Phase 5's
transfer experiment, since every language variant of one algorithm shares a
`problem_id`.

Each algorithm's code is written to look like a normal, complete, parseable
solution in its language (a Java entry includes its `class Solution { ... }`
wrapper, a Go entry its `package main`, etc.) -- unlike oracle/runner.py's
bare-snippet contract for solutions it executes and wraps itself, this
corpus's `code` field is meant to be parsed by parsing/normalize.py exactly
as a real scraped solution would be, and is never run.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from core.taxonomy import SpaceClass, TimeClass
from data.corpus import CorpusRecord, write_jsonl

SOURCE = "synth"
_DEFAULT_OUT_PATH = Path("data/processed/synth.jsonl")


@dataclass(frozen=True, slots=True)
class SynthAlgorithm:
    name: str
    time_class: TimeClass
    space_class: SpaceClass | None
    code_by_language: dict[str, str]


ALGORITHMS: tuple[SynthAlgorithm, ...] = (
    SynthAlgorithm(
        name="constant_access",
        time_class=TimeClass.O_1,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def constant_access(arr):
    if not arr:
        return 0
    return arr[0] + arr[-1]
""",
            "cpp": """\
int constant_access(int arr[], int n) {
    if (n == 0) {
        return 0;
    }
    return arr[0] + arr[n - 1];
}
""",
            "c": """\
int constant_access(int arr[], int n) {
    if (n == 0) {
        return 0;
    }
    return arr[0] + arr[n - 1];
}
""",
            "java": """\
class Solution {
    static int constantAccess(int[] arr) {
        if (arr.length == 0) {
            return 0;
        }
        return arr[0] + arr[arr.length - 1];
    }
}
""",
            "javascript": """\
function constantAccess(arr) {
    if (arr.length === 0) {
        return 0;
    }
    return arr[0] + arr[arr.length - 1];
}
""",
            "go": """\
package main

func constantAccess(arr []int) int {
    if len(arr) == 0 {
        return 0
    }
    return arr[0] + arr[len(arr)-1]
}
""",
        },
    ),
    SynthAlgorithm(
        name="binary_search",
        time_class=TimeClass.O_LOG_N,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
            "cpp": """\
int binary_search(int arr[], int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}
""",
            "c": """\
int binary_search(int arr[], int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}
""",
            "java": """\
class Solution {
    static int binarySearch(int[] arr, int target) {
        int lo = 0, hi = arr.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (arr[mid] == target) {
                return mid;
            }
            if (arr[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }
}
""",
            "javascript": """\
function binarySearch(arr, target) {
    let lo = 0, hi = arr.length - 1;
    while (lo <= hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (arr[mid] === target) {
            return mid;
        }
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}
""",
            "go": """\
package main

func binarySearch(arr []int, target int) int {
    lo, hi := 0, len(arr)-1
    for lo <= hi {
        mid := (lo + hi) / 2
        if arr[mid] == target {
            return mid
        }
        if arr[mid] < target {
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    return -1
}
""",
        },
    ),
    SynthAlgorithm(
        name="linear_scan",
        time_class=TimeClass.O_N,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def linear_scan(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1
""",
            "cpp": """\
int linear_scan(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
            "c": """\
int linear_scan(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
            "java": """\
class Solution {
    static int linearScan(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
""",
            "javascript": """\
function linearScan(arr, target) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === target) {
            return i;
        }
    }
    return -1;
}
""",
            "go": """\
package main

func linearScan(arr []int, target int) int {
    for i, x := range arr {
        if x == target {
            return i
        }
    }
    return -1
}
""",
        },
    ),
    SynthAlgorithm(
        name="sort_then_return",
        time_class=TimeClass.O_N_LOG_N,
        space_class=SpaceClass.O_N,
        code_by_language={
            "python": """\
def sort_then_return(arr):
    return sorted(arr)
""",
            "cpp": """\
#include <algorithm>
#include <cstdlib>
#include <cstring>
int *sort_then_return(int arr[], int n) {
    int *copy = (int *)malloc(sizeof(int) * n);
    memcpy(copy, arr, sizeof(int) * n);
    std::sort(copy, copy + n);
    return copy;
}
""",
            "c": """\
#include <stdlib.h>
#include <string.h>
static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}
int *sort_then_return(int arr[], int n) {
    int *copy = malloc(sizeof(int) * (size_t)n);
    memcpy(copy, arr, sizeof(int) * (size_t)n);
    qsort(copy, (size_t)n, sizeof(int), cmp_int);
    return copy;
}
""",
            "java": """\
class Solution {
    static int[] sortThenReturn(int[] arr) {
        int[] copy = arr.clone();
        java.util.Arrays.sort(copy);
        return copy;
    }
}
""",
            "javascript": """\
function sortThenReturn(arr) {
    return [...arr].sort((a, b) => a - b);
}
""",
            "go": """\
package main

import "sort"

func sortThenReturn(arr []int) []int {
    cp := make([]int, len(arr))
    copy(cp, arr)
    sort.Ints(cp)
    return cp
}
""",
        },
    ),
    SynthAlgorithm(
        name="level_buffers",
        time_class=TimeClass.O_N_LOG_N,
        space_class=SpaceClass.O_N_LOG_N,
        code_by_language={
            "python": """\
import math


def level_buffers(n):
    levels = max(1, int(math.log2(max(n, 2))) + 1)
    buffers = []
    for _ in range(levels):
        buffers.append([0] * n)
    return buffers
""",
            "cpp": """\
#include <cmath>
#include <vector>
std::vector<std::vector<int>> level_buffers(int n) {
    int levels = (int)(std::log2((double)(n > 2 ? n : 2))) + 1;
    std::vector<std::vector<int>> buffers;
    for (int level = 0; level < levels; level++) {
        buffers.push_back(std::vector<int>(n, 0));
    }
    return buffers;
}
""",
            "c": """\
#include <math.h>
#include <stdlib.h>
int **level_buffers(int n, int *out_levels) {
    int levels = (int)(log2((double)(n > 2 ? n : 2))) + 1;
    int **buffers = malloc(sizeof(int *) * (size_t)levels);
    for (int level = 0; level < levels; level++) {
        buffers[level] = calloc((size_t)n, sizeof(int));
    }
    *out_levels = levels;
    return buffers;
}
""",
            "java": """\
class Solution {
    static int[][] levelBuffers(int n) {
        int levels = (int) (Math.log(Math.max(n, 2)) / Math.log(2)) + 1;
        int[][] buffers = new int[levels][];
        for (int level = 0; level < levels; level++) {
            buffers[level] = new int[n];
        }
        return buffers;
    }
}
""",
            "javascript": """\
function levelBuffers(n) {
    const levels = Math.floor(Math.log2(Math.max(n, 2))) + 1;
    const buffers = [];
    for (let level = 0; level < levels; level++) {
        buffers.push(new Array(n).fill(0));
    }
    return buffers;
}
""",
            "go": """\
package main

import "math"

func levelBuffers(n int) [][]int {
    nf := float64(n)
    if n < 2 {
        nf = 2
    }
    levels := int(math.Log2(nf)) + 1
    buffers := make([][]int, levels)
    for level := 0; level < levels; level++ {
        buffers[level] = make([]int, n)
    }
    return buffers
}
""",
        },
    ),
    SynthAlgorithm(
        name="pairwise_sum",
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def pairwise_sum(arr):
    total = 0
    n = len(arr)
    for i in range(n):
        for j in range(n):
            total += arr[i] + arr[j]
    return total
""",
            "cpp": """\
long pairwise_sum(int arr[], int n) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total += arr[i] + arr[j];
        }
    }
    return total;
}
""",
            "c": """\
long pairwise_sum(int arr[], int n) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total += arr[i] + arr[j];
        }
    }
    return total;
}
""",
            "java": """\
class Solution {
    static long pairwiseSum(int[] arr) {
        long total = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                total += arr[i] + arr[j];
            }
        }
        return total;
    }
}
""",
            "javascript": """\
function pairwiseSum(arr) {
    let total = 0;
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            total += arr[i] + arr[j];
        }
    }
    return total;
}
""",
            "go": """\
package main

func pairwiseSum(arr []int) int64 {
    var total int64
    for i := 0; i < len(arr); i++ {
        for j := 0; j < len(arr); j++ {
            total += int64(arr[i] + arr[j])
        }
    }
    return total
}
""",
        },
    ),
    SynthAlgorithm(
        name="build_matrix",
        time_class=TimeClass.O_N2,
        space_class=SpaceClass.O_N2,
        code_by_language={
            "python": """\
def build_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(i * n + j)
        matrix.append(row)
    return matrix
""",
            "cpp": """\
#include <vector>
std::vector<std::vector<int>> build_matrix(int n) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i * n + j;
        }
    }
    return matrix;
}
""",
            "c": """\
#include <stdlib.h>
int **build_matrix(int n) {
    int **matrix = malloc(sizeof(int *) * (size_t)n);
    for (int i = 0; i < n; i++) {
        matrix[i] = malloc(sizeof(int) * (size_t)n);
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i * n + j;
        }
    }
    return matrix;
}
""",
            "java": """\
class Solution {
    static int[][] buildMatrix(int n) {
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = i * n + j;
            }
        }
        return matrix;
    }
}
""",
            "javascript": """\
function buildMatrix(n) {
    const matrix = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            row.push(i * n + j);
        }
        matrix.push(row);
    }
    return matrix;
}
""",
            "go": """\
package main

func buildMatrix(n int) [][]int {
    matrix := make([][]int, n)
    for i := 0; i < n; i++ {
        matrix[i] = make([]int, n)
        for j := 0; j < n; j++ {
            matrix[i][j] = i*n + j
        }
    }
    return matrix
}
""",
        },
    ),
    SynthAlgorithm(
        name="triple_nested",
        time_class=TimeClass.O_N3,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def triple_nested(arr):
    total = 0
    n = len(arr)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += arr[i] + arr[j] + arr[k]
    return total
""",
            "cpp": """\
long triple_nested(int arr[], int n) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                total += arr[i] + arr[j] + arr[k];
            }
        }
    }
    return total;
}
""",
            "c": """\
long triple_nested(int arr[], int n) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                total += arr[i] + arr[j] + arr[k];
            }
        }
    }
    return total;
}
""",
            "java": """\
class Solution {
    static long tripleNested(int[] arr) {
        long total = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                for (int k = 0; k < arr.length; k++) {
                    total += arr[i] + arr[j] + arr[k];
                }
            }
        }
        return total;
    }
}
""",
            "javascript": """\
function tripleNested(arr) {
    let total = 0;
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            for (let k = 0; k < arr.length; k++) {
                total += arr[i] + arr[j] + arr[k];
            }
        }
    }
    return total;
}
""",
            "go": """\
package main

func tripleNested(arr []int) int64 {
    var total int64
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            for k := 0; k < n; k++ {
                total += int64(arr[i] + arr[j] + arr[k])
            }
        }
    }
    return total
}
""",
        },
    ),
    SynthAlgorithm(
        name="naive_fibonacci",
        time_class=TimeClass.O_2N,
        space_class=SpaceClass.O_N,
        code_by_language={
            "python": """\
def naive_fibonacci(n):
    if n <= 1:
        return n
    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2)
""",
            "cpp": """\
int naive_fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2);
}
""",
            "c": """\
int naive_fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2);
}
""",
            "java": """\
class Solution {
    static int naiveFibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return naiveFibonacci(n - 1) + naiveFibonacci(n - 2);
    }
}
""",
            "javascript": """\
function naiveFibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return naiveFibonacci(n - 1) + naiveFibonacci(n - 2);
}
""",
            "go": """\
package main

func naiveFibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return naiveFibonacci(n-1) + naiveFibonacci(n-2)
}
""",
        },
    ),
    # --- Phase-5 prep: expanded for cross-language transfer signal (plan
    # §7/§9). Deliberately built only from IR symbols already consistently
    # mapped across all six lang/*.toml files (LOOP, ARRAY_ALLOC, ARRAY_INDEX,
    # RECURSE, BRANCH) -- a hash-based example was considered and dropped
    # after checking the real mapping tables: Python only maps dict/set
    # *literals* (not .add()/set()/dict.get() calls) to HASH_*, C/C++/Go have
    # no hash-container mapping at all yet (object_creation_expression in
    # java.toml and new_expression in cpp.toml both fall through to the
    # generic ARRAY_ALLOC), so a hash example today would silently diverge
    # across languages rather than align. Closing that mapping gap is
    # IR-design work for a future phase, not a training-data addition.
    SynthAlgorithm(
        name="recursive_sum",
        time_class=TimeClass.O_N,
        space_class=SpaceClass.O_N,
        code_by_language={
            "python": """\
def recursive_sum(arr, i):
    if i >= len(arr):
        return 0
    return arr[i] + recursive_sum(arr, i + 1)
""",
            "cpp": """\
long recursive_sum(int arr[], int n, int i) {
    if (i >= n) {
        return 0;
    }
    return arr[i] + recursive_sum(arr, n, i + 1);
}
""",
            "c": """\
long recursive_sum(int arr[], int n, int i) {
    if (i >= n) {
        return 0;
    }
    return arr[i] + recursive_sum(arr, n, i + 1);
}
""",
            "java": """\
class Solution {
    static long recursiveSum(int[] arr, int i) {
        if (i >= arr.length) {
            return 0;
        }
        return arr[i] + recursiveSum(arr, i + 1);
    }
}
""",
            "javascript": """\
function recursiveSum(arr, i) {
    if (i >= arr.length) {
        return 0;
    }
    return arr[i] + recursiveSum(arr, i + 1);
}
""",
            "go": """\
package main

func recursiveSum(arr []int, i int) int64 {
    if i >= len(arr) {
        return 0
    }
    return int64(arr[i]) + recursiveSum(arr, i+1)
}
""",
        },
    ),
    SynthAlgorithm(
        name="binary_search_recursive",
        time_class=TimeClass.O_LOG_N,
        space_class=SpaceClass.O_LOG_N,
        code_by_language={
            "python": """\
def binary_search_recursive(arr, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi)
    return binary_search_recursive(arr, target, lo, mid - 1)
""",
            "cpp": """\
int binary_search_recursive(int arr[], int target, int lo, int hi) {
    if (lo > hi) {
        return -1;
    }
    int mid = (lo + hi) / 2;
    if (arr[mid] == target) {
        return mid;
    }
    if (arr[mid] < target) {
        return binary_search_recursive(arr, target, mid + 1, hi);
    }
    return binary_search_recursive(arr, target, lo, mid - 1);
}
""",
            "c": """\
int binary_search_recursive(int arr[], int target, int lo, int hi) {
    if (lo > hi) {
        return -1;
    }
    int mid = (lo + hi) / 2;
    if (arr[mid] == target) {
        return mid;
    }
    if (arr[mid] < target) {
        return binary_search_recursive(arr, target, mid + 1, hi);
    }
    return binary_search_recursive(arr, target, lo, mid - 1);
}
""",
            "java": """\
class Solution {
    static int binarySearchRecursive(int[] arr, int target, int lo, int hi) {
        if (lo > hi) {
            return -1;
        }
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            return binarySearchRecursive(arr, target, mid + 1, hi);
        }
        return binarySearchRecursive(arr, target, lo, mid - 1);
    }
}
""",
            "javascript": """\
function binarySearchRecursive(arr, target, lo, hi) {
    if (lo > hi) {
        return -1;
    }
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) {
        return mid;
    }
    if (arr[mid] < target) {
        return binarySearchRecursive(arr, target, mid + 1, hi);
    }
    return binarySearchRecursive(arr, target, lo, mid - 1);
}
""",
            "go": """\
package main

func binarySearchRecursive(arr []int, target int, lo int, hi int) int {
    if lo > hi {
        return -1
    }
    mid := (lo + hi) / 2
    if arr[mid] == target {
        return mid
    }
    if arr[mid] < target {
        return binarySearchRecursive(arr, target, mid+1, hi)
    }
    return binarySearchRecursive(arr, target, lo, mid-1)
}
""",
        },
    ),
    SynthAlgorithm(
        name="halving_loop",
        time_class=TimeClass.O_LOG_N,
        space_class=SpaceClass.O_1,
        code_by_language={
            "python": """\
def halving_loop(n):
    count = 0
    while n > 0:
        n = n // 2
        count += 1
    return count
""",
            "cpp": """\
int halving_loop(int n) {
    int count = 0;
    while (n > 0) {
        n = n / 2;
        count++;
    }
    return count;
}
""",
            "c": """\
int halving_loop(int n) {
    int count = 0;
    while (n > 0) {
        n = n / 2;
        count++;
    }
    return count;
}
""",
            "java": """\
class Solution {
    static int halvingLoop(int n) {
        int count = 0;
        while (n > 0) {
            n = n / 2;
            count++;
        }
        return count;
    }
}
""",
            "javascript": """\
function halvingLoop(n) {
    let count = 0;
    while (n > 0) {
        n = Math.floor(n / 2);
        count++;
    }
    return count;
}
""",
            "go": """\
package main

func halvingLoop(n int) int {
    count := 0
    for n > 0 {
        n = n / 2
        count++
    }
    return count
}
""",
        },
    ),
    SynthAlgorithm(
        name="merge_sort",
        time_class=TimeClass.O_N_LOG_N,
        space_class=SpaceClass.O_N,
        code_by_language={
            "python": """\
def merge(left, right):
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
""",
            "cpp": """\
#include <vector>
std::vector<int> merge(std::vector<int> left, std::vector<int> right) {
    std::vector<int> result;
    size_t i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) {
            result.push_back(left[i]);
            i++;
        } else {
            result.push_back(right[j]);
            j++;
        }
    }
    while (i < left.size()) {
        result.push_back(left[i]);
        i++;
    }
    while (j < right.size()) {
        result.push_back(right[j]);
        j++;
    }
    return result;
}

std::vector<int> merge_sort(std::vector<int> arr) {
    if (arr.size() <= 1) {
        return arr;
    }
    size_t mid = arr.size() / 2;
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());
    return merge(merge_sort(left), merge_sort(right));
}
""",
            "c": """\
#include <stdlib.h>
static int *merge(int left[], int left_n, int right[], int right_n, int *out_n) {
    int *result = malloc(sizeof(int) * (size_t)(left_n + right_n));
    int i = 0, j = 0, k = 0;
    while (i < left_n && j < right_n) {
        if (left[i] <= right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }
    while (i < left_n) {
        result[k++] = left[i++];
    }
    while (j < right_n) {
        result[k++] = right[j++];
    }
    *out_n = k;
    return result;
}

int *merge_sort(int arr[], int n, int *out_n) {
    if (n <= 1) {
        *out_n = n;
        int *copy = malloc(sizeof(int) * (size_t)n);
        for (int i = 0; i < n; i++) {
            copy[i] = arr[i];
        }
        return copy;
    }
    int mid = n / 2;
    int left_n, right_n;
    int *left = merge_sort(arr, mid, &left_n);
    int *right = merge_sort(arr + mid, n - mid, &right_n);
    int *result = merge(left, left_n, right, right_n, out_n);
    free(left);
    free(right);
    return result;
}
""",
            "java": """\
class Solution {
    static int[] merge(int[] left, int[] right) {
        int[] result = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result[k++] = left[i++];
            } else {
                result[k++] = right[j++];
            }
        }
        while (i < left.length) {
            result[k++] = left[i++];
        }
        while (j < right.length) {
            result[k++] = right[j++];
        }
        return result;
    }

    static int[] mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return arr;
        }
        int mid = arr.length / 2;
        int[] left = java.util.Arrays.copyOfRange(arr, 0, mid);
        int[] right = java.util.Arrays.copyOfRange(arr, mid, arr.length);
        return merge(mergeSort(left), mergeSort(right));
    }
}
""",
            "javascript": """\
function merge(left, right) {
    const result = [];
    let i = 0;
    let j = 0;
    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i]);
            i++;
        } else {
            result.push(right[j]);
            j++;
        }
    }
    while (i < left.length) {
        result.push(left[i]);
        i++;
    }
    while (j < right.length) {
        result.push(right[j]);
        j++;
    }
    return result;
}

function mergeSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    return merge(left, right);
}
""",
            "go": """\
package main

func merge(left []int, right []int) []int {
    result := make([]int, 0, len(left)+len(right))
    i, j := 0, 0
    for i < len(left) && j < len(right) {
        if left[i] <= right[j] {
            result = append(result, left[i])
            i++
        } else {
            result = append(result, right[j])
            j++
        }
    }
    result = append(result, left[i:]...)
    result = append(result, right[j:]...)
    return result
}

func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    left := mergeSort(arr[:mid])
    right := mergeSort(arr[mid:])
    return merge(left, right)
}
""",
        },
    ),
    SynthAlgorithm(
        name="memoized_fibonacci",
        time_class=TimeClass.O_N,
        space_class=SpaceClass.O_N,
        code_by_language={
            "python": """\
def fib_helper(n, memo):
    if n <= 1:
        return n
    if memo[n] != -1:
        return memo[n]
    memo[n] = fib_helper(n - 1, memo) + fib_helper(n - 2, memo)
    return memo[n]


def memoized_fibonacci(n):
    memo = [-1] * (n + 1)
    return fib_helper(n, memo)
""",
            "cpp": """\
#include <vector>
int fib_helper(int n, std::vector<int> &memo) {
    if (n <= 1) {
        return n;
    }
    if (memo[n] != -1) {
        return memo[n];
    }
    memo[n] = fib_helper(n - 1, memo) + fib_helper(n - 2, memo);
    return memo[n];
}

int memoized_fibonacci(int n) {
    std::vector<int> memo(n + 1, -1);
    return fib_helper(n, memo);
}
""",
            "c": """\
#include <stdlib.h>
static int fib_helper(int n, int memo[]) {
    if (n <= 1) {
        return n;
    }
    if (memo[n] != -1) {
        return memo[n];
    }
    memo[n] = fib_helper(n - 1, memo) + fib_helper(n - 2, memo);
    return memo[n];
}

int memoized_fibonacci(int n) {
    int *memo = malloc(sizeof(int) * (size_t)(n + 1));
    for (int i = 0; i <= n; i++) {
        memo[i] = -1;
    }
    int result = fib_helper(n, memo);
    free(memo);
    return result;
}
""",
            "java": """\
class Solution {
    static int fibHelper(int n, int[] memo) {
        if (n <= 1) {
            return n;
        }
        if (memo[n] != -1) {
            return memo[n];
        }
        memo[n] = fibHelper(n - 1, memo) + fibHelper(n - 2, memo);
        return memo[n];
    }

    static int memoizedFibonacci(int n) {
        int[] memo = new int[n + 1];
        java.util.Arrays.fill(memo, -1);
        return fibHelper(n, memo);
    }
}
""",
            "javascript": """\
function fibHelper(n, memo) {
    if (n <= 1) {
        return n;
    }
    if (memo[n] !== -1) {
        return memo[n];
    }
    memo[n] = fibHelper(n - 1, memo) + fibHelper(n - 2, memo);
    return memo[n];
}

function memoizedFibonacci(n) {
    const memo = new Array(n + 1).fill(-1);
    return fibHelper(n, memo);
}
""",
            "go": """\
package main

func fibHelper(n int, memo []int) int {
    if n <= 1 {
        return n
    }
    if memo[n] != -1 {
        return memo[n]
    }
    memo[n] = fibHelper(n-1, memo) + fibHelper(n-2, memo)
    return memo[n]
}

func memoizedFibonacci(n int) int {
    memo := make([]int, n+1)
    for i := range memo {
        memo[i] = -1
    }
    return fibHelper(n, memo)
}
""",
        },
    ),
)


def build_records() -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    for algo in ALGORITHMS:
        problem_id = f"synth_{algo.name}"
        for language, code in algo.code_by_language.items():
            records.append(
                CorpusRecord(
                    problem_id=problem_id,
                    solution_id=f"{problem_id}_{language}",
                    source=SOURCE,
                    language=language,
                    code=code,
                    time_class=algo.time_class.value,
                    space_class=algo.space_class.value if algo.space_class else None,
                )
            )
    return records


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    records = build_records()
    write_jsonl(records, args.out)

    languages = sorted({r.language for r in records})
    print(
        json.dumps(
            {
                "algorithms": len(ALGORITHMS),
                "languages": languages,
                "records": len(records),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
