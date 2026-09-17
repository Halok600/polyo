"""BIC curve-shape fitting tests (plan §15: "Fit tests -- fit.py recovers
known curves from synthetic timing series, including noisy ones.")
"""
from __future__ import annotations

import math
import random

import pytest

from oracle.fit import (
    Candidate,
    InsufficientSamplesError,
    classify_space,
    classify_time,
    fit_space_bytes,
    fit_space_depth,
    fit_time,
)
from oracle.runner import Sample

_NS = [64, 256, 1024, 4096, 16384, 65536]


def _make_samples(ns: list[int], shape, scale: float = 1000.0, noise: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    samples = []
    for n in ns:
        cost = max(int(scale * shape(n) * math.exp(rng.uniform(-noise, noise))), 1)
        samples.append(Sample(n=n, time_ns=cost, peak_bytes=cost, max_call_depth=cost))
    return samples


@pytest.mark.parametrize(
    "candidate,shape",
    [
        (Candidate.CONST, lambda n: 1.0),
        (Candidate.LOG, lambda n: math.log(n)),
        (Candidate.LINEAR, lambda n: n),
        (Candidate.LINEARITHMIC, lambda n: n * math.log(n)),
        (Candidate.QUADRATIC, lambda n: n**2),
        (Candidate.CUBIC, lambda n: n**3),
        (Candidate.EXPONENTIAL, lambda n: 2**n),
    ],
)
def test_fit_recovers_the_exact_generating_shape(candidate, shape):
    ns = _NS if candidate is not Candidate.EXPONENTIAL else [8, 10, 12, 14, 16, 18]
    samples = _make_samples(ns, shape, noise=0.0)
    result = fit_time(samples)
    assert result.best.candidate == candidate
    assert result.best.r_squared > 0.999


def test_fit_recovers_shape_under_moderate_noise():
    samples = _make_samples(_NS, lambda n: n**2, noise=0.05, seed=42)
    result = fit_time(samples)
    assert result.best.candidate == Candidate.QUADRATIC
    assert result.best.r_squared >= 0.8


def test_heavy_noise_can_push_r_squared_below_the_floor():
    samples = _make_samples(_NS, lambda n: n, noise=5.0, seed=7)
    result = fit_time(samples)
    # Not asserting a specific shape wins under this much noise -- only that
    # the floor does its job of discarding a bad fit rather than guessing.
    assert classify_time(result) is None or result.best.r_squared >= 0.8


def test_fit_requires_at_least_two_distinct_n():
    samples = [Sample(n=10, time_ns=100, peak_bytes=1, max_call_depth=1)]
    with pytest.raises(InsufficientSamplesError):
        fit_time(samples)


def test_confidence_is_high_when_one_shape_dominates():
    samples = _make_samples(_NS, lambda n: n**2, noise=0.0)
    result = fit_time(samples)
    assert result.confidence > 0.9


def test_classify_time_uses_r_squared_floor():
    clean = _make_samples(_NS, lambda n: n, noise=0.0)
    assert classify_time(fit_time(clean)) is not None

    noisy = _make_samples(_NS, lambda n: n, noise=8.0, seed=3)
    noisy_result = fit_time(noisy)
    if noisy_result.best.r_squared < 0.8:
        assert classify_time(noisy_result) is None


def test_classify_space_takes_the_larger_growing_signal():
    # Heap bytes look constant (e.g. an in-place function); call depth grows
    # linearly (e.g. plain recursion) -- true auxiliary space is the latter.
    byte_samples = _make_samples(_NS, lambda n: 1.0, noise=0.0)
    depth_samples = _make_samples(_NS, lambda n: n, noise=0.0)
    combined = [
        Sample(n=b.n, time_ns=1, peak_bytes=b.peak_bytes, max_call_depth=d.max_call_depth)
        for b, d in zip(byte_samples, depth_samples, strict=True)
    ]
    byte_result = fit_space_bytes(combined)
    depth_result = fit_space_depth(combined)
    classified = classify_space(byte_result, depth_result)
    assert classified is not None
    space_class, _confidence = classified
    assert space_class.value == "O(n)"
