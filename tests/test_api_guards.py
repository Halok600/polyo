"""Tests for request guards: size cap constant, parse timeout, rate
limiter (plan §10)."""
from __future__ import annotations

import time

import pytest

from api.guards import (
    MAX_CODE_BYTES,
    MAX_REQUEST_BODY_BYTES,
    ParseTimeoutError,
    RateLimiter,
    run_with_timeout,
)


def test_max_code_bytes_is_64kb():
    assert MAX_CODE_BYTES == 64 * 1024


def test_max_request_body_bytes_leaves_headroom_over_max_code_bytes():
    # Must comfortably exceed MAX_CODE_BYTES -- JSON string-escaping can
    # multiply a code sample's encoded size, and the request has other
    # fields besides `code`. See tests/test_api_endpoints.py for the
    # HTTP-layer 413 behavior this constant actually gates.
    assert MAX_REQUEST_BODY_BYTES > MAX_CODE_BYTES


def test_run_with_timeout_returns_the_function_result_when_fast_enough():
    assert run_with_timeout(lambda: 1 + 1, timeout_s=1.0) == 2


def test_run_with_timeout_raises_when_the_function_is_too_slow():
    with pytest.raises(ParseTimeoutError):
        run_with_timeout(lambda: time.sleep(0.5), timeout_s=0.05)


def test_run_with_timeout_propagates_the_functions_own_exception():
    def raises():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_with_timeout(raises, timeout_s=1.0)


def test_rate_limiter_allows_requests_up_to_capacity():
    limiter = RateLimiter(capacity=3.0, refill_per_second=0.0)
    assert limiter.allow("ip-a")
    assert limiter.allow("ip-a")
    assert limiter.allow("ip-a")
    assert not limiter.allow("ip-a")


def test_rate_limiter_tracks_separate_keys_independently():
    limiter = RateLimiter(capacity=1.0, refill_per_second=0.0)
    assert limiter.allow("ip-a")
    assert not limiter.allow("ip-a")
    assert limiter.allow("ip-b")  # a different key has its own bucket


def test_rate_limiter_refills_over_time():
    limiter = RateLimiter(capacity=1.0, refill_per_second=10.0)
    assert limiter.allow("ip-a", now=0.0)
    assert not limiter.allow("ip-a", now=0.01)
    assert limiter.allow("ip-a", now=1.0)  # 10 tokens/s * 1s >> 1 token needed
