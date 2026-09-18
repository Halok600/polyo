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
    client_key,
    run_with_timeout,
)


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    """Minimal stand-in for starlette.requests.Request -- client_key only
    touches .headers.get(...) and .client.host, so a full ASGI scope would
    be pure ceremony here."""

    def __init__(self, client_host: str | None, headers: dict[str, str] | None = None) -> None:
        self.client = _FakeClient(client_host) if client_host is not None else None
        self.headers = headers or {}


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


def test_client_key_ignores_x_forwarded_for_when_not_trusted(monkeypatch):
    # The untrusted/local/docker-compose default: a spoofed header must have
    # zero effect, full stop, regardless of what it claims.
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    request = _FakeRequest("203.0.113.77", {"x-forwarded-for": "9.9.9.9"})
    assert client_key(request) == "203.0.113.77"


def test_client_key_takes_the_rightmost_forwarded_for_entry_when_trusted(monkeypatch):
    # The exact bug this replaced: an earlier version (uvicorn's own
    # --forwarded-allow-ips="*") took the *leftmost* entry, which is
    # whatever the client itself claims -- trivially spoofable, and fully
    # defeats the rate limiter by rotating the header per request. Render's
    # edge appends the real client IP as the *last* hop; every entry before
    # it (including this attacker-set "1.2.3.4") must be ignored.
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    request = _FakeRequest(
        "10.0.0.5",  # Render's edge itself, as seen by the direct TCP peer
        {"x-forwarded-for": "1.2.3.4, 203.0.113.77"},
    )
    assert client_key(request) == "203.0.113.77"


def test_client_key_trims_whitespace_around_the_rightmost_entry(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    request = _FakeRequest("10.0.0.5", {"x-forwarded-for": "1.2.3.4,  203.0.113.77 "})
    assert client_key(request) == "203.0.113.77"


def test_client_key_falls_back_to_the_tcp_peer_when_trusted_but_header_absent(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    request = _FakeRequest("203.0.113.77")
    assert client_key(request) == "203.0.113.77"


def test_client_key_returns_unknown_when_there_is_no_client_at_all(monkeypatch):
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    request = _FakeRequest(None)
    assert client_key(request) == "unknown"
