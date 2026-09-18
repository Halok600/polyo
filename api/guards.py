"""Request guards (plan §10): 64 KB body cap, parse timeout, per-IP rate
limiting.

The timeout uses a thread pool, not `signal.alarm` (Unix-only, and this
image is Linux but the dev machine is Windows -- portability matters for
local `docker compose up` testing too) and not a subprocess (forbidden by
`tests/test_isolation.py`; a timeout on our own parsing code is not the
same thing as executing untrusted code, but a subprocess-based timeout
would still trip that test's import check).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TypeVar

MAX_CODE_BYTES = 64 * 1024
PARSE_TIMEOUT_SECONDS = 5.0

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="polyo-predict")


class ParseTimeoutError(TimeoutError):
    """Raised when parsing/feature extraction/prediction exceeds
    `PARSE_TIMEOUT_SECONDS`."""


def run_with_timeout(fn: Callable[[], T], timeout_s: float = PARSE_TIMEOUT_SECONDS) -> T:
    future = _executor.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as e:
        raise ParseTimeoutError(f"exceeded {timeout_s}s") from e


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """In-process token bucket per key, e.g. client IP (plan §10: "no
    Redis") -- correct for a single free-tier instance (Render's free web
    service runs exactly one), not for multiple instances behind a load
    balancer, which this project's deployment (plan §13) never has."""

    def __init__(self, capacity: float = 20.0, refill_per_second: float = 0.5):
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(tokens=self._capacity, last_refill=now)
                self._buckets[key] = bucket
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_per_second)
            bucket.last_refill = now
            if bucket.tokens < 1.0:
                return False
            bucket.tokens -= 1.0
            return True
