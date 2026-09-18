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

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TypeVar

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_CODE_BYTES = 64 * 1024
# Comfortably above MAX_CODE_BYTES even accounting for JSON string-escaping
# overhead (worst case, e.g. a code sample that's mostly control characters,
# can multiply encoded size several-fold) plus the request's other small
# fields -- this is a transport-level circuit breaker against a genuinely
# oversized POST, not the authoritative "reject too-long code" check (that's
# api/predict.py's MAX_CODE_BYTES comparison, which stays the one producing
# a clean 400 with a real error message).
MAX_REQUEST_BODY_BYTES = 512 * 1024
PARSE_TIMEOUT_SECONDS = 5.0

T = TypeVar("T")


class MaxBodySizeMiddleware:
    """Rejects an oversized request body via its Content-Length header,
    before FastAPI reads and JSON-decodes it into memory at all.

    api/predict.py's own MAX_CODE_BYTES check runs *after* FastAPI has
    already buffered and parsed the full request body -- a multi-hundred-MB
    POST is fully read into memory before that check ever runs. This is
    plain ASGI, not Starlette's BaseHTTPMiddleware (which itself buffers
    the whole body to hand to `call_next`), so it can reject before the
    body is touched at all.

    Deliberately Content-Length-only, not a byte-counting wrapper around
    `receive()`: every real client here (browser fetch, curl, requests)
    sets Content-Length for a JSON POST, and a client sophisticated enough
    to omit it and stream chunked instead is a materially different (and
    here, unlikely) threat model than "someone points a large POST at this
    endpoint," which is what this exists to stop.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            for name, value in scope.get("headers", []):
                if name == b"content-length" and int(value) > self.max_bytes:
                    response = JSONResponse({"detail": "request body too large"}, status_code=413)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)

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


def client_key(request: Request) -> str:
    """The per-IP rate-limit key (plan §10). The real client IP behind
    Render's edge proxy when TRUST_PROXY_HEADERS=1 (see render.yaml), the
    direct TCP peer otherwise.

    **Replaces a real, shipped bug, not a hypothetical one**: an earlier
    version of this trust decision lived in uvicorn's own CLI, via
    `--proxy-headers --forwarded-allow-ips="*"`. uvicorn's
    `_TrustedHosts.get_trusted_client_host` has an `always_trust` branch for
    `"*"` that returns `x_forwarded_for_hosts[0]` -- the *first* entry,
    i.e. whatever the client itself put in the header. Proxies
    conventionally *append* to X-Forwarded-For, never replace it (RFC 7239's
    predecessor convention; every hop from client to origin adds one entry),
    so the first entry is always attacker-controlled and the *last* entry is
    the one Render's own edge actually appended -- uvicorn's "*" mode reads
    exactly the wrong end of the list. Rotating the header per request fully
    defeated the rate limiter under that version. Done here instead, in
    application code, specifically to control which end of the list wins
    directly, rather than depend on a middleware whose "*" branch doesn't
    match this deployment's actual proxy topology.
    """
    if os.environ.get("TRUST_PROXY_HEADERS") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


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
