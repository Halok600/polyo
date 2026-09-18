"""PolyO API -- parses code and predicts complexity. Executes nothing.

Hard invariant, enforced by tests/test_isolation.py: no module reachable
from this package may import anything under `oracle/` (which executes
untrusted code offline, during dataset labelling), and no file in this
package may contain exec/eval/compile/subprocess. Predicting from source
must never require running it.
"""
from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.guards import (
    MAX_REQUEST_BODY_BYTES,
    MaxBodySizeMiddleware,
    ParseTimeoutError,
    RateLimiter,
    run_with_timeout,
)
from api.logging_utils import configure_logging, log_event
from api.metrics import metrics
from api.models_registry import ModelRegistry, ModelsNotTrainedError, load_registry
from api.predict import SERVED_LANGUAGES, PredictionError, predict

_logger = configure_logging()
_rate_limiter = RateLimiter()
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Loaded once, at process start (see `lifespan` below) -- never
    lazily from a request handler, and never from `/health`, which must
    stay instant and untouched by model state (plan §10)."""
    if _registry is None:
        raise ModelsNotTrainedError("model registry was not loaded at startup")
    return _registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _registry
    try:
        _registry = load_registry()
    except ModelsNotTrainedError as e:
        # Logged, not raised: /health and /v1/languages must still work
        # (e.g. so CI's docker-build smoke check and Render's keep-alive
        # succeed) even before `python -m models.train_production` has
        # ever been run. /v1/predict fails loudly (503) instead.
        log_event(_logger, "startup_models_not_trained", error=str(e))
    yield


app = FastAPI(
    title="PolyO",
    description="Static, multi-language time & space complexity prediction.",
    version="0.0.0",
    lifespan=lifespan,
)

# The frontend (Vercel) and API (Render) are different origins by design
# (plan §13) -- ALLOWED_ORIGINS is set at deploy time to the real Vercel
# URL; the local Next.js dev server's origin is always allowed so
# `npm run dev` + `uvicorn api.main:app` works out of the box.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_configured_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*_configured_origins, "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# Added after CORSMiddleware on purpose: Starlette builds its middleware
# stack outer-to-inner in *reverse* add_middleware call order (each call
# prepends), so this becomes the outer layer -- it sees and can reject an
# oversized request before CORS, JSON parsing, or the route handler ever
# touch the body.
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)


class Health(BaseModel):
    status: str


@app.get("/health", response_model=Health)
def health() -> Health:
    # Deliberately touches no model, dataset, or DB. This is the Render
    # free-tier keep-alive target (see plan §0/§13) and must stay instant.
    return Health(status="ok")


class Language(BaseModel):
    id: str
    tier: int


SUPPORTED_LANGUAGES_RESPONSE: list[Language] = [
    Language(id="python", tier=1),
    Language(id="cpp", tier=1),
    Language(id="java", tier=1),
    Language(id="javascript", tier=1),
    Language(id="c", tier=2),
    Language(id="go", tier=2),
]
# Single source of truth for "served live" is `api.predict.SERVED_LANGUAGES`
# -- deliberately NOT `parsing.parse.SUPPORTED_LANGUAGES`, which also
# covers Tier 3 (plan §3: TypeScript, Rust, C#, Kotlin -- IR mapping file
# only, no oracle, added Phase 7). Whether to advertise and serve Tier 3
# live is its own product decision, deferred, not an accident of which
# constant this validation happens to import. Checked, not just asserted
# in a comment: the response list above and `SERVED_LANGUAGES` must name
# the same languages, or `/v1/languages` and `/v1/predict` would silently
# disagree about what's supported.
assert {lang.id for lang in SUPPORTED_LANGUAGES_RESPONSE} == SERVED_LANGUAGES


@app.get("/v1/languages", response_model=list[Language])
def languages() -> list[Language]:
    return SUPPORTED_LANGUAGES_RESPONSE


class PredictRequest(BaseModel):
    language: str = Field(default="auto")
    code: str = Field(min_length=1)
    entrypoint: str | None = None


class ClassPrediction(BaseModel):
    class_: str = Field(alias="class")
    rank: int
    confidence: float
    distribution: dict[str, float]

    model_config = {"populate_by_name": True}


class AttributionItem(BaseModel):
    feature: str
    contribution: float
    spans: list[list[int]]


class CurveDimension(BaseModel):
    predicted_class: str
    series: dict[str, list[float]]


class Curve(BaseModel):
    n: list[int]
    time: CurveDimension
    space: CurveDimension


class IrSummary(BaseModel):
    nodes: int
    edges: int
    histogram: dict[str, int]


class PredictResponse(BaseModel):
    language_detected: str
    time: ClassPrediction
    space: ClassPrediction
    attribution: list[AttributionItem]
    curve: Curve
    ir: IrSummary
    warnings: list[str]


@app.post("/v1/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest, http_request: Request) -> dict[str, object]:
    start = time.monotonic()
    client_ip = http_request.client.host if http_request.client else "unknown"

    if request.language != "auto" and request.language not in SERVED_LANGUAGES:
        metrics.record("bad_request", time.monotonic() - start)
        raise HTTPException(status_code=400, detail=f"unsupported language: {request.language!r}")

    if not _rate_limiter.allow(client_ip):
        metrics.record("rate_limited", time.monotonic() - start)
        raise HTTPException(status_code=429, detail="rate limit exceeded, try again shortly")

    try:
        registry = get_registry()
    except ModelsNotTrainedError as e:
        metrics.record("service_unavailable", time.monotonic() - start)
        raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        result = run_with_timeout(lambda: predict(registry, request.code, request.language))
    except PredictionError as e:
        metrics.record("bad_request", time.monotonic() - start)
        log_event(_logger, "predict_bad_request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ParseTimeoutError as e:
        metrics.record("timeout", time.monotonic() - start)
        log_event(_logger, "predict_timeout", error=str(e))
        raise HTTPException(status_code=504, detail="prediction timed out") from e

    latency_s = time.monotonic() - start
    metrics.record("ok", latency_s)
    log_event(
        _logger,
        "predict_ok",
        language_detected=result["language_detected"],
        latency_s=round(latency_s, 4),
    )
    return result


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(content=metrics.render_prometheus_text(), media_type="text/plain")
