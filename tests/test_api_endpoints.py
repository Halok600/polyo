"""HTTP-layer tests for `api/main.py` (plan §10): status codes, guards, and
the isolation invariant, exercised through a real FastAPI `TestClient` --
not the module-level logic in `api/predict.py`, which
tests/test_api_predict.py already covers directly.

The module-level model registry is monkeypatched to the `tiny_registry`
fixture rather than depending on `models/artifacts/` (CI never trains
those).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as main_module

_LINEAR_PY = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"


def _client_with_registry(monkeypatch, registry):
    monkeypatch.setattr(main_module, "_registry", registry)
    monkeypatch.setattr(main_module, "_rate_limiter", main_module.RateLimiter())
    return TestClient(main_module.app)


def test_health_never_touches_the_model(monkeypatch):
    # No registry set at all -- /health must still succeed (plan §10).
    monkeypatch.setattr(main_module, "_registry", None)
    client = TestClient(main_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_languages_matches_the_plan_tiering(monkeypatch):
    monkeypatch.setattr(main_module, "_registry", None)
    client = TestClient(main_module.app)
    response = client.get("/v1/languages")
    assert response.status_code == 200
    by_id = {lang["id"]: lang["tier"] for lang in response.json()}
    assert by_id == {"python": 1, "cpp": 1, "java": 1, "javascript": 1, "c": 2, "go": 2}


def test_predict_returns_503_when_models_are_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "_registry", None)
    client = TestClient(main_module.app)
    response = client.post("/v1/predict", json={"language": "python", "code": _LINEAR_PY})
    assert response.status_code == 503


def test_predict_returns_200_with_the_documented_shape(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    response = client.post("/v1/predict", json={"language": "python", "code": _LINEAR_PY})
    assert response.status_code == 200
    body = response.json()
    assert body["language_detected"] == "python"
    assert "class" in body["time"]
    assert "class" in body["space"]


def test_predict_returns_400_for_an_unsupported_language(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    response = client.post("/v1/predict", json={"language": "cobol", "code": _LINEAR_PY})
    assert response.status_code == 400


def test_predict_returns_400_for_empty_code(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    response = client.post("/v1/predict", json={"language": "python", "code": ""})
    assert response.status_code == 422  # pydantic min_length=1 rejects it before the handler runs


def test_predict_returns_429_once_the_rate_limit_is_exhausted(monkeypatch, tiny_registry):
    monkeypatch.setattr(main_module, "_registry", tiny_registry)
    monkeypatch.setattr(main_module, "_rate_limiter", main_module.RateLimiter(capacity=1.0))
    client = TestClient(main_module.app)
    payload = {"language": "python", "code": _LINEAR_PY}
    first = client.post("/v1/predict", json=payload)
    second = client.post("/v1/predict", json=payload)
    assert first.status_code == 200
    assert second.status_code == 429


def test_predict_returns_413_for_an_oversized_body(monkeypatch, tiny_registry):
    # A genuinely large body (well over api.guards.MAX_REQUEST_BODY_BYTES),
    # not JSON -- MaxBodySizeMiddleware rejects on Content-Length before
    # FastAPI ever tries to parse it, so the content doesn't matter.
    client = _client_with_registry(monkeypatch, tiny_registry)
    response = client.post("/v1/predict", content=b"x" * (600 * 1024))
    assert response.status_code == 413


def test_metrics_endpoint_reflects_recorded_requests(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    client.post("/v1/predict", json={"language": "python", "code": _LINEAR_PY})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "polyo_requests_total" in response.text


_QUADRATIC_PY = "def f(a):\n    for x in a:\n        for y in a:\n            pass\n"


def test_predict_serves_the_v2_fields_over_http(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    response = client.post("/v1/predict", json={"language": "python", "code": _QUADRATIC_PY})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "symbolic"
    assert body["time"]["class"] == "O(n^2)"
    assert body["time"]["expression"] == "O(n^2)"
    assert body["time"]["certainty"] == "certain"
    assert body["time"]["extended_class"] == "O(n^2)"
    assert body["time"]["projection_lossy"] is False
    assert [step["kind"] for step in body["derivation"]][:2] == ["loop", "loop"]
    assert body["assumptions"] == []
    assert body["entry"] == "f"


def test_a_client_that_only_knows_the_legacy_fields_still_gets_them_all(monkeypatch, tiny_registry):
    client = _client_with_registry(monkeypatch, tiny_registry)
    body = client.post("/v1/predict", json={"language": "python", "code": _QUADRATIC_PY}).json()
    assert {"language_detected", "time", "space", "attribution", "curve", "ir", "warnings"} <= set(
        body
    )
    for dimension in ("time", "space"):
        assert {
            "class",
            "rank",
            "confidence",
            "distribution",
            "conformal_set",
            "conformal_coverage",
            "abstain",
        } <= set(body[dimension])
    assert {"n", "time", "space"} == set(body["curve"])
    assert all(
        {"feature", "contribution", "spans"} == set(item) for item in body["attribution"]
    )
