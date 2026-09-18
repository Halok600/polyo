"""Tests for the core prediction pipeline (plan §10's POST /v1/predict
logic, minus the HTTP layer -- see tests/test_api_endpoints.py for that).
Uses the `tiny_registry` fixture (tests/conftest.py): a real, tiny,
fast-to-train registry, not the production artifacts.
"""
from __future__ import annotations

import pytest

from api.guards import MAX_CODE_BYTES
from api.predict import PredictionError, predict

_LINEAR_PY = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"


def test_predict_returns_every_documented_top_level_field(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    assert set(result.keys()) == {
        "language_detected",
        "time",
        "space",
        "attribution",
        "curve",
        "ir",
        "warnings",
    }


def test_predict_detects_the_requested_language_when_given_explicitly(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    assert result["language_detected"] == "python"


def test_predict_auto_detects_language_from_code_content(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "auto")
    assert result["language_detected"] == "python"


def test_predict_time_and_space_fields_have_the_documented_shape(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    for dimension in ("time", "space"):
        entry = result[dimension]
        assert set(entry.keys()) == {"class", "rank", "confidence", "distribution"}
        assert isinstance(entry["rank"], int)
        assert 0.0 <= entry["confidence"] <= 1.0
        # Each class probability is independently rounded to 4 decimals
        # for the response (api/predict.py) -- across up to 7 classes
        # (time), the worst-case cumulative rounding error is
        # 7 * 0.5e-4 = 3.5e-4, so the tolerance here must clear that, not
        # the tighter bound a truly unrounded softmax sum would allow.
        assert abs(sum(entry["distribution"].values()) - 1.0) < 1e-3


def test_predict_curve_shares_one_n_grid_across_both_dimensions(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    assert result["curve"]["n"]
    assert result["curve"]["time"]["predicted_class"] == result["time"]["class"]
    assert result["curve"]["space"]["predicted_class"] == result["space"]["class"]


def test_predict_ir_summary_matches_a_real_parse(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    assert result["ir"]["nodes"] > 0
    assert result["ir"]["histogram"].get("LOOP_FOR") == 1


def test_predict_attribution_entries_have_the_documented_shape(tiny_registry):
    result = predict(tiny_registry, _LINEAR_PY, "python")
    for item in result["attribution"]:
        assert set(item.keys()) == {"feature", "contribution", "spans"}


def test_predict_rejects_code_over_the_size_cap(tiny_registry):
    huge_code = "x = 1\n" * (MAX_CODE_BYTES // 6 + 1000)
    with pytest.raises(PredictionError):
        predict(tiny_registry, huge_code, "python")


def test_predict_rejects_an_unsupported_language(tiny_registry):
    with pytest.raises(PredictionError):
        predict(tiny_registry, _LINEAR_PY, "cobol")


def test_predict_rejects_an_explicit_tier_3_language(tiny_registry):
    # Tier 3 (plan §3: TypeScript, Rust, C#, Kotlin) parses fine -- it's
    # not "unsupported" the way "cobol" is -- but serving it live is a
    # deliberately deferred product decision (see api/predict.py's
    # SERVED_LANGUAGES), not something that should work by accident.
    with pytest.raises(PredictionError):
        predict(tiny_registry, "fn f() {}", "rust")


def test_predict_rejects_tier_3_even_via_auto_detect(tiny_registry):
    # The gate must hold for the *auto-detected* result too, not just an
    # explicit `language="rust"` request -- auto-detect internally scans
    # every parseable language, Tier 3 included.
    rust_code = "fn linear_search(arr: &[i32], target: i32) -> i32 {\n    return -1;\n}\n"
    with pytest.raises(PredictionError):
        predict(tiny_registry, rust_code, "auto")


def test_predict_never_executes_the_submitted_code(tiny_registry, tmp_path):
    # A concrete, checkable side effect that would prove execution happened,
    # if it ever did -- plan §10's hard invariant, exercised end-to-end here
    # rather than only checked statically by tests/test_isolation.py.
    marker = tmp_path / "executed.marker"
    malicious = (
        f"import os\nos.system('echo executed > {marker.as_posix()}')\n"
        "def f():\n    return 1\n"
    )
    result = predict(tiny_registry, malicious, "python")
    assert result["language_detected"] == "python"
    assert not marker.exists()
