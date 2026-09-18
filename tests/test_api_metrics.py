"""Tests for the hand-rolled Prometheus text-format metrics (plan §10)."""
from __future__ import annotations

from api.metrics import _Metrics


def test_records_request_counts_by_status():
    m = _Metrics()
    m.record("ok", 0.01)
    m.record("ok", 0.02)
    m.record("bad_request", 0.01)
    assert m.requests_total == {"ok": 2, "bad_request": 1}


def test_latency_buckets_are_cumulative():
    m = _Metrics()
    m.record("ok", 0.03)  # falls into every bucket >= 0.05
    text = m.render_prometheus_text()
    assert 'polyo_predict_latency_seconds_bucket{le="0.05"} 1' in text
    assert 'polyo_predict_latency_seconds_bucket{le="+Inf"} 1' in text


def test_render_prometheus_text_includes_sum_and_count():
    m = _Metrics()
    m.record("ok", 0.1)
    m.record("ok", 0.3)
    text = m.render_prometheus_text()
    assert "polyo_predict_latency_seconds_count 2" in text
    assert "polyo_predict_latency_seconds_sum 0.4" in text


def test_render_prometheus_text_is_valid_looking_exposition_format():
    m = _Metrics()
    m.record("ok", 0.01)
    text = m.render_prometheus_text()
    assert text.startswith("# HELP")
    assert "# TYPE polyo_requests_total counter" in text
    assert "# TYPE polyo_predict_latency_seconds histogram" in text
