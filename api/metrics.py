"""Minimal Prometheus text-format metrics (plan §10's `GET /metrics`) --
hand-rolled counters/histogram, not the `prometheus_client` package, to
keep the served image's dependency list exactly what plan §13 budgets for
(no extra package for a handful of counters).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

_LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf"))


@dataclass
class _Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    requests_total: dict[str, int] = field(default_factory=dict)  # status -> count
    # Already cumulative per Prometheus histogram semantics: bucket i holds
    # the count of observations <= its bound, so `record` increments every
    # bucket whose bound is >= the observed latency, not just the smallest
    # one that fits.
    latency_bucket_counts: list[int] = field(
        default_factory=lambda: [0] * len(_LATENCY_BUCKETS_SECONDS)
    )
    latency_sum: float = 0.0
    latency_count: int = 0

    def record(self, status: str, latency_s: float) -> None:
        with self._lock:
            self.requests_total[status] = self.requests_total.get(status, 0) + 1
            for i, bound in enumerate(_LATENCY_BUCKETS_SECONDS):
                if latency_s <= bound:
                    self.latency_bucket_counts[i] += 1
            self.latency_sum += latency_s
            self.latency_count += 1

    def render_prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP polyo_requests_total Total predict requests by status.",
                "# TYPE polyo_requests_total counter",
            ]
            for status, count in sorted(self.requests_total.items()):
                lines.append(f'polyo_requests_total{{status="{status}"}} {count}')
            lines.append("# HELP polyo_predict_latency_seconds Predict request latency.")
            lines.append("# TYPE polyo_predict_latency_seconds histogram")
            bucket_pairs = zip(_LATENCY_BUCKETS_SECONDS, self.latency_bucket_counts, strict=True)
            for bound, count in bucket_pairs:
                bound_label = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'polyo_predict_latency_seconds_bucket{{le="{bound_label}"}} {count}')
            lines.append(f"polyo_predict_latency_seconds_sum {self.latency_sum}")
            lines.append(f"polyo_predict_latency_seconds_count {self.latency_count}")
            return "\n".join(lines) + "\n"


metrics = _Metrics()
