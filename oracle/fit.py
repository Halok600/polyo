"""BIC-based curve-shape classification for oracle measurements (plan §6).

Each candidate is a single-parameter model cost ~= a * basis(n) (a fixed
functional shape, only the scale `a` is free), fit in log space so models of
wildly different growth rates land on a comparable residual scale: taking
logs turns "cost = a * basis(n)" into "ln(cost) = ln(a) + ln(basis(n))",
whose least-squares solution is the closed-form mean residual, no numerical
solver needed. Confidence comes from Schwarz/BIC weights against the
runner-up; fits below an R^2 floor are discarded rather than guessed (plan
§6, §16) -- a smaller clean dataset beats a larger noisy one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.taxonomy import SpaceClass, TimeClass, space_rank
from oracle.runner import Sample

R2_FLOOR = 0.8


class Candidate(str, Enum):
    CONST = "const"
    LOG = "log"
    LINEAR = "linear"
    LINEARITHMIC = "linearithmic"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    EXPONENTIAL = "exponential"


_TIME_CANDIDATES: tuple[Candidate, ...] = tuple(Candidate)
# SpaceClass has no cubic/exponential rung (plan §3) -- don't offer shapes
# the taxonomy can't express.
_SPACE_CANDIDATES: tuple[Candidate, ...] = (
    Candidate.CONST,
    Candidate.LOG,
    Candidate.LINEAR,
    Candidate.LINEARITHMIC,
    Candidate.QUADRATIC,
)

_TIME_CLASS_BY_CANDIDATE: dict[Candidate, TimeClass] = {
    Candidate.CONST: TimeClass.O_1,
    Candidate.LOG: TimeClass.O_LOG_N,
    Candidate.LINEAR: TimeClass.O_N,
    Candidate.LINEARITHMIC: TimeClass.O_N_LOG_N,
    Candidate.QUADRATIC: TimeClass.O_N2,
    Candidate.CUBIC: TimeClass.O_N3,
    Candidate.EXPONENTIAL: TimeClass.O_2N,
}
_SPACE_CLASS_BY_CANDIDATE: dict[Candidate, SpaceClass] = {
    Candidate.CONST: SpaceClass.O_1,
    Candidate.LOG: SpaceClass.O_LOG_N,
    Candidate.LINEAR: SpaceClass.O_N,
    Candidate.LINEARITHMIC: SpaceClass.O_N_LOG_N,
    Candidate.QUADRATIC: SpaceClass.O_N2,
}


def _log_basis(candidate: Candidate, n: int) -> float:
    ln_n = math.log(n)
    if candidate is Candidate.CONST:
        return 0.0
    if candidate is Candidate.LOG:
        return math.log(ln_n)
    if candidate is Candidate.LINEAR:
        return ln_n
    if candidate is Candidate.LINEARITHMIC:
        return ln_n + math.log(ln_n)
    if candidate is Candidate.QUADRATIC:
        return 2 * ln_n
    if candidate is Candidate.CUBIC:
        return 3 * ln_n
    if candidate is Candidate.EXPONENTIAL:
        return n * math.log(2)
    raise AssertionError(candidate)  # pragma: no cover -- exhaustive above


class InsufficientSamplesError(ValueError):
    """Fewer than two distinct-n samples were given -- no shape can be fit."""


@dataclass(frozen=True, slots=True)
class CandidateFit:
    candidate: Candidate
    log_scale: float  # ln(a)
    r_squared: float
    bic: float


@dataclass(frozen=True, slots=True)
class FitResult:
    ranked: tuple[CandidateFit, ...]  # best (lowest BIC) first

    @property
    def best(self) -> CandidateFit:
        return self.ranked[0]

    @property
    def runner_up(self) -> CandidateFit | None:
        return self.ranked[1] if len(self.ranked) > 1 else None

    @property
    def confidence(self) -> float:
        """Schwarz/BIC weight of the best fit against all candidates."""
        weights = [math.exp(-0.5 * (f.bic - self.best.bic)) for f in self.ranked]
        return weights[0] / sum(weights)


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return float(ordered[mid])


def _fit_one(candidate: Candidate, ns: list[int], log_costs: list[float]) -> CandidateFit:
    residuals = [lc - _log_basis(candidate, n) for n, lc in zip(ns, log_costs, strict=True)]
    log_scale = sum(residuals) / len(residuals)
    rss = sum((r - log_scale) ** 2 for r in residuals)
    mean_log_cost = sum(log_costs) / len(log_costs)
    tss = sum((lc - mean_log_cost) ** 2 for lc in log_costs)
    r_squared = 1.0 if rss == 0.0 else (1.0 - rss / tss if tss > 0 else 0.0)
    n_samples = len(ns)
    bic = n_samples * math.log(max(rss, 1e-12) / n_samples) + math.log(n_samples)
    return CandidateFit(candidate=candidate, log_scale=log_scale, r_squared=r_squared, bic=bic)


def fit(
    samples: list[Sample],
    cost: str,
    candidates: tuple[Candidate, ...] = _TIME_CANDIDATES,
) -> FitResult:
    by_n: dict[int, list[int]] = {}
    for s in samples:
        by_n.setdefault(s.n, []).append(getattr(s, cost))
    ns = sorted(by_n)
    if len(ns) < 2:
        raise InsufficientSamplesError(
            f"need samples at >=2 distinct n to fit a shape, got {len(ns)}"
        )
    # Median per n -- robust to one-off scheduler jitter without discarding
    # a whole n's worth of measurement.
    log_costs = [math.log(max(_median(by_n[n]), 1.0)) for n in ns]

    fits = [_fit_one(c, ns, log_costs) for c in candidates]
    ranked = tuple(sorted(fits, key=lambda f: f.bic))
    return FitResult(ranked=ranked)


def fit_time(samples: list[Sample]) -> FitResult:
    return fit(samples, cost="time_ns", candidates=_TIME_CANDIDATES)


def fit_space_bytes(samples: list[Sample]) -> FitResult:
    return fit(samples, cost="peak_bytes", candidates=_SPACE_CANDIDATES)


def fit_space_depth(samples: list[Sample]) -> FitResult | None:
    """None when this language's driver doesn't measure call-stack depth
    (plan §14 Phase 4: Python only, see `Sample.max_call_depth`'s docstring)
    -- every sample from one run shares a language, so checking the first is
    enough to know whether the whole batch has the signal at all."""
    if samples[0].max_call_depth is None:
        return None
    return fit(samples, cost="max_call_depth", candidates=_SPACE_CANDIDATES)


def classify_time(result: FitResult) -> tuple[TimeClass, float] | None:
    if result.best.r_squared < R2_FLOOR:
        return None
    return _TIME_CLASS_BY_CANDIDATE[result.best.candidate], result.confidence


def classify_space(
    byte_result: FitResult, depth_result: FitResult | None
) -> tuple[SpaceClass, float] | None:
    """True auxiliary space is at least the larger-growing of two
    independent signals: heap allocation and call-stack depth -- heap
    allocation alone cannot see a solution that recurses without allocating,
    and plan §3 explicitly requires the recursion stack to count.
    `depth_result` is None wherever that second signal isn't measured (plan
    §14 Phase 4: every language but Python currently) -- classification then
    honestly rests on heap allocation alone rather than a faked signal, so a
    recursive, allocation-light solution in those languages can undercount
    its true space class. See `Sample.max_call_depth`'s docstring."""
    candidates: list[tuple[SpaceClass, float]] = []
    if byte_result.best.r_squared >= R2_FLOOR:
        candidates.append(
            (_SPACE_CLASS_BY_CANDIDATE[byte_result.best.candidate], byte_result.confidence)
        )
    if depth_result is not None and depth_result.best.r_squared >= R2_FLOOR:
        candidates.append(
            (_SPACE_CLASS_BY_CANDIDATE[depth_result.best.candidate], depth_result.confidence)
        )
    if not candidates:
        return None
    return max(candidates, key=lambda c: space_rank(c[0]))
