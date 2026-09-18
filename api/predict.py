"""Core prediction pipeline (plan §10's `POST /v1/predict`): parse, extract
features, run the served GNN (numpy, no torch) for the class prediction,
calibrate, attribute via a precomputed feature-importance weighting
(`api/attribution.py`), generate an illustrative growth curve, and
summarise the IR. Executes nothing (plan §10's hard invariant, enforced by
`tests/test_isolation.py`) -- everything here is static analysis over the
parsed IR.
"""
from __future__ import annotations

from api.attribution import top_contributions
from api.curve import growth_curve
from api.guards import MAX_CODE_BYTES
from api.models_registry import ModelRegistry
from core.taxonomy import SpaceClass, TimeClass, space_rank, time_rank
from features.tabular import extract_features
from models.calibrate import Calibrator
from models.gbdt import FEATURE_NAMES
from models.graph_batch import to_example_graph
from parsing.normalize import normalize_source
from parsing.parse import UnsupportedLanguageError, detect_language_from_source

_UNKNOWN_SHARE_WARNING_THRESHOLD = 0.2
# Which dimension's prediction the GBDT explains (plan §10's response has
# one flat `attribution` list, not one per dimension) -- time is the
# project's headline metric (plan §1's title order, and every other
# rung/report leads with it too).
_ATTRIBUTION_DIMENSION = "time"

# Tier 1/2 only (plan §3) -- NOT `parsing.parse.SUPPORTED_LANGUAGES`, which
# also covers Tier 3 (TypeScript, Rust, C#, Kotlin: IR mapping file only,
# no oracle, added Phase 7). The GNN is language-agnostic by construction
# (it reads the IR, never source text) and would in principle score Tier
# 3 code too, but whether to actually serve that live is a deliberate,
# deferred product decision -- enforced here so it can't happen by
# accident through auto-detect (which internally scans every parseable
# language, Tier 3 included) even though an explicit `language: "rust"`
# request is already rejected one level up, in `api/main.py`.
SERVED_LANGUAGES: frozenset[str] = frozenset(
    {"python", "cpp", "java", "javascript", "c", "go"}
)


class PredictionError(ValueError):
    """A request-level problem (bad language, code too large, unparseable)
    -- distinct from `ModelsNotTrainedError`, which is a deployment
    problem."""


def _rank_enum(dimension: str) -> type[TimeClass] | type[SpaceClass]:
    return TimeClass if dimension == "time" else SpaceClass


def _rank_fn(dimension: str):  # noqa: ANN201 -- returns one of two functions with different arg types
    return time_rank if dimension == "time" else space_rank


def _predict_dimension(
    registry: ModelRegistry, dimension: str, symbol_ids, edges_by_kind
) -> tuple[dict[str, object], dict[str, float]]:
    raw_scores = registry.gnn.forward(symbol_ids, edges_by_kind)[dimension]
    calibration = registry.calibration[dimension]
    calibrator = Calibrator(temperature=calibration.temperature, classes=calibration.classes)
    proba = calibrator.calibrate(raw_scores.reshape(1, -1))[0]
    pred_idx = int(proba.argmax())
    predicted_class = calibration.classes[pred_idx]

    class_proba_pairs = zip(calibration.classes, proba, strict=True)
    distribution = {cls: round(float(p), 4) for cls, p in class_proba_pairs}
    response = {
        "class": predicted_class,
        "rank": _rank_fn(dimension)(_rank_enum(dimension)(predicted_class)),
        "confidence": round(float(proba[pred_idx]), 4),
        "distribution": distribution,
    }
    return response, distribution


def _attribution_for(registry: ModelRegistry, features, ir):
    importance_weights = registry.feature_importance.get(_ATTRIBUTION_DIMENSION, {})
    feature_values = {name: features.as_dict()[name] for name in FEATURE_NAMES}
    return top_contributions(feature_values, importance_weights, registry.feature_scales, ir)


def predict(registry: ModelRegistry, code: str, language: str) -> dict[str, object]:
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise PredictionError(f"code exceeds the {MAX_CODE_BYTES}-byte cap")

    if language != "auto" and language not in SERVED_LANGUAGES:
        raise PredictionError(f"unsupported language: {language!r}")

    detected = detect_language_from_source(code) if language == "auto" else language
    if detected not in SERVED_LANGUAGES:
        raise PredictionError(
            f"auto-detected language {detected!r} is not yet served (Tier 3 -- "
            "parses, but has no trained model behind it in production)"
        )
    try:
        ir = normalize_source(code, detected)
    except UnsupportedLanguageError as e:
        raise PredictionError(str(e)) from e

    features = extract_features(ir)
    warnings: list[str] = []
    unknown_count = ir.symbol_histogram().get("UNKNOWN", 0)
    if ir.nodes and unknown_count / len(ir.nodes) > _UNKNOWN_SHARE_WARNING_THRESHOLD:
        warnings.append(
            "a large share of parsed nodes were unmapped (UNKNOWN) -- predictions may be unreliable"
        )

    graph = to_example_graph(ir)

    time_response, _ = _predict_dimension(registry, "time", graph.symbol_ids, graph.edges_by_kind)
    space_response, _ = _predict_dimension(registry, "space", graph.symbol_ids, graph.edges_by_kind)

    time_curve = growth_curve("time", str(time_response["class"]))
    space_curve = growth_curve("space", str(space_response["class"]))

    attribution = _attribution_for(registry, features, ir)

    return {
        "language_detected": detected,
        "time": time_response,
        "space": space_response,
        "attribution": attribution,
        "curve": {
            "n": time_curve["n"],
            "time": {
                "predicted_class": time_curve["predicted_class"],
                "series": time_curve["series"],
            },
            "space": {
                "predicted_class": space_curve["predicted_class"],
                "series": space_curve["series"],
            },
        },
        "ir": {
            "nodes": len(ir.nodes),
            "edges": len(ir.edges),
            "histogram": dict(ir.symbol_histogram()),
        },
        "warnings": warnings,
    }
