"""Core prediction pipeline (plan section 10's `POST /v1/predict`), a hybrid since PolyO v2 phase 5.

1. Parse the code to the IR (needed for the IR summary and for the ML model).
2. Run the symbolic engine (`analysis/`): it derives the cost from the loop and recursion structure
   instead of reading graph size the way the GNN did, and says how sure it is ("certain",
   "assumed" = it had to assume a bound, "unknown").
3. Certain or assumed: serve the engine's answer. Its `confidence` is its MEASURED accuracy at that
   certainty level (`analysis/calibration.py`), never a hard-coded 1.0, and the response carries the
   exact expression, the extended class, the assumptions and a line-numbered derivation.
4. Unknown (a recursion no static argument can bound) or an engine failure: ask the served GNN
   (numpy, no torch) exactly as before, with its calibrated distribution, conformal set and abstain
   rule, and say so in `warnings`.

The legacy response fields keep their shape; everything new is additive. Executes nothing (plan
section 10's hard invariant, enforced by `tests/test_isolation.py`) -- the engine reads code, it
never runs it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from analysis.calibration import SymbolicCalibration, load_calibration
from analysis.engine import Analysis, analyze
from api.attribution import top_contributions
from api.curve import growth_curve
from api.guards import MAX_CODE_BYTES
from api.models_registry import ModelRegistry
from core.ir import IRGraph
from core.taxonomy import SpaceClass, TimeClass, space_rank, time_rank
from features.tabular import FEATURE_NAMES, extract_features
from models.calibrate import Calibrator
from models.example_graph import to_example_graph
from parsing.normalize import normalize_source
from parsing.parse import UnsupportedLanguageError, detect_language_from_source

_logger = logging.getLogger("polyo.api")

_UNKNOWN_SHARE_WARNING_THRESHOLD = 0.2
# The served operating point: 90% marginal coverage (see models/conformal.py
# and PHASE5_REPORT.md's risk-coverage table for 0.05/0.20 too -- offline
# only, not served live, since a live API should have one predictable
# behaviour rather than a caller-tunable coverage knob nobody asked for).
_CONFORMAL_ALPHA = 0.10
# "Too uncertain to be useful" -- half or more of the 7-class time taxonomy
# (or all but one of space's 5) in the set stops being a legible answer.
_ABSTAIN_SET_SIZE = 4
# Which dimension's prediction the GBDT explains (plan section 10's response has
# one flat `attribution` list, not one per dimension) -- time is the
# project's headline metric (plan section 1's title order, and every other
# rung/report leads with it too).
_ATTRIBUTION_DIMENSION = "time"
# `hybrid` is what the API serves; `ml` is the GNN alone, kept so the research arm stays measurable.
_MODES = ("hybrid", "ml")
# How many derivation steps the code panel highlights, and how salient each kind is: loops and
# recursion decide the growth, calls and allocations support it (a heuristic, like the legacy
# field).
_MAX_ATTRIBUTION_ITEMS = 8
_SALIENCE = {"loop": 1.0, "recursion": 1.0, "call": 0.5, "alloc": 0.4}

# Tier 1/2 only (plan section 3) -- NOT `parsing.parse.SUPPORTED_LANGUAGES`, which
# also covers Tier 3 (TypeScript, Rust, C#, Kotlin: IR mapping file only,
# no oracle, added Phase 7). The GNN is language-agnostic by construction
# (it reads the IR, never source text) and would in principle score Tier
# 3 code too, but whether to actually serve that live is a deliberate,
# deferred product decision -- enforced here so it can't happen by
# accident through auto-detect (which internally scans every parseable
# language, Tier 3 included) even though an explicit `language: "rust"`
# request is already rejected one level up, in `api/main.py`.
SERVED_LANGUAGES: frozenset[str] = frozenset({"python", "cpp", "java", "javascript", "c", "go"})


class PredictionError(ValueError):
    """A request-level problem (bad language, code too large, unparseable)
    -- distinct from `ModelsNotTrainedError`, which is a deployment
    problem."""


def _rank_enum(dimension: str) -> type[TimeClass] | type[SpaceClass]:
    return TimeClass if dimension == "time" else SpaceClass


def _rank_fn(dimension: str):  # noqa: ANN201 -- returns one of two functions with different arg types
    return time_rank if dimension == "time" else space_rank


# ---------------------------------------------------------------------------------- ML path
def _predict_dimension(
    registry: ModelRegistry, dimension: str, symbol_ids, edges_by_kind, engine: str
) -> dict[str, object]:
    raw_scores = registry.gnn.forward(symbol_ids, edges_by_kind)[dimension]
    calibration = registry.calibration[dimension]
    calibrator = Calibrator(temperature=calibration.temperature, classes=calibration.classes)
    proba = calibrator.calibrate(raw_scores.reshape(1, -1))[0]
    pred_idx = int(proba.argmax())
    predicted_class = calibration.classes[pred_idx]

    class_proba_pairs = zip(calibration.classes, proba, strict=True)
    distribution = {cls: round(float(p), 4) for cls, p in class_proba_pairs}

    conformal_set = registry.conformal[dimension].predict_set(proba, alpha=_CONFORMAL_ALPHA)
    return {
        "class": predicted_class,
        "rank": _rank_fn(dimension)(_rank_enum(dimension)(predicted_class)),
        "confidence": round(float(proba[pred_idx]), 4),
        "distribution": distribution,
        "conformal_set": conformal_set,
        "conformal_coverage": 1.0 - _CONFORMAL_ALPHA,
        "abstain": len(conformal_set) >= _ABSTAIN_SET_SIZE,
        # the model gives a class and nothing more
        "engine": engine,
        "certainty": None,
        "expression": None,
        "extended_class": predicted_class,
        "projection_lossy": False,
    }


def _attribution_for(registry: ModelRegistry, features, ir):
    importance_weights = registry.feature_importance.get(_ATTRIBUTION_DIMENSION, {})
    feature_values = {name: features.as_dict()[name] for name in FEATURE_NAMES}
    return top_contributions(feature_values, importance_weights, registry.feature_scales, ir)


# ----------------------------------------------------------------------------- symbolic path
@lru_cache(maxsize=1)
def _default_calibration() -> SymbolicCalibration:
    """The committed calibration, for a registry built without one (tests)."""
    return load_calibration()


def _symbolic_dimension(
    analysis: Analysis, dimension: str, calibration: SymbolicCalibration
) -> dict[str, object]:
    result = analysis.time if dimension == "time" else analysis.space
    taxonomy = _rank_enum(dimension)
    classes = [c.value for c in taxonomy]
    confidence = round(calibration.confidence(analysis.certainty, dimension), 4)
    # The engine is deterministic: it has an answer and a track record, not a distribution. The
    # measured accuracy sits on the answer; the rest is spread evenly over the other classes (the
    # maximum-entropy choice, so no unmeasured claim about WHERE a miss would land).
    others = round((1.0 - confidence) / (len(classes) - 1), 4)
    distribution = {c: (confidence if c == result.cls else others) for c in classes}
    return {
        "class": result.cls,
        "rank": _rank_fn(dimension)(taxonomy(result.cls)),
        "confidence": confidence,
        "distribution": distribution,
        # a singleton set whose empirical coverage is the measured accuracy (not a conformal
        # guarantee: the engine is checked against labelled programs, not calibrated on them)
        "conformal_set": [result.cls],
        "conformal_coverage": confidence,
        "abstain": False,
        "engine": "symbolic",
        "certainty": analysis.certainty,
        "expression": result.text,
        "extended_class": result.ext_cls,
        "projection_lossy": result.lossy,
    }


def _derivation_attribution(analysis: Analysis, code: str) -> list[dict[str, object]]:
    """The legacy `attribution` field for a symbolic answer: the derivation steps, each pointing at
    the statement it explains. The code panel highlights `spans` (0-based line, column, line,
    column) and shows `feature` as the tooltip, so a user sees WHICH loop or call costs what."""
    lines = code.splitlines()
    scored: list[tuple[float, dict[str, object]]] = []
    for step in analysis.steps:
        if step.kind not in _SALIENCE or not 1 <= step.line <= len(lines):
            continue
        stripped = lines[step.line - 1].rstrip()
        first = len(stripped) - len(stripped.lstrip())
        if first >= len(stripped):
            continue
        salience = _SALIENCE[step.kind]
        item: dict[str, object] = {
            "feature": step.text,
            "contribution": salience,
            "spans": [[step.line - 1, first, step.line - 1, len(stripped)]],
        }
        scored.append((salience, item))
    scored.sort(key=lambda pair: -pair[0])  # stable: equal salience keeps source order
    return [item for _, item in scored[:_MAX_ATTRIBUTION_ITEMS]]


def _run_engine(code: str, language: str) -> tuple[Analysis | None, str | None]:
    """The engine's analysis, or why there is none. An engine bug must degrade to the model, never
    to a 500."""
    try:
        return analyze(code, language), None
    except Exception as error:  # noqa: BLE001 -- see above
        _logger.warning("symbolic engine failed: %s: %s", type(error).__name__, error)
        return None, f"{type(error).__name__}: {error}"


def _unknown_reason(analysis: Analysis) -> str:
    for note in analysis.notes:
        if note.kind == "unknown":
            where = f"line {note.line}: " if note.line else ""
            return f"{where}{note.reason}"
    return "no cost bound could be derived"


# ----------------------------------------------------------------------------------- response
def _curves(time_class: str, space_class: str) -> dict[str, object]:
    time_curve = growth_curve("time", time_class)
    space_curve = growth_curve("space", space_class)
    return {
        "n": time_curve["n"],
        "time": {
            "predicted_class": time_curve["predicted_class"],
            "series": time_curve["series"],
        },
        "space": {
            "predicted_class": space_curve["predicted_class"],
            "series": space_curve["series"],
        },
    }


def _ir_summary(ir: IRGraph) -> dict[str, object]:
    return {
        "nodes": len(ir.nodes),
        "edges": len(ir.edges),
        "histogram": dict(ir.symbol_histogram()),
    }


def predict(
    registry: ModelRegistry, code: str, language: str, *, mode: str = "hybrid"
) -> dict[str, object]:
    if mode not in _MODES:
        raise ValueError(f"unknown prediction mode {mode!r}; known: {list(_MODES)}")
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

    warnings: list[str] = []
    unknown_count = ir.symbol_histogram().get("UNKNOWN", 0)
    if ir.nodes and unknown_count / len(ir.nodes) > _UNKNOWN_SHARE_WARNING_THRESHOLD:
        warnings.append(
            "a large share of parsed nodes were unmapped (UNKNOWN) -- predictions may be unreliable"
        )

    engine_name = "ml"
    if mode == "hybrid":
        analysis, failure = _run_engine(code, detected)
        if analysis is not None and analysis.certainty != "unknown":
            return _symbolic_response(analysis, code, detected, ir, warnings, registry)
        reason = failure or (_unknown_reason(analysis) if analysis is not None else "")
        warnings.append(
            f"the symbolic engine could not bound this code ({reason}); the answer below comes "
            "from the ML fallback model, which is less reliable"
        )
        engine_name = "ml_fallback"
    return _ml_response(registry, code, detected, ir, warnings, engine_name)


def _symbolic_response(
    analysis: Analysis,
    code: str,
    detected: str,
    ir: IRGraph,
    warnings: list[str],
    registry: ModelRegistry,
) -> dict[str, object]:
    calibration = registry.symbolic or _default_calibration()
    time_part = _symbolic_dimension(analysis, "time", calibration)
    space_part = _symbolic_dimension(analysis, "space", calibration)
    if analysis.assumptions:
        warnings.append(
            f"the symbolic answer rests on {len(analysis.assumptions)} assumed bound(s) (see "
            "`assumptions`): it may over-estimate"
        )
    return {
        "language_detected": detected,
        "time": time_part,
        "space": space_part,
        "attribution": _derivation_attribution(analysis, code),
        "curve": _curves(str(time_part["class"]), str(space_part["class"])),
        "ir": _ir_summary(ir),
        "warnings": warnings,
        "engine": "symbolic",
        "entry": analysis.entry,
        "assumptions": [{"line": a.line, "reason": a.reason} for a in analysis.assumptions],
        "derivation": [{"line": s.line, "kind": s.kind, "text": s.text} for s in analysis.steps],
    }


def _ml_response(
    registry: ModelRegistry,
    code: str,
    detected: str,
    ir: IRGraph,
    warnings: list[str],
    engine: str,
) -> dict[str, object]:
    features = extract_features(ir)
    graph = to_example_graph(ir)
    time_part = _predict_dimension(registry, "time", graph.symbol_ids, graph.edges_by_kind, engine)
    space_part = _predict_dimension(
        registry, "space", graph.symbol_ids, graph.edges_by_kind, engine
    )
    return {
        "language_detected": detected,
        "time": time_part,
        "space": space_part,
        "attribution": _attribution_for(registry, features, ir),
        "curve": _curves(str(time_part["class"]), str(space_part["class"])),
        "ir": _ir_summary(ir),
        "warnings": warnings,
        "engine": engine,
        "entry": None,
        "assumptions": [],
        "derivation": [],
    }
