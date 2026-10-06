"""One interface for every complexity predictor the scorecard compares.

A predictor is `(source, language) -> Prediction`. It never raises: a failure is recorded in
`Prediction.error` so one bad snippet is a counted miss rather than a crash that hides the rest of
the suite. Predictors are registered by name; later phases add `symbolic` and `hybrid` here.

Why a rule predictor sits next to the shipped GNN: on the cases the GNN got wrong (C++/Java/JS
nested loops) a ten-line lookup on loop-nesting depth was right every time. Any new engine has to
be compared against that floor, not just against the model it replaces.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from features.tabular import extract_features
from models.rule import predict_space, predict_time
from parsing.normalize import normalize_source


@dataclass(frozen=True, slots=True)
class Prediction:
    time_class: str | None
    space_class: str | None
    time_expr: str | None = None
    space_expr: str | None = None
    certainty: str | None = None
    engine: str = ""
    error: str | None = None


Predictor = Callable[[str, str], Prediction]


def _failed(engine: str, error: Exception) -> Prediction:
    return Prediction(None, None, engine=engine, error=f"{type(error).__name__}: {error}")


def _rule(source: str, language: str) -> Prediction:
    try:
        features = extract_features(normalize_source(source, language))
        return Prediction(
            time_class=predict_time(features).value,
            space_class=predict_space(features).value,
            engine="rule",
        )
    except Exception as error:  # noqa: BLE001 -- the contract is "never raises"
        return _failed("rule", error)


@lru_cache(maxsize=1)
def _registry():  # noqa: ANN202 -- ModelRegistry, imported lazily to keep `rule` torch/numpy-free
    """The served artifacts, or the directory `POLYO_ARTIFACTS_DIR` names: a retrained candidate is
    compared against the served model (`eval/compare.py`) before it replaces it."""
    from api.models_registry import load_registry

    candidate = os.environ.get("POLYO_ARTIFACTS_DIR")
    return load_registry(Path(candidate)) if candidate else load_registry()


def _from_api(engine: str, source: str, language: str, mode: str = "hybrid") -> Prediction:
    try:
        from api.predict import predict

        response = predict(_registry(), source, language, mode=mode)
        time_part = response["time"]
        space_part = response["space"]
        assert isinstance(time_part, dict) and isinstance(space_part, dict)
        return Prediction(
            time_class=str(time_part["class"]),
            space_class=str(space_part["class"]),
            time_expr=_optional_str(time_part.get("expression")),
            space_expr=_optional_str(space_part.get("expression")),
            certainty=_optional_str(time_part.get("certainty")),
            engine=_optional_str(time_part.get("engine")) or engine,
        )
    except Exception as error:  # noqa: BLE001 -- the contract is "never raises"
        return _failed(engine, error)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _gnn(source: str, language: str) -> Prediction:
    """The shipped ML model on its own: the GNN with no engine in front of it, so the research arm
    stays measurable now that `product` (the API) is a hybrid."""
    return _from_api("gnn", source, language, mode="ml")


def _product(source: str, language: str) -> Prediction:
    """Exactly what `POST /v1/predict` answers: the thing users actually see (the hybrid)."""
    return _from_api("product", source, language)


def _symbolic(source: str, language: str) -> Prediction:
    """The v2 symbolic engine on its own: no model, no fallback."""
    try:
        from analysis.engine import analyze

        result = analyze(source, language)
        return Prediction(
            time_class=result.time.cls,
            space_class=result.space.cls,
            time_expr=result.time.text,
            space_expr=result.space.text,
            certainty=result.certainty,
            engine="symbolic",
        )
    except Exception as error:  # noqa: BLE001 -- the contract is "never raises"
        return _failed("symbolic", error)


_REGISTRY: dict[str, Predictor] = {
    "rule": _rule,
    "gnn": _gnn,
    "product": _product,
    "symbolic": _symbolic,
}


def register_predictor(name: str, predictor: Predictor) -> None:
    _REGISTRY[name] = predictor


def available_predictors() -> list[str]:
    return sorted(_REGISTRY)


def get_predictor(name: str) -> Predictor:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown predictor {name!r}; known: {available_predictors()}") from None
