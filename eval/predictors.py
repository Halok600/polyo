"""One interface for every complexity predictor the scorecard compares.

A predictor is `(source, language) -> Prediction`. It never raises: a failure is recorded in
`Prediction.error` so one bad snippet is a counted miss rather than a crash that hides the rest of
the suite. Predictors are registered by name; later phases add `symbolic` and `hybrid` here.

Why a rule predictor sits next to the shipped GNN: on the cases the GNN got wrong (C++/Java/JS
nested loops) a ten-line lookup on loop-nesting depth was right every time. Any new engine has to
be compared against that floor, not just against the model it replaces.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

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
    from api.models_registry import load_registry

    return load_registry()


def _from_api(engine: str, source: str, language: str) -> Prediction:
    try:
        from api.predict import predict

        response = predict(_registry(), source, language)
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
    """The shipped ML model on its own. (Until the API becomes a hybrid this is the same path as
    `product`; the two are separate names so the ML arm stays measurable afterwards.)"""
    return _from_api("gnn", source, language)


def _product(source: str, language: str) -> Prediction:
    """Exactly what `POST /v1/predict` answers: the thing users actually see."""
    return _from_api("product", source, language)


_REGISTRY: dict[str, Predictor] = {"rule": _rule, "gnn": _gnn, "product": _product}


def register_predictor(name: str, predictor: Predictor) -> None:
    _REGISTRY[name] = predictor


def available_predictors() -> list[str]:
    return sorted(_REGISTRY)


def get_predictor(name: str) -> Predictor:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown predictor {name!r}; known: {available_predictors()}") from None
