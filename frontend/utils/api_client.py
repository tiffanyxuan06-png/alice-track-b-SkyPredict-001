"""Client for the SkyPredict FastAPI backend.

This is the *only* place the dashboard talks to the backend — pages and
components call these functions rather than issuing HTTP or running any
prediction logic themselves (per the project guidelines). Read-only lookups
are cached; every call raises ``APIError`` with a human-readable message on
failure so pages can show a clean error instead of a traceback.

Model-backed calls accept an optional ``model`` key ("rf"/"xgb") passed to the
backend as ``?model=``; when omitted the backend uses its configured default.
"""

from __future__ import annotations

import requests
import streamlit as st

from utils.config import API_BASE_URL, REQUEST_TIMEOUT


class APIError(Exception):
    """Raised when the backend is unreachable or returns an error."""


def _format_validation_error(body: dict) -> str:
    errors = body.get("errors", [])
    if not errors:
        return body.get("detail", "Invalid input.")
    lines = [f"- {e.get('field', '?')}: {e.get('message', '')}" for e in errors]
    return "Invalid input:\n" + "\n".join(lines)


def _model_params(model: str | None) -> dict | None:
    return {"model": model} if model else None


def _get(path: str, params: dict | None = None) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise APIError(f"Could not reach the backend at {API_BASE_URL}. ({exc})")


def _post(path: str, payload: dict, params: dict | None = None) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE_URL}{path}", json=payload, params=params, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise APIError(f"Could not reach the backend at {API_BASE_URL}. ({exc})")
    if resp.status_code == 422:
        raise APIError(_format_validation_error(resp.json()))
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise APIError(str(exc))
    return resp.json()


# --- Health (never cached; must reflect current state) ----------------------
def health() -> dict:
    return _get("/health")


# --- Read-only lookups (cached per model; change only on re-export) ---------
@st.cache_data(ttl=60, show_spinner=False)
def model_info(model: str | None = None) -> dict:
    return _get("/model-info", _model_params(model))


@st.cache_data(ttl=60, show_spinner=False)
def global_importances(model: str | None = None) -> list[dict]:
    return _get("/explain/global", _model_params(model))


@st.cache_data(ttl=60, show_spinner=False)
def permutation_importances(model: str | None = None) -> list[dict]:
    return _get("/explain/permutation-importance", _model_params(model))


@st.cache_data(ttl=60, show_spinner=False)
def partial_dependence(model: str | None = None, feature: str | None = None) -> list[dict]:
    params: dict = {}
    if model:
        params["model"] = model
    if feature:
        params["feature"] = feature
    return _get("/explain/partial-dependence", params or None)


@st.cache_data(ttl=300, show_spinner=False)
def example_reading() -> dict:
    """A realistic sample payload, taken from the backend's OpenAPI example."""
    spec = _get("/openapi.json")
    schema = spec.get("components", {}).get("schemas", {}).get("EngineReading", {})
    return schema.get("example", {})


# --- Predictions ------------------------------------------------------------
def predict(reading: dict, model: str | None = None) -> dict:
    return _post("/predict", reading, _model_params(model))


def explain(reading: dict, model: str | None = None) -> dict:
    return _post("/explain", reading, _model_params(model))


def predict_batch(readings: list[dict], model: str | None = None) -> dict:
    return _post("/predict/batch", {"readings": readings}, _model_params(model))


def prioritize(readings: list[dict], model: str | None = None) -> dict:
    return _post("/prioritize", {"readings": readings}, _model_params(model))
