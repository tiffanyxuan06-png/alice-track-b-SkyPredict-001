"""Health and model-info endpoints."""

from __future__ import annotations


def test_root(client):
    body = client.get("/").json()
    assert "docs" in body


def test_health_reports_loaded_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"rf", "xgb"}
    assert body["default_model"] in body["models_loaded"]
    assert body["app_version"]


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_key"] == "xgb"  # the default
    assert info["target"] == "RUL"
    assert info["n_features"] == len(info["feature_names"]) == 24
    assert set(info["risk_thresholds"]) == {"high_below", "medium_below"}
    assert {"rmse", "mae", "r2"} <= set(info["metrics"]["test"])


def test_model_info_selects_model(client):
    assert client.get("/model-info", params={"model": "rf"}).json()["model_key"] == "rf"
    assert client.get("/model-info", params={"model": "xgb"}).json()["model_key"] == "xgb"
